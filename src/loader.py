# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# load_framework.py


import glob
import os
import PIL
import random
import warnings
from os.path import dirname, abspath, exists, join

from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.prepare_inception_moments import prepare_inception_moments
from utils.log import make_run_name, make_logger, make_checkpoint_dir
from utils.sample import latent_sampler
from utils.losses import *
from utils.load_checkpoint import load_verify_checkpoint
from utils.misc import *
from utils.biggan_utils import ema_
from sync_batchnorm.batchnorm import convert_model
from worker import Worker

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter



RUN_NAME_FORMAT = (
    "{framework}-"
    "{phase}-"
    "{timestamp}"
)


def load_setup(cfgs, hdf5_path_train, **_):
    if cfgs.seed == 82624:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        fix_all_seed(cfgs.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if cfgs.disable_debugging_API:
        torch.autograd.set_detect_anomaly(False)

    n_gpus = torch.cuda.device_count()
    default_device = torch.cuda.current_device()

    check_flag_0(cfgs.batch_size, n_gpus, cfgs.standing_statistics, cfgs.ema, cfgs.freeze_layers, cfgs.checkpoint_folder)
    assert cfgs.batch_size % n_gpus == 0, "batch_size should be divided by the number of gpus "

    if n_gpus == 1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None
    standing_step = cfgs.standing_step if cfgs.standing_statistics is True else cfgs.batch_size

    run_name = make_run_name(RUN_NAME_FORMAT,
                             framework=cfgs.config_path.split('/')[3][:-5],
                             phase='train')

    logger = make_logger(run_name, None)
    writer = SummaryWriter(log_dir=join('./logs', run_name))
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(cfgs.train_configs)
    logger.info(cfgs.model_configs)

    logger.info('Loading train datasets...')
    train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, resize_size=cfgs.img_size,
                                hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing)
    if cfgs.reduce_train_dataset < 1.0:
        num_train = int(cfgs.reduce_train_dataset*len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Loading {mode} datasets...'.format(mode=cfgs.eval_type))
    eval_mode = True if cfgs.eval_type == 'train' else False
    eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                               hdf5_path=None, random_flip=False)
    logger.info('Eval dataset size : {dataset_size}'.format(dataset_size=len(eval_dataset)))

    logger.info('Building model...')
    if cfgs.architecture == "dcgan":
        assert cfgs.img_size == 32, "Sry, StudioGAN does not support dcgan models for generation of images larger than 32 resolution."
    module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
    logger.info('Modules are located on models.{architecture}'.format(architecture=cfgs.architecture))
    Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                           cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                           cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision).to(default_device)

    Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
                               cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
                               cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(default_device)

    if cfgs.ema:
        print('Preparing EMA for G with decay of {}'.format(cfgs.ema_decay))
        Gen_copy = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                                    cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                                    initialize=False, G_depth=cfgs.G_depth, mixed_precision=cfgs.mixed_precision).to(default_device)
        Gen_ema = ema_(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    logger.info(count_parameters(Gen))
    logger.info(Gen)

    logger.info(count_parameters(Dis))
    logger.info(Dis)

    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=cfgs.num_workers, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=cfgs.num_workers, drop_last=False)

    lt_sampler = latent_sampler(cfgs.prior, cfgs.z_dim, cfgs.num_classes, default_device)
    grad_reg = gradient_regularizer(Gen, Dis, cfgs.conditional_strategy, cfgs.latent_op_rate, cfgs.latent_op_alpha, cfgs.latent_op_beta, default_device)

    G_loss = {'vanilla': loss_dcgan_gen, 'least_square': loss_lsgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
    D_loss = {'vanilla': loss_dcgan_dis, 'least_square': loss_lsgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}

    if cfgs.optimizer == "SGD":
        G_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
    elif cfgs.optimizer == "RMSprop":
        G_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
        D_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
    elif cfgs.optimizer == "Adam":
        G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
        D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
    else:
        raise NotImplementedError

    if cfgs.checkpoint_folder is not None:
        checkpoint_dir, writer, prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path =\
            load_verify_checkpoint(cfgs, run_name, Gen, G_optimizer, Dis, D_optimizer, Gen_copy, Gen_ema)
    else:
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)

    if n_gpus > 1:
        Gen = DataParallel(Gen, output_device=default_device)
        Dis = DataParallel(Dis, output_device=default_device)
        if cfgs.ema:
            Gen_copy = DataParallel(Gen_copy, output_device=default_device)

        if cfgs.synchronized_bn:
            Gen = convert_model(Gen).to(default_device)
            Dis = convert_model(Dis).to(default_device)
            if cfgs.ema:
                Gen_copy = convert_model(Gen_copy).to(default_device)

    if cfgs.eval:
        inception_model = InceptionV3().to(default_device)
        if n_gpus > 1:
            inception_model = DataParallel(inception_model, output_device=default_device)
        mu, sigma = prepare_inception_moments(dataloader=eval_dataloader,
                                              eval_mode=cfgs.eval_type,
                                              gen=Gen,
                                              inception_model=inception_model,
                                              lt_sampler=lt_sampler,
                                              grad_reg=grad_reg,
                                              splits=1,
                                              run_name=run_name,
                                              logger=logger,
                                              device=default_device)
    else:
        mu, sigma, inception_model = None, None, None


    worker = Worker(
        cfgs=cfgs,
        run_name=run_name,
        best_step=best_step,
        logger=logger,
        writer=writer,
        n_gpus=n_gpus,
        gen_model=Gen,
        dis_model=Dis,
        inception_model=inception_model,
        Gen_copy=Gen_copy,
        Gen_ema=Gen_ema,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        lt_sampler=lt_sampler,
        grad_reg=grad_reg,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        G_loss=G_loss[cfgs.adv_loss],
        D_loss=D_loss[cfgs.adv_loss],
        prev_ada_p=prev_ada_p,
        checkpoint_dir=checkpoint_dir,
        mu=mu,
        sigma=sigma,
        best_fid=best_fid,
        best_fid_checkpoint_path=best_fid_checkpoint_path,
        default_device=default_device,
    )

    attributes = [attr for attr in dir(cfgs) if not attr.startswith('__')]
    for a in attributes: setattr(worker, a, getattr(cfgs, a))
    worker.prepare_worker()

    if cfgs.train_configs['train']:
        step = worker.train(current_step=step, total_step=cfgs.total_step)

    if cfgs.train_configs['eval']:
        is_save = worker.evaluate(step=step)

    if cfgs.save_images:
        worker.save_images(is_generate=True, png=True, npz=True)

    if cfgs.image_visualization:
        worker.run_image_visualization(nrow=cfgs.nrow, ncol=cfgs.ncol)

    if cfgs.k_nearest_neighbor:
        worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol)

    if cfgs.interpolation:
        assert cfgs.architecture in ["big_resnet", "biggan_deep"], "Not supported except for biggan and biggan_deep."
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False)
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True)

    if cfgs.frequency_analysis:
        worker.run_frequency_analysis(num_images=len(train_dataset)//cfgs.num_classes)

# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/load_checkpoint.py



import glob
import os
from os.path import dirname, abspath, exists, join
from utils.log import make_run_name, make_logger, make_checkpoint_dir

import torch
from torch.utils.tensorboard import SummaryWriter



def load_checkpoint(model, optimizer, filename, metric=False, ema=False):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_step = 0
    if ema:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        checkpoint = torch.load(filename)
        seed = checkpoint['seed']
        run_name = checkpoint['run_name']
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ada_p = checkpoint['ada_p']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if metric:
            best_step = checkpoint['best_step']
            best_fid = checkpoint['best_fid']
            best_fid_checkpoint_path = checkpoint['best_fid_checkpoint_path']
            return model, optimizer, seed, run_name, start_step, ada_p, best_step, best_fid, best_fid_checkpoint_path
    return model, optimizer, seed, run_name, start_step, ada_p


def load_verify_checkpoint(cfgs, run_name, Gen, G_optimizer, Dis, D_optimizer, Gen_copy, Gen_ema, ):
    when = "current" if cfgs.load_current is True else "best"
    if not exists(abspath(cfgs.checkpoint_folder)):
        raise NotADirectoryError

    checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
    g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
    d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]

    Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
    Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
        load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)

    logger = make_logger(run_name, None)
    if cfgs.ema:
        g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
        Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
        Gen_ema.source, Gen_ema.target = Gen, Gen_copy

    writer = SummaryWriter(log_dir=join('./logs', run_name))
    if cfgs.train_configs['train']:
        assert cfgs.seed == trained_seed, "seed for sampling random numbers should be same!"
    logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
    logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))

    if cfgs.freeze_layers > -1:
        prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None
    return checkpoint_dir, writer, prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path


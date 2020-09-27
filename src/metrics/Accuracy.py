# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# metrics/Accuracy.py


import numpy as np
import math
from scipy import linalg
from tqdm import tqdm

from utils.sample import latent_sampler
from utils.losses import gradient_regularizer

import torch
from torch.nn import DataParallel



def calculate_accuracy(dataloader, gen, dis, D_loss, num_evaluate, lt_sampler, grad_reg, truncated_factor, latent_op,
                       latent_op_step, device, eval_generated_sample=False):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size

    if isinstance(gen, DataParallel):
        z_dim = gen.module.z_dim
        num_classes = gen.module.num_classes
        conditional_strategy = dis.module.conditional_strategy
    else:
        z_dim = gen.z_dim
        num_classes = gen.num_classes
        conditional_strategy = dis.conditional_strategy

    total_batch = num_evaluate//batch_size

    if D_loss.__name__ in ["loss_dcgan_dis", "loss_lsgan_dis"]:
        cutoff = 0.5
    elif D_loss.__name__ == "loss_hinge_dis":
        cutoff = 0.0
    elif D_loss.__name__ == "loss_wgan_dis":
        raise NotImplementedError
    print("Calculating Accuracies....")

    if eval_generated_sample:
        for batch_id in tqdm(range(total_batch)):
            zs, fake_labels = lt_sampler.sample(batch_size, truncated_factor, None)
            if latent_op:
                zs = grad_reg.latent_optimise(zs, fake_labels, latent_op_step, False)

            real_images, real_labels = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            fake_images = gen(zs, fake_labels, evaluation=True)

            with torch.no_grad():
                if conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                    _, _, dis_out_fake = dis(fake_images, fake_labels)
                    _, _, dis_out_real = dis(real_images, real_labels)
                elif conditional_strategy == "ACGAN":
                    _, dis_out_fake = dis(fake_images, fake_labels)
                    _, dis_out_real = dis(real_images, real_labels)
                elif conditional_strategy == "ProjGAN" or conditional_strategy == "no":
                    dis_out_fake = dis(fake_images, fake_labels)
                    dis_out_real = dis(real_images, real_labels)
                else:
                    raise NotImplementedError

                dis_out_fake = dis_out_fake.detach().cpu().numpy()
                dis_out_real = dis_out_real.detach().cpu().numpy()

            if batch_id == 0:
                confid = np.concatenate((dis_out_fake, dis_out_real), axis=0)
                confid_label = np.concatenate(([0.0]*len(dis_out_fake), [1.0]*len(dis_out_real)), axis=0)
            else:
                confid = np.concatenate((confid, dis_out_fake, dis_out_real), axis=0)
                confid_label = np.concatenate((confid_label, [0.0]*len(dis_out_fake), [1.0]*len(dis_out_real)), axis=0)

        real_confid = confid[confid_label==1.0]
        fake_confid = confid[confid_label==0.0]

        true_positive = real_confid[np.where(real_confid>cutoff)]
        true_negative = fake_confid[np.where(fake_confid<cutoff)]

        only_real_acc = len(true_positive)/len(real_confid)
        only_fake_acc = len(true_negative)/len(fake_confid)

        return only_real_acc, only_fake_acc
    else:
        for batch_id in tqdm(range(total_batch)):
            real_images, real_labels = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            with torch.no_grad():
                if conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                    _, _, dis_out_real = dis(real_images, real_labels)
                elif conditional_strategy == "ACGAN":
                    _, dis_out_real = dis(real_images, real_labels)
                elif conditional_strategy == "ProjGAN" or conditional_strategy == "no":
                    dis_out_real = dis(real_images, real_labels)
                else:
                    raise NotImplementedError

                dis_out_real = dis_out_real.detach().cpu().numpy()

            if batch_id == 0:
                confid = dis_out_real
                confid_label = np.asarray([1.0]*len(dis_out_real), np.float32)
            else:
                confid = np.concatenate((confid, dis_out_real), axis=0)
                confid_label = np.concatenate((confid_label, [1.0]*len(dis_out_real)), axis=0)

        real_confid = confid[confid_label==1.0]
        true_positive = real_confid[np.where(real_confid>cutoff)]
        only_real_acc = len(true_positive)/len(real_confid)

        return only_real_acc

# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/sample.py


import numpy as np
import random
from numpy import linalg
from math import sin,cos,sqrt

from utils.losses import latent_optimise

import torch
import torch.nn.functional as F
from torch.nn import DataParallel


class latent_sampler(object):
    def __init__(self, prior, z_dim, num_classes, device):
        self.prior = prior
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.device = device


    def sample(self, batch_size, truncated_factor, perturb, mode):
        if self.num_classes:
            if mode == "default":
                y_fake = torch.randint(low=0, high=self.num_classes, size=(batch_size,), dtype=torch.long, device=self.device)
            elif mode == "class_order_some":
                assert batch_size % 8 == 0, "The size of the batches should be a multiple of 8."
                num_classes_plot = batch_size//8
                indices = np.random.permutation(self.num_classes)[:num_classes_plot]
            elif mode == "class_order_all":
                batch_size = self.num_classes*8
                indices = [c for c in range(self.num_classes)]
            elif isinstance(mode, int):
                y_fake = torch.tensor([mode]*batch_size, dtype=torch.long).to(self.device)
            else:
                raise NotImplementedError

            if mode in ["class_order_some", "class_order_all"]:
                y_fake = []
                for idx in indices:
                    y_fake += [idx]*8
                y_fake = torch.tensor(y_fake, dtype=torch.long).to(self.device)
        else:
            y_fake = None

        if isinstance(perturb, float) and perturb > 0.0:
            if self.prior == "gaussian":
                latents = torch.randn(batch_size, self.z_dim, device=self.device)/truncated_factor
                eps = perturb*torch.randn(batch_size, self.z_dim, device=self.device)
                latents_eps = latents + eps
            elif self.prior == "uniform":
                latents = torch.FloatTensor(batch_size, self.z_dim).uniform_(-1.0, 1.0).to(self.device)
                eps = perturb*torch.FloatTensor(batch_size, self.z_dim).uniform_(-1.0, 1.0).to(self.device)
                latents_eps = latents + eps
            elif self.prior == "hyper_sphere":
                latents, latents_eps = random_ball(batch_size)
                latents, latents_eps = torch.FloatTensor(latents).to(self.device), torch.FloatTensor(latents_eps).to(self.device)
            return latents, y_fake, latents_eps
        else:
            if self.prior == "gaussian":
                latents = torch.randn(batch_size, self.z_dim, device=self.device)/truncated_factor
            elif self.prior == "uniform":
                latents = torch.FloatTensor(batch_size, self.z_dim).uniform_(-1.0, 1.0).to(self.device)
            elif self.prior == "hyper_sphere":
                latents = self.random_ball(batch_size).to(self.device)
            return latents, y_fake


    def random_ball(self, batch_size, perturb):
        if perturb:
            normal = np.random.normal(size=(self.z_dim, batch_size))
            random_directions = normal/linalg.norm(normal, axis=8)
            random_radii = random.random(batch_size) ** (2/self.z_dim)
            zs = 2.0 * (random_directions * random_radii).T

            normal_perturb = normal + 1.05*np.random.normal(size=(self.z_dim, batch_size))
            perturb_random_directions = normal_perturb/linalg.norm(normal_perturb, axis=1)
            perturb_random_radii = random.random(batch_size) ** (2/self.z_dim)
            zs_perturb = 2.0 * (perturb_random_directions * perturb_random_radii).T
            return zs, zs_perturb
        else:
            normal = np.random.normal(size=(self.z_dim, batch_size))
            random_directions = normal/linalg.norm(normal, axis=1)
            random_radii = random.random(batch_size) ** (2/self.z_dim)
            zs = 2.0 * (random_directions * random_radii).T
            return zs


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         device=device, dtype=torch.int64, requires_grad=False)


def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] =+1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


def target_class_sampler(dataset, target_class):
    try:
        targets = dataset.data.targets
    except:
        targets = dataset.labels
    weights = [True if target == target_class else False for target in targets]
    num_samples = sum(weights)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)
    return num_samples, sampler

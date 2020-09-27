# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# metrics/prepare_inception_moments.py

import numpy as np
import os

from metrics.FID import calculate_activation_statistics
from metrics.IS import evaluator



def prepare_inception_moments(dataloader, eval_mode, gen, inception_model, lt_sampler, grad_reg, splits, run_name, logger, device):
    dataset_name = dataloader.dataset.dataset_name
    inception_model.eval()

    save_path = os.path.abspath(os.path.join("./data", dataset_name + "_" + eval_mode +'_' + 'inception_moments.npz'))
    is_file = os.path.isfile(save_path)

    if is_file:
        mu = np.load(save_path)['mu']
        sigma = np.load(save_path)['sigma']
    else:
        logger.info('Calculate moments of {} dataset'.format(eval_mode))
        mu, sigma = calculate_activation_statistics(data_loader=dataloader,
                                                    gen=gen,
                                                    inception_model=inception_model,
                                                    num_generate=None,
                                                    lt_sampler=lt_sampler,
                                                    grad_reg=grad_reg,
                                                    truncated_factor=None,
                                                    is_generate=False,
                                                    latent_op=False,
                                                    latent_op_step=None,
                                                    device=device,
                                                    tqdm_disable=False,
                                                    run_name=run_name)

        logger.info('Saving calculated means and covariances to disk...')
        np.savez(save_path, **{'mu': mu, 'sigma': sigma})

    if is_file:
        pass
    else:
        logger.info('calculate inception score of {} dataset'.format(eval_mode))
        evaluator_instance = evaluator(inception_model, device=device)
        is_score, is_std = evaluator_instance.eval_dataset(dataloader, splits=splits)
        logger.info('Inception score={is_score}-Inception_std={is_std}'.format(is_score=is_score, is_std=is_std))
    return mu, sigma

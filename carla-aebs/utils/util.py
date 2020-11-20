import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from tqdm import tqdm

from datasets.perception import PerceptionDataset
from models.icad.vae import VAE


_SUPPORTED_DEVICES = ['cpu', 'gpu']


def configure_device(device):
    if device not in _SUPPORTED_DEVICES:
        raise NotImplementedError(f'The device type `{device}` is not supported')

    if device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception('CUDA support is not available on your platform. Re-run using CPU or TPU mode')
        return 'cuda'
    return 'cpu'


def get_loss(name):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    else:
        raise NotImplementedError(f'The loss {name} has not been implemented yet!')


def get_optimizer(name, net, lr, **kwargs):
    optim_cls = getattr(optim, name, None)
    if optim_cls is None:
        raise ValueError(
            f"""The optimizer {name} is not supported by torch.optim.
            Refer to https://pytorch.org/docs/stable/optim.html#algorithms
            for an overview of the algorithms supported"""
        )
    return optim_cls(
        [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': lr }],
        lr=lr, **kwargs
    )


def get_lr_scheduler(optimizer, num_epochs, sched_type='poly', **kwargs):
    if sched_type == 'poly':
        # A poly learning rate scheduler
        lambda_fn = lambda i: pow((1 - i / num_epochs), 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    elif sched_type == 'cosine':
        # Cosine learning rate annealing with Warm restarts
        T_0 = kwargs['t0']
        T_mul = kwargs.get('tmul', 1)
        eta_min = 0
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=T_mul, eta_min=eta_min
        )
    else:
        raise ValueError(f'The lr_scheduler type {sched_type} has not been implemented yet')


def compute_calibration_scores(data_dir, vae_chkpt_path, save_path=os.getcwd()):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    dataset = PerceptionDataset(data_dir, mode='val', transform=transform, seed=0)
    non_conformity_scores = []
    state_dict = torch.load(vae_chkpt_path)
    vae = VAE()
    vae.load_state_dict(state_dict['model'])
    for data, _ in tqdm(dataset):
        data = data.unsqueeze(0)
        non_conformity_score = vae.get_non_conformity_score(data)
        non_conformity_scores.append(non_conformity_score)

    save_path = os.path.join(save_path, 'calibration')
    np.save(save_path, non_conformity_scores)
    print(f'Generated calibration scores at: {save_path}')

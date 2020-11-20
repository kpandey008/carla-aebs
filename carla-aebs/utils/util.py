import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    return loss


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


def predict_distance(image_batch, chkpt_path):
    state_dict = torch.load(chkpt_path)
    net = PerceptionNet()
    net.eval()
    net.load_state_dict(state_dict['model'])
    with torch.no_grad():
        dist = net(image_batch)
    return dist

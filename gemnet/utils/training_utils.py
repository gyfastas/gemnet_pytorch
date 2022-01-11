"""
utils during training. 

For example, gradient clipping.

"""
import torch

@staticmethod
def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

@staticmethod
def adaptive_gradient_clipping(parameters, clip_factor=0.05, eps=1e-3, norm_type=2.0):
    """
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py

    Adapted from High-Performance Large-Scale Image Recognition Without Normalization:
    https://github.com/deepmind/deepmind-research/blob/master/nfnets/optim.py"""
    with torch.no_grad():
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            if p.grad is None:
                continue
            p_data = p
            g_data = p.grad
            max_norm = (
                unitwise_norm(p_data, norm_type=norm_type)
                .clamp_(min=eps)
                .mul_(clip_factor)
            )
            grad_norm = unitwise_norm(g_data, norm_type=norm_type)
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
            new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
            p.grad.copy_(new_grads)
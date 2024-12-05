import torch
import torch.nn.functional as F
import math
from models.utils import multi_vmf

def criterion(output, target):
    """
    Computes the mean squared error loss.

    Parameters:
    - output (torch.Tensor): Output tensor of the model.
    - target (torch.Tensor): Target tensor.

    Returns:
    - torch.Tensor: Loss value.
    """
    return F.mse_loss(output, target)

def negative_log_likelihood(rawdata, weight, mu, kappa, **kwargs):
    """
    Computes the negative log-likelihood loss.

    Parameters:
    - rawdata (torch.Tensor): Raw data tensor. Shape: (bz, data_sizes, 3)
    - weight (torch.Tensor): Weights of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 1)
    - mu (torch.Tensor): Axes of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 3)
    - kappa (torch.Tensor): Concentration parameters of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 1)
    - kwargs (dict): Additional keyword arguments.

    Returns:
    - torch.Tensor: Loss value.
    """



    kl_lambda = kwargs['kl_lambda'] 
    l1_lambda = kwargs['l1_lambda']
    l2_lambda = kwargs['l2_lambda']
    p = kwargs['p']
    w = kwargs['w']

    total_prob = multi_vmf(weight, mu, kappa, rawdata) # Shape: (bz, data_sizes)
    if kl_lambda > 0:
        nll = -torch.log(total_prob + 1e-10)
        nll = torch.mean(nll)
    else:
        nll = torch.tensor(0.0)

    q = multi_vmf(weight, mu, kappa, w) # Shape: (bz, data_sizes)
    if l1_lambda > 0:
        rec_loss = torch.mean(torch.abs(q - p))
    else:
        rec_loss = torch.tensor(0.0)

    if l2_lambda > 0:
        l2_loss = torch.norm(q - p, 2).mean()
    else:
        l2_loss = torch.tensor(0.0)

    loss = kl_lambda * nll + l1_lambda * rec_loss + l2_lambda * l2_loss
    # loss = torch.mean(loss)
    return loss, {'NLL': nll, 'Rec': rec_loss, 'L2': l2_loss}
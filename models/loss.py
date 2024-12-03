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

    kl_lambda = kwargs['kl_lambda'] 
    l1_lambda = kwargs['l1_lambda']
    l2_lambda = kwargs['l2_lambda']
    p = kwargs['p']
    w = kwargs['w']

    total_prob = multi_vmf(weight, mu, kappa, rawdata)
    if kl_lambda > 0:
        nll = -torch.log(total_prob + 1e-10).mean()
    else:
        nll = torch.tensor(0.0)

    q = multi_vmf(weight, mu, kappa, w)
    if l1_lambda > 0:
        rec_loss = torch.abs(q - p).mean()
    else:
        rec_loss = torch.tensor(0.0)

    if l2_lambda > 0:
        l2_loss = torch.norm(q - p, p=2).mean()
    else:
        l2_loss = torch.tensor(0.0)

    loss = kl_lambda * nll + l1_lambda * rec_loss + l2_lambda * l2_loss
    return loss, {'NLL': nll, 'Rec': rec_loss, 'L2': l2_loss}
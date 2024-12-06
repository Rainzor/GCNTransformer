
# Install required packages.
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
# Helper function for visualization.
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import json

    
# Multi von Mises-Fisher Distribution
def multi_vmf(weights, axes, kappas, w):
    """
    Computes the probability density function of a multi von Mises-Fisher distribution.

    Parameters:
    - weights (torch.Tensor): Weights of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 1) or (num_spheres, 1)
    - axes (torch.Tensor): Axes of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 3) or (num_spheres, 3)
    - kappas (torch.Tensor): Concentration parameters of the von Mises-Fisher distributions. Shape: (bz, num_spheres, 1) or (num_spheres, 1)
    - w (torch.Tensor): Input data. Shape: (bz, data_sizes, 3), dtype: torch.float32 or torch.float16

    Returns:
    - torch.Tensor: Probability density function values. Shape: (bz, data_sizes) or (data_sizes) if bz=1
    """

    # Define thresholds for approximations
    large_kappa_threshold = 1e3  # Threshold for considering kappa as "large"
    small_kappa_threshold = 1e-3  # Threshold for considering kappa as "small"

    # Approximate normalization constant for large and small kappa values


    norm_const = torch.where(
        kappas > large_kappa_threshold,
        kappas / (2 * math.pi),  # Approximation for large kappa
        kappas / (2 * math.pi * (1-torch.exp(-2*kappas)))
    ) # Shape: (bz, num_spheres, 1)
    # norm_const = kappas / (4 * math.pi * (1-torch.exp(-2*kappas)))

    # Compute dot products between input w and the axes of the spheres (unit vectors)


    dot_products = torch.matmul(axes, w.transpose(-1,-2))-1  # Shape: (bz, num_spheres, data_sizes)
    # Compute the weighted von Mises-Fisher pdf values
    weighted_exps = weights * norm_const * torch.exp(kappas * dot_products)  # Shape: (bz, num_spheres, data_sizes)
    q = torch.sum(weighted_exps, dim=-2)  # Shape: (bz, data_sizes)
    return q # Shape: (bz, data_sizes) or (data_sizes) if bz=1

def plot_outputs_3d(references, predictions, sizes, save_path=None, return_fig=False):
    # Define z and phi range
    z_min, z_max = -1, 1
    phi_min, phi_max = -np.pi, np.pi

    # Create a grid for 3D plotting
    z_in = np.linspace(z_min, z_max, sizes[0])
    phi_in = np.linspace(phi_min, phi_max, sizes[1])
    Z, Phi = np.meshgrid(z_in, phi_in, indexing='ij')

    target_img = references
    predict_img = predictions

    # Ensure input data matches grid shape
    if predict_img.shape != Z.shape:
        predict_img = predict_img.reshape(Z.shape)
    if target_img is not None and target_img.shape != Z.shape:
        target_img = target_img.reshape(Z.shape)

    # Set up subplots for 3D visualization
    if target_img is not None and np.sum(target_img) > 0:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot prediction 3D surface
    ax1.plot_surface(Z, Phi, predict_img, rstride=1, cstride=1, cmap='rainbow')
    ax1.set_title('Prediction')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Phi')
    ax1.set_zlabel('Value')

    # Plot reference 3D surface, if available
    if target_img is not None and np.sum(target_img) > 0:
        ax2.plot_surface(Z, Phi, target_img, rstride=1, cstride=1, cmap='rainbow')
        ax2.set_title('Reference')
        ax2.set_xlabel('Z')
        ax2.set_ylabel('Phi')
        ax2.set_zlabel('Value')

    # Adjust layout
    plt.tight_layout()

    # Save or return the figure
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    elif return_fig:
        return fig
    else:
        plt.show()




def smooth_curve(values, smoothing_factor=0.9):
    smoothed_values = []
    last = values[0]
    for value in values:
        smoothed_value = last * smoothing_factor + (1 - smoothing_factor) * value
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return smoothed_values

def plot_losses(train_losses, val_losses = None):
    train_losses_smoothed = smooth_curve(train_losses)
    if val_losses is not None:
        val_losses_smoothed = smooth_curve(val_losses)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_smoothed, label="Training Loss (Smoothed)", color="blue")
    if val_losses is not None:
        plt.plot(val_losses_smoothed, label="Validation Loss (Smoothed)", color="red")
    plt.yscale("log")  # Log scale for the y-axis
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Loss (Log Scale with Smoothing)")
    plt.legend()
    plt.show()


def extract_param(vmf_param):
    vmf_param = vmf_param.reshape(-1,4)  
    weights = F.softmax(vmf_param[:,0],dim=-1).unsqueeze(1)
    kappas = torch.exp(vmf_param[:,1]).unsqueeze(1)
    theta = torch.sigmoid(vmf_param[:,2])* math.pi 
    phi = torch.sigmoid(vmf_param[:,3])* math.pi * 2
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    mus = torch.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), dim=1)
    return weights, mus, kappas
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), './init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100,
                   smooth_f=0.05, diverge_th=5, use_tqdm=True):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)

        with tqdm(total=num_iter, disable=not use_tqdm) as t:
            for iteration in range(num_iter):

                loss = self._train_batch(iterator)

                lrs.append(lr_scheduler.get_last_lr()[0])

                # update lr
                lr_scheduler.step()

                if iteration > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

                if loss < best_loss:
                    best_loss = loss

                losses.append(loss)

                if loss > diverge_th * best_loss:
                    print("Stopping early, the loss has diverged")
                    break

                t.set_postfix({'loss': loss})
                t.update()
            

        # reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        graph_data = iterator.get_batch()

        graph_data = graph_data.to(self.device)

        y_pred = self.model(graph_data).reshape(-1)

        loss = self.criterion(y_pred, graph_data.y-graph_data.y_first)

        loss.backward()

        self.optimizer.step()

        return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in
                self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs = next(self._iterator)

        return inputs

    def get_batch(self):
        return next(self)

def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5):

    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()
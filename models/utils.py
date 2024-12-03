
# Install required packages.
import os
import torch
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
    # Ensure kappas are non-negative for stability
    kappas = torch.clamp(kappas, min=1e-10, max=1e5)

    # Define thresholds for approximations
    large_kappa_threshold = 1e5  # Threshold for considering kappa as "large"
    small_kappa_threshold = 1e-3  # Threshold for considering kappa as "small"

    # Approximate normalization constant for large and small kappa values


    norm_const = torch.where(
        kappas > large_kappa_threshold,
        kappas / (2 * math.pi),  # Approximation for large kappa
        kappas / (2 * math.pi * (1-torch.exp(-2*kappas)))
    )
    # norm_const = kappas / (4 * math.pi * (1-torch.exp(-2*kappas)))

    # Compute dot products between input w and the axes of the spheres (unit vectors)
    dot_products = torch.matmul(w, axes.transpose(0, 1))-1  # Shape: (data_sizes, num_spheres)

    # Compute the weighted von Mises-Fisher pdf values
    weighted_exps = weights * norm_const * torch.exp(kappas * dot_products)  # Shape: (data_sizes, num_spheres)
    q = torch.sum(weighted_exps, dim=-1)  # Shape: (data_sizes,)
    q = torch.clamp(q, min=1e-10, max=1e10)  # Further clamping to avoid extreme values
    return q

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
    weights = F.softmax(vmf_param[:,0],dim=-1)
    kappas = torch.exp(vmf_param[:,1])
    theta = torch.sigmoid(vmf_param[:,2])* math.pi 
    phi = torch.sigmoid(vmf_param[:,3])* math.pi * 2
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    mus = torch.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), dim=1)
    return weights, mus, kappas

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np

import threading
import os
import sys
import time
import argparse
import math


# Set the device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class vMFMixtureModel(nn.Module):
    def __init__(self, num_components):
        super(vMFMixtureModel, self).__init__()
        self.num_components = num_components

        # 混合系数的对数几率
        self.w_logits = nn.Parameter(torch.randn(num_components))

        # 角度参数 theta 和 phi
        self.theta_phi = nn.Parameter(torch.randn(2 * num_components)*3)

        # 浓度参数的对数
        self.log_kappa = nn.Parameter(torch.randn(num_components))

    def forward(self):
        # 计算混合系数
        weight = F.softmax(self.w_logits, dim=0)

        # 使用 sigmoid 将 theta 和 phi 限制在 [0,1]
        theta_phi = torch.sigmoid(self.theta_phi)

        # 映射到实际角度范围
        theta = theta_phi[:self.num_components] * math.pi          # [0, π]
        phi = theta_phi[self.num_components:] * 2 * math.pi        # [0, 2π]

        # 计算球坐标系下的 μ
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        mu = torch.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), dim=1)  # Shape: (num_components, 3)

        # 计算浓度参数
        kappa = torch.exp(self.log_kappa)
        # kappa = torch.clamp(kappa, min=1e-10, max=1e5)  # 限制 kappa 的最大值，防止数值不稳定

        return weight, mu, kappa
    
    

def negative_log_likelihood(raw_data, weight, mu, kappa, **kwargs):

    kl_lambda = kwargs['kl_lambda']
    l1_lambda = kwargs['l1_lambda']
    l2_lambda = kwargs['l2_lambda']
    p = kwargs['p']
    w = kwargs['w']

    total_prob = multi_vmf(weight, mu, kappa, raw_data)
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

# 多重 von Mises-Fisher 分布函数
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

def plot_outputs_3d(references, predictions, sizes, save_path=None):
    # 定义 z 和 phi 的范围
    z_min, z_max = -1, 1
    phi_min, phi_max = -np.pi, np.pi

    # 创建用于 3D 绘图的网格
    z_in = np.linspace(z_min, z_max, sizes[0])
    phi_in = np.linspace(phi_min, phi_max, sizes[1])

    Z, Phi = np.meshgrid(z_in, phi_in, indexing='ij')

    target_img = references
    predict_img = predictions

    # 确保输入数据的形状与网格匹配
    if predict_img.shape != Z.shape:
        predict_img = predict_img.reshape(Z.shape)
    if target_img is not None and target_img.shape != Z.shape:
        target_img = target_img.reshape(Z.shape)

    # 设置用于 3D 可视化的子图
    if target_img is not None and np.sum(target_img) > 0:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # 绘制预测数据的 3D 曲面
    ax1.plot_surface(Z, Phi, predict_img, rstride=1, cstride=1, cmap='rainbow')
    ax1.set_title('Prediction')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Phi')
    ax1.set_zlabel('Value')

    # 如果有参考数据，绘制其 3D 曲面
    if target_img is not None and np.sum(target_img) > 0:
        ax2.plot_surface(Z, Phi, target_img, rstride=1, cstride=1, cmap='rainbow')
        ax2.set_title('Reference')
        ax2.set_xlabel('Z')
        ax2.set_ylabel('Phi')
        ax2.set_zlabel('Value')

    # 显示图形
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
def get_gridX(sizes):
    i_idx = torch.arange(sizes[0], dtype=torch.float, device=device) / sizes[0]
    j_idx = torch.arange(sizes[1], dtype=torch.float, device=device) / sizes[1]
    i_grid, j_grid = torch.meshgrid(i_idx, j_idx)
    pos_x = i_grid * 2 - 1
    pos_phi = (j_grid * 2 - 1) * np.pi
    pos_r = torch.sqrt(1 - pos_x**2)
    pos_y = pos_r * torch.cos(pos_phi)
    pos_z = pos_r * torch.sin(pos_phi)
    X = torch.stack((pos_x, pos_y, pos_z), dim=-1).reshape(-1, 3).to(device)
    return X   

# 加载 ray 数据的函数
def load_rawdata(filename, sizes, device, verbose=False):
    X = get_gridX(sizes)

    rawdata = np.fromfile(filename, dtype=np.float32)
    rawdata = rawdata.reshape(-1, 4)
    # print(rawdata.shape)
    x = rawdata[:,0]
    # print(np.max(data[:,1]))
    phi = rawdata[:,1]-np.pi
    r = np.sqrt(1 - x**2)
    y = r * np.cos(phi)
    z = r * np.sin(phi)


    # 创建网格分布
    x_edges = np.linspace(-1, 1, sizes[0]+1)  
    phi_edges = np.linspace(-np.pi, np.pi, sizes[1]+1)

    # 统计 (X, φ) 分布
    H, _, _ = np.histogram2d(x, phi, bins=[x_edges, phi_edges])
    ray_data = torch.tensor(H, dtype=torch.float32, device=device).reshape(-1, 1)
    if ray_data.shape[0] != sizes[0] * sizes[1]:
        print("Error: ray data shape mismatch!")
        exit(1)
    area = 4 * math.pi / (ray_data.shape[0])
    ray_data = ray_data / torch.sum(ray_data) / area
    

    raw_X = np.column_stack((x, y, z))
    np.random.shuffle(raw_X)
    raw_num = min(8192, raw_X.shape[0])
    raw_X = raw_X[:raw_num, :]

    # 将数据转换为张量
    raw_data = torch.tensor(raw_X, dtype=torch.float32, device=device)
    
    if verbose:
        print("X Mesh shape:", X.shape)
        print("ray data shape:", ray_data.shape)
        print("raw data shape:", raw_data.shape)
    return raw_data, ray_data, X

def train_model(model_id, vmf, optimizer, dataset, hyperparams, device, save_path=None, verbose=False):
    kl_lambda = hyperparams['kl_lambda']
    l1_lambda = hyperparams['l1_lambda']
    l2_lambda = hyperparams['l2_lambda']
    num_epochs = hyperparams['num_epochs']

    # 将数据移动到指定设备
    samples = dataset["samples"].to(device, non_blocking=True)
    target = dataset["target"].to(device, non_blocking=True)
    w_data = dataset["w_data"].to(device, non_blocking=True)

    def update_model():
        optimizer.zero_grad()
        pi, mu, kappa = vmf()
        loss, loss_dict = negative_log_likelihood(
            samples, pi, mu, kappa,
            kl_lambda=kl_lambda, l1_lambda=l1_lambda, l2_lambda=l2_lambda,
            p=target, w=w_data
        )
        loss.backward()
        optimizer.step()
        return loss, loss_dict

    vmf.train()

    if verbose:
        loss_history = []
        with tqdm(total=num_epochs, desc=f'Model {model_id} Training on {device}') as pbar:
            for epoch in range(num_epochs):
                loss, loss_dict = update_model()
                loss_history.append(loss.item())
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'KL': f'{loss_dict["NLL"].item():.4f}',
                    'Rec': f'{loss_dict["Rec"].item():.4f}',
                    'L2': f'{loss_dict["L2"].item():.4f}'
                })
                pbar.update(1)

        plot_losses(loss_history)
    else:
        for epoch in range(num_epochs):
            update_model()

    # 保存模型参数
    if save_path is not None:
        model_save_path = os.path.join(save_path, f"vmf_parameters.pth")
        torch.save(vmf.state_dict(), model_save_path)
        print(f"Model {model_id} parameters saved to {model_save_path}")

# 加载模型参数
def load_model(model_path, num_components, device):
    # 初始化模型
    model = vMFMixtureModel(num_components=num_components).to(device)
    
    # 加载保存的参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 切换到评估模式
    model.eval()
    
    return model

def generate_filenames(base_path, num_files):
    """Generate a list of file paths."""
    return [os.path.join(base_path, f"{i}/rawdataNonSpe.bin") for i in range(0, num_files)]

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Script to train vMF mixture models.')
    parser.add_argument('--base_path', type=str, default='datasets/raw_data/foam0',
                        help='Base path for data files. Example: datasets/')
    parser.add_argument('--num_files', type=int, default=1,
                        help='Number of data files. Default is 1.')
    parser.add_argument('--num_components', type=int, default=64,
                        help='Number of components in the vMF mixture model. Default is 64.')
    parser.add_argument('--num_epochs', type=int, default=10000,
                        help='Number of training epochs. Default is 10000.')
    parser.add_argument('--kl_lambda', type=float, default=1.0,
                        help='Weight for KL divergence. Default is 1.')
    parser.add_argument('--l1_lambda', type=float, default=10.0,
                        help='Weight for L1 regularization. Default is 10.')
    parser.add_argument('--l2_lambda', type=float, default=0.5,
                        help='Weight for L2 regularization. Default is 0.5.')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate. Default is 5e-3.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (for Adam optimizer). Default is 1e-5.')
    parser.add_argument('--sizes', type=str, default='64,64',
                        help='Sizes as a comma-separated list of integers. Example: "64,64"')
    # 默认设备设为所有可用GPU，如果没有GPU则为CPU
    if torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    parser.add_argument('--device', type=str, default=default_device,
                        help='Computation device. Defaults to CUDA if available, otherwise CPU.')

    args = parser.parse_args()

    print(f"Training vMF mixture models with the datasets in {args.base_path}") 
    # 初始化参数
    base_path = args.base_path
    num_files = args.num_files
    num_components = args.num_components
    num_epochs = args.num_epochs
    kl_lambda = args.kl_lambda
    l1_lambda = args.l1_lambda
    l2_lambda = args.l2_lambda
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    # 移除全局device，改为每个模型单独分配
    # device = args.device

    sizes = [int(s) for s in args.sizes.split(',')]

    hyperparams = {
        "kl_lambda": kl_lambda,
        "l1_lambda": l1_lambda,
        "l2_lambda": l2_lambda,
        "num_epochs": num_epochs
    }

    if not os.path.isdir(base_path):
        print(f"Error: The base path '{base_path}' does not exist.")
        sys.exit(1)

    filenames = generate_filenames(base_path, num_files)
    num_models = len(filenames)

    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0 and args.device.startswith('cuda'):
        print("CUDA is not available. Falling back to CPU.")
        device_list = ['cpu'] * num_models
    elif num_gpus > 0:
        # 分配每个模型到不同的GPU，轮询分配
        device_list = [f'cuda:{i % num_gpus}' for i in range(num_models)]
    else:
        device_list = ['cpu'] * num_models

    datasets = []  
    optimizers = [] 
    models = []

    for i in range(num_models):
        device = device_list[i]
        vmf = vMFMixtureModel(num_components=num_components).to(device)
        optimizer = torch.optim.Adam(vmf.parameters(), lr=learning_rate, weight_decay=weight_decay)
        models.append(vmf)
        optimizers.append(optimizer)

        # 加载对应的数据集
        raw_data, ray_data, X = load_rawdata(filenames[i], sizes, device, verbose=False)
        dataset = {
            "samples": raw_data.clone().detach(),
            "target": ray_data.reshape(-1).to(device),
            "w_data": X.clone().detach()
        }
        datasets.append(dataset)

    # 确定保存路径（基于每个数据文件夹）
    save_paths = [os.path.dirname(f) for f in filenames]
    start_time = time.time()
    # 启动线程进行并行训练
    threads = []
    for i in range(num_models):
        thread = threading.Thread(
            target=train_model,
            args=(
                i, models[i], optimizers[i],
                datasets[i],
                hyperparams,
                device_list[i],
                save_paths[i]
            )
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # # 训练完成后进行模型评估
    # for i in range(num_models):
    #     model = models[i]
    #     dataset = datasets[i]
    #     X = dataset["w_data"]
    #     reference = dataset["target"]

    #     with torch.no_grad():
    #         model.eval()
    #         weights, axes, kappas = model()
    #         predictions = multi_vmf(weights, axes, kappas, X).cpu().numpy() 
    #         predictions = predictions.reshape(sizes[0], sizes[1])
    #         reference_np = reference.cpu().numpy().reshape(sizes[0], sizes[1])
    #         plot_save_path = os.path.join(save_paths[i], "compare.png")
    #         plot_outputs_3d(reference_np, predictions, sizes, save_path=plot_save_path)

if __name__ == "__main__":
    main()


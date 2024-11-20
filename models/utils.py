
# Install required packages.
import os
import torch
# Helper function for visualization.
import matplotlib.pyplot as plt
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def parse_cell_txt(file_path):
    cells = []  # 用于存储解析后的数据

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        # 每三行表示一个cell的数据
        center_line = lines[i].strip()
        cell_neighbors_line = lines[i + 1].strip()
        node_neighbors_line = lines[i + 2].strip()

        # 解析中心位置和半径
        center_data = list(map(float, center_line.split()))
        cell_center = {
            "x": center_data[0],
            "y": center_data[1],
            "z": center_data[2],
            "radius": center_data[3]
        }

        # 解析cell的连接关系
        cell_neighbors = list(map(int, cell_neighbors_line.split()))

        # 解析node的连接关系
        node_neighbors = list(map(int, node_neighbors_line.split()))

        # 组合到一个字典
        cell = {
            "cell_center": cell_center,
            "cell_neighbors": cell_neighbors,
            "node_neighbors": node_neighbors
        }

        # 添加到结果列表
        cells.append(cell)

    return cells

def sort_nodes_and_edges(node_features_np: np.ndarray, edges_np: np.ndarray):
    """
    对节点按 (r, x, phi) 排序，并更新边的索引。

    Args:
        node_features (numpy.ndarray): 节点特征，形状为 [num_nodes, feature_dim]。
        edges (numpy.ndarray): 边列表，形状为 [2, num_edges]。

    Returns:
        sorted_node_features: 排序后的节点特征 (torch.Tensor)。
        sorted_edges: 排序后的边索引 (torch.Tensor)。
    """
    # 提取 x, y, z 坐标
    x = node_features_np[:, 0]
    y = node_features_np[:, 1]
    z = node_features_np[:, 2]

    # 计算球坐标
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(z, y) % (2 * np.pi)

    # 构造球坐标用于排序
    spherical_coords = np.stack((r, x, phi), axis=1)

    # 按球坐标排序 (r, x, phi)
    sorted_order = np.lexsort((spherical_coords[:, 2], spherical_coords[:, 1], spherical_coords[:, 0]))
    sorted_node_features_np = node_features_np[sorted_order]

    # 创建从原始索引到排序后索引的映射
    index_mapping = {original_idx: sorted_idx for sorted_idx, original_idx in enumerate(sorted_order)}

    # 更新边索引
    sorted_edges = []
    for edge in edges_np:  # 遍历每条边
        node1, node2 = edge
        sorted_node1 = index_mapping[node1]
        sorted_node2 = index_mapping[node2]
        sorted_edges.append([sorted_node1, sorted_node2])

    return sorted_node_features_np, sorted_edges

def get_gnn_dataset(cells, device='cpu'):
    """
    根据解析后的cells数据构建GNN数据集
    Args:
        cells (list): 包含每个cell的中心和连接关系的字典列表
    Returns:
        torch_geometric.data.Data: 包含节点特征和边信息的图数据
    """
    # 节点特征: 使用cell_center (x, y, z, radius)
    node_features = []
    edges = []

    for node_id, cell in enumerate(cells):
        # 添加节点特征
        center = cell["cell_center"]
        node_features.append([center["x"], center["y"], center["z"], center["radius"]])

        # 添加边: 从当前节点到其邻居
        for neighbor_id in cell["cell_neighbors"]:
            # 添加双向边
            edges.append((node_id, neighbor_id))
            edges.append((neighbor_id, node_id))
    
    # 转换为numpy数组
    node_features_np = np.array(node_features)
    edges_np = np.array(edges)
    edges_np = np.unique(edges_np, axis=0)  # 去重

    # 对节点和边进行排序
    sorted_node_features_np, sorted_edges_np = sort_nodes_and_edges(node_features_np, edges_np)

    # 转换为torch.Tensor
    node_features_tensor = torch.tensor(sorted_node_features_np, dtype=torch.float).to(device)
    edges_tensor = torch.tensor(sorted_edges_np, dtype=torch.long).t().contiguous().to(device)

    # 创建图数据
    data = Data(x=node_features_tensor[:,3:], edge_index=edges_tensor)
    data.pos = node_features_tensor[:,0:3]

    return data

def load_target_data(filename, device='cpu'):
    """
    Loads ray data from parameter file.
    """
    datafile = filename

    parameters = torch.load(filename, map_location=device, weights_only=True)
    w_logits = parameters['w_logits']
    theta_phi = parameters['theta_phi']
    log_kappa = parameters['log_kappa']
    theta = theta_phi[:log_kappa.size(0)]
    phi = theta_phi[log_kappa.size(0):]
    indices = np.lexsort((-phi.cpu().numpy(), -theta.cpu().numpy(), -log_kappa.cpu().numpy(), -w_logits.cpu().numpy()))
    sorted_indices = torch.tensor(indices)
    target_data = torch.stack([w_logits,log_kappa,theta,phi], dim=1)
    target_data = target_data[sorted_indices,:]
    target_data = torch.flatten(target_data)
    return target_data.to(device)
    
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

    

def get_gridX(sizes, device):
    i_idx = torch.arange(sizes[0], dtype=torch.float, device=device) / sizes[0]
    j_idx = torch.arange(sizes[1], dtype=torch.float, device=device) / sizes[1]
    i_grid, j_grid = torch.meshgrid(i_idx, j_idx, indexing='ij')
    pos_x = i_grid * 2 - 1
    pos_phi = (j_grid * 2 - 1) * np.pi
    pos_r = torch.sqrt(1 - pos_x**2)
    pos_y = pos_r * torch.cos(pos_phi)
    pos_z = pos_r * torch.sin(pos_phi)
    X = torch.stack((pos_x, pos_y, pos_z), dim=-1).reshape(-1, 3).to(device)
    return X


# 加载 ray 数据的函数
def load_rawdata(filename, sizes, device='cpu', verbose=False):
    X = get_gridX(sizes, device)

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
    raw_num = min(4096, raw_X.shape[0])
    raw_X = raw_X[:raw_num, :]

    # 将数据转换为张量
    raw_data = torch.tensor(raw_X, dtype=torch.float32, device=device)
    
    if verbose:
        print("X Mesh shape:", X.shape)
        print("ray data shape:", ray_data.shape)
        print("raw data shape:", raw_data.shape)
    return raw_data, ray_data, X



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

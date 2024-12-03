import os
import torch
import numpy as np
import math
import json
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
from collections import deque, defaultdict

def parse_cell_txt(file_path):
    cells = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        center_line = lines[i].strip()
        cell_neighbors_line = lines[i + 1].strip()
        node_neighbors_line = lines[i + 2].strip()

        # Parse cell center data
        center_data = list(map(float, center_line.split()))
        cell_center = {
            "x": center_data[0],
            "y": center_data[1],
            "z": center_data[2],
            "radius": center_data[3]
        }

        cell_neighbors = list(map(int, cell_neighbors_line.split()))

        node_neighbors = list(map(int, node_neighbors_line.split()))

        cell = {
            "cell_center": cell_center,
            "cell_neighbors": cell_neighbors,
            "node_neighbors": node_neighbors
        }

        cells.append(cell)

    return cells

def generate_filenames(folder_path, target_file, num_files):
    return [os.path.join(folder_path, f"{i}", target_file) for i in range(num_files)]

def sort_nodes_and_edges_bfs(pos_np, edges_np):
    """
    Sorts nodes and edges based on distance from origin and BFS traversal.
    Args:
        node_features_np (numpy.ndarray): [num_nodes, feature_dim]
        edges_np (numpy.ndarray): [2, num_edges]

    Returns:
        sorted_node_features_np (numpy.ndarray): [num_nodes, feature_dim]
        sorted_edges_np (numpy.ndarray): [2, num_edges]
    """
    num_nodes = pos_np.shape[0]

    # Step 1: Calculate Euclidean distances from origin
    distances = np.linalg.norm(pos_np, axis=1)

    # Step 2: Find nodes sorted by distance
    sorted_distance_indices = np.argsort(distances)

    # Step 3: Build adjacency list
    adjacency = defaultdict(list)
    for edge in edges_np:
        src, dst = edge
        adjacency[src].append(dst)

    # Step 4: BFS traversal
    visited = np.zeros(num_nodes, dtype=bool)
    new_order = []
    for node in sorted_distance_indices:
        if not visited[node]:
            queue = deque()
            queue.append(node)
            visited[node] = True
            while queue:
                current = queue.popleft()
                new_order.append(current)
                for neighbor in adjacency[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

    return np.array(new_order)  # Shape [num_nodes]

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

def get_gnn_dataset(cells, json_path=None, device='cpu'):
    """
    Converts cell data to a PyG graph dataset.
    Args:
        cells (list): contains cell data
    Returns:
        torch_geometric.data.Data: graph dataset
    """
    # Node Feature: 使用cell_center (x, y, z, radius)
    node_features = []
    edges = []

    for node_id, cell in enumerate(cells):
        # Add node feature
        center = cell["cell_center"]
        node_features.append([center["x"], center["y"], center["z"], center["radius"]])

        # Add edges
        for neighbor_id in cell["cell_neighbors"]:
            edges.append((node_id, neighbor_id))
            edges.append((neighbor_id, node_id))
    
    # Convert to numpy
    node_features_np = np.array(node_features)
    edges_np = np.array(edges) # Shape [num_edges, 2]
    edges_np = np.unique(edges_np, axis=0)
    pos_np = node_features_np[:, 0:3]
    # Sort nodes and edges based on distance from origin and BFS traversal
    new_order = sort_nodes_and_edges_bfs(pos_np, edges_np)
    sorted_node_features_np = node_features_np[new_order]
    test_edges_np = np.zeros_like(edges_np)
    sorted_edges_np = new_order[edges_np]
    for i, edge in enumerate(edges_np):
        test_edges_np[i, 0] = new_order[edge[0]]
        test_edges_np[i, 1] = new_order[edge[1]]
    assert np.all(sorted_edges_np == test_edges_np), "Error: sorted_edges_np != test_edges_np"

    if json_path is not None:
        with open(json_path, 'r') as f:
            json.dump({"node_features": sorted_node_features_np.tolist(), "edges": sorted_edges_np.tolist()}, f)

    # Convert to PyG tensors
    node_features_tensor = torch.tensor(sorted_node_features_np, dtype=torch.float).to(device)
    edges_tensor = torch.tensor(sorted_edges_np, dtype=torch.long).t().contiguous().to(device)

    # Graph data
    data = Data(x=node_features_tensor[:,3:], edge_index=edges_tensor)
    data.pos = node_features_tensor[:,0:3]

    return data

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

# Load raw data: ray position and ray direction
def load_rawdata(filename, sizes, device, verbose=False, dtype=np.float32):
    X = get_gridX(sizes, device)

    rawdata = np.fromfile(filename, dtype=dtype)
    
    # 如果是 float16，转换为 float32
    if dtype == np.float16:
        if verbose:
            print(f"Converting data from float16 to float32")
        rawdata = rawdata.astype(np.float32)
    rawdata = rawdata.reshape(-1, 4)
    # print(rawdata.shape)
    x = rawdata[:,0]
    # print(np.max(data[:,1]))
    phi = rawdata[:,1]-np.pi
    r = np.sqrt(1 - x**2)
    y = r * np.cos(phi)
    z = r * np.sin(phi)


    # Create histogram edges
    x_edges = np.linspace(-1, 1, sizes[0]+1)  
    phi_edges = np.linspace(-np.pi, np.pi, sizes[1]+1)

    # Statistics of ray data
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

    raw_data = torch.tensor(raw_X, dtype=torch.float32, device=device)
    
    if verbose:
        print("X Mesh shape:", X.shape)
        print("ray data shape:", ray_data.shape)
        print("raw data shape:", raw_data.shape)
    return raw_data, ray_data, X


class FoamDataset(InMemoryDataset):
    def __init__(self, root, foam_names, file_num, sizes, transform=None, pre_transform=None, device='cpu'):
        self.foam_names = foam_names
        self.file_num = file_num
        self.sizes = sizes
        self.device = device
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=self.device, weights_only=False)

    @property
    def raw_file_names(self):
        # 用于检查原始文件是否存在
        return [os.path.join(foam, f"{i}") for foam in self.foam_names for i in range(self.file_num)]

    @property
    def processed_file_names(self):
        # 用于检查处理后的文件是否存在
        return ['data.pt']

    def download(self):
        # 如果需要从远程服务器下载数据，请实现此方法
        pass

    def process(self):
        datasets = []
        ray_datasets = []

        for foam_n in self.foam_names:
            folder_path = os.path.join(self.raw_dir, foam_n)  # 假设 raw_data 放在 root/raw 文件夹
            graphs_path = generate_filenames(folder_path, "cells.txt", self.file_num)
            vmf_path = generate_filenames(folder_path, "vmf_parameters.pth", self.file_num)
            rays_path = generate_filenames(folder_path, "rawdataNonSpe.bin", self.file_num)

            for i in range(self.file_num):
                # 解析图数据
                parsed_cells = parse_cell_txt(graphs_path[i])
                graph_data = get_gnn_dataset(parsed_cells, device=self.device)
                
                # 加载目标数据
                vmf_data = load_target_data(vmf_path[i], self.device)
                graph_data.y = vmf_data.reshape(-1)

                # 加载原始数据
                _, ray_data, _ = load_rawdata(rays_path[i], self.sizes, self.device)
                ray_datasets.append(ray_data)
                datasets.append(graph_data)

        # 转换为 InMemoryDataset 格式
        data_list = datasets
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  # 保存处理后的数据






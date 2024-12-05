import os
import sys
import torch
import numpy as np
import math
import json
from torch_geometric.data import Data
from collections import deque, defaultdict

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

def sort_nodes_and_edges(pos_np: np.ndarray, edges_np: np.ndarray):
    """
    Sorts nodes and edges based on distance from origin.
    """
    x = pos_np[:, 0]
    y = pos_np[:, 1]
    z = pos_np[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(z, y) % (2 * np.pi)

    spherical_coords = np.stack((r, x, phi), axis=1)

    sorted_order = np.lexsort((spherical_coords[:, 2], spherical_coords[:, 1], spherical_coords[:, 0]))
    return sorted_order


def get_gridX(sizes, device='cpu'):
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
def load_rawdata(filename, sizes,  
                samples = 8192, 
                dtype=np.float32, 
                force_reload=False,
                verbose=False,
                device='cpu'):
    dirname = os.path.dirname(filename)
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(dirname, f"{base_filename}.pth")

    if os.path.exists(save_path) and not force_reload:
        if verbose:
            print(f"Loading cached data from {save_path}...")
        cached_data = torch.load(save_path, map_location=device, weights_only=True)
        rawdata_tensor = cached_data["raw"]
        raydata_tensor = cached_data["ray"]
        if verbose:
            print("ray data shape:", raydata_tensor.shape)
            print("raw data shape:", rawdata_tensor.shape)
        return rawdata_tensor, raydata_tensor

    rawdata_np = np.fromfile(filename, dtype=dtype)

    # If the data is in float16, convert it to float32
    if dtype == np.float16:
        if verbose:
            print(f"Converting data from float16 to float32")
        rawdata_np = rawdata_np.astype(np.float32)
    rawdata_np = rawdata_np.reshape(-1, 4)
    # print(rawdata_np.shape)
    x = rawdata_np[:,0]
    # print(np.max(data[:,1]))
    phi = rawdata_np[:,1]-np.pi
    r = np.sqrt(1 - x*x)
    y = r * np.cos(phi)
    z = r * np.sin(phi)


    # Create histogram edges
    x_edges = np.linspace(-1, 1, sizes[0]+1)  
    phi_edges = np.linspace(-np.pi, np.pi, sizes[1]+1)

    # Statistics of ray data
    H, _, _ = np.histogram2d(x, phi, bins=[x_edges, phi_edges])
    raydata_tensor = torch.tensor(H, dtype=torch.float32, device=device).reshape(-1, 1)
    if raydata_tensor.shape[0] != sizes[0] * sizes[1]:
        print("Error: ray data shape mismatch!")
        sys.exit(1)
    area = 4 * math.pi / (raydata_tensor.shape[0])
    raydata_tensor = raydata_tensor / torch.sum(raydata_tensor) / area
    
    raw_X = np.stack((x, y, z), axis=1)
    raw_X = np.random.permutation(raw_X)
    raw_num = min(samples, raw_X.shape[0])
    raw_X = raw_X[:raw_num, :]
    rawdata_tensor = torch.tensor(raw_X, dtype=torch.float32, device=device)
    
    if verbose:
        print("ray data shape:", raydata_tensor.shape)
        print("raw data shape:", rawdata_tensor.shape)

    torch.save({"raw": rawdata_tensor, "ray": raydata_tensor}, save_path)

    return rawdata_tensor, raydata_tensor
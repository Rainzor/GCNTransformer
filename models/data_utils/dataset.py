import os
import glob
import sys
import torch
import numpy as np
import math
import json
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
from collections import deque, defaultdict
from models.data_utils.utils import sort_nodes_and_edges_bfs, sort_nodes_and_edges, get_gridX
from torch.utils.data import Dataset

class FoamDataset(InMemoryDataset):
    def __init__(self, root, json_path, sizes, 
                transform=None, pre_transform=None,
                use_pre_transform=True, 
                force_reload = False,
                device='cpu'):
        self.json_path = json_path
        self.sizes = sizes
        self.device = device
        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        processed_file = self.processed_paths[1] if use_pre_transform else self.processed_paths[0]
        self.data, self.slices = torch.load(processed_file, map_location=self.device, weights_only=False)

    @property
    def raw_file_names(self):
        # Check if the raw files exist
        return [self.json_path]

    @property
    def processed_file_names(self):
        # Check if the processed files exist
        return ['data.pt', 'data_pre_transform.pt']

    def download(self):
        # Download the raw data from the source
        pass

    def process(self):
        datasets = []
        ray_datasets = []
        json_path = os.path.join(self.raw_dir, self.json_path)
        if not os.path.isfile(json_path):
            print(f"Error: The file '{json_path}' does not exist.")
            sys.exit(1)
        directories = []
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['rawdata']:
                directories.append(os.path.join(self.raw_dir, entry["folder"]))
        
        config_files = []
        for directory in directories:
            pattern = os.path.join(directory, "*","*", "config.json")
            config_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(config_files)} graph data files.")
        for config_file in config_files:
            subdir = os.path.dirname(config_file)
            node_path = os.path.join(subdir, "nodes.txt")
            edge_path = os.path.join(subdir, "edges.txt")
            vmfshell_path = ""
            vmffirst_path = ""
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    vmfshell_path = os.path.join(subdir, entry["vmfShell"])
                    vmffirst_path = os.path.join(subdir, entry["vmfFirst"])
            # Parse graph data
            graph_data = self.read_graph(node_path, edge_path, self.device)
            # Load VMF data
            vmf_shell = self.read_vmf(vmfshell_path, self.device)
            vmf_first = self.read_vmf(vmffirst_path, self.device)

            graph_data.y = vmf_shell
            graph_data.y_first = vmf_first

            datasets.append(graph_data)


        # Cnvert to PyG dataset
        data_list = datasets
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  # Save processed data to file

        # if pre_transform is not None, apply pre_transform to data
        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]
            data_pre, slices_pre = self.collate(data_list)
            torch.save((data_pre, slices_pre), self.processed_paths[1])
            print(f"Transformed {len(data_list)} datasets.")
        else: # Save original data if no pre_transform
            torch.save((data, slices), self.processed_paths[1])
    
    @staticmethod
    def read_graph(node_path, edge_path, device = 'cpu'):
        """
        Converts node, edge data to a PyG graph dataset.
        Args:
            node_path (str): path to node data
            edge_path (str): path to edge data
        Returns:
            torch_geometric.data.Data: graph dataset
        """
        node_attr = []
        edge_index = []
        edge_attr = []

        with open(node_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                node_attr.append([float(x) for x in parts[1:]])
        
        with open(edge_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                edge_index.append([int(parts[0]), int(parts[1])])
                edge_index.append([int(parts[1]), int(parts[0])])
                edge_attr.append([float(x) for x in parts[2:]])
                edge_attr.append([float(x) for x in parts[2:]])
        
        # Convert to numpy
        node_attr_np = np.array(node_attr) # Shape [num_nodes, num_node_features]
        edge_index_np = np.array(edge_index) # Shape [num_edges, 2]
        edge_attr_np = np.array(edge_attr) # Shape [num_edges, num_edge_features]
        pos_np = node_attr_np[:, 0:3]
        # Sort nodes and edges based on distance from origin and BFS traversal
        # new_order = sort_nodes_and_edges_bfs(pos_np, edge_index_np)
        new_order = sort_nodes_and_edges(pos_np, edge_index_np)

        sorted_node_attr_np = node_attr_np[new_order] 
        sorted_edge_index_np = new_order[edge_index_np] # map old indices to new indices


        # Convert to PyG tensors
        node_attr_tensor = torch.tensor(sorted_node_attr_np, dtype=torch.float).to(device)
        edge_index_tensor = torch.tensor(sorted_edge_index_np, dtype=torch.long).t().contiguous().to(device)
        edge_attr_tensor = torch.tensor(edge_attr_np, dtype=torch.float).to(device)

        # Graph data
        data = Data(x=node_attr_tensor[:,3:], 
                    edge_index=edge_index_tensor, 
                    pos=node_attr_tensor[:,0:3],
                    edge_attr=edge_attr_tensor)
        return data
    
    @staticmethod
    def read_vmf(vmf_path, device='cpu'):
        """
        Loads vmf data from parameter file.
        """
        parameters = torch.load(vmf_path, map_location=device, weights_only=True)
        w_logits = parameters['w_logits'].detach()
        theta_phi = parameters['theta_phi'].detach()
        log_kappa = parameters['log_kappa'].detach()
        theta = theta_phi[:log_kappa.size(0)]
        phi = theta_phi[log_kappa.size(0):]
        indices = np.lexsort((-phi.cpu().numpy(), -theta.cpu().numpy(), -log_kappa.cpu().numpy(), -w_logits.cpu().numpy()))
        sorted_indices = torch.tensor(indices)
        vmf_data = torch.stack([w_logits, log_kappa, theta, phi], dim=1) # Shape [num_nodes, 4]
        vmf_data = vmf_data[sorted_indices,:]
        # vmf_data = torch.flatten(vmf_data) # Shape [num_nodes * 4]
        return vmf_data.to(device)

        
class VMFDataset(Dataset):
    def __init__(self, rawdata_paths, sizes, dtype, samples=8192, device='cpu', force_reload=False):
        self.rawdata_paths = rawdata_paths
        self.sizes = sizes
        self.dtype = dtype
        self.device = device
        self.samples = samples
        self.force_reload = force_reload

    def __len__(self):
        return len(self.rawdata_paths)

    def __getitem__(self, idx):
        rp, sp = self.rawdata_paths[idx]
        raw_data, ray_data = VMFDataset.read_rawdata(rp, self.sizes,
                                                      samples=self.samples,
                                                      device=self.device, 
                                                      verbose=False, dtype=self.dtype, 
                                                      force_reload=self.force_reload)
        
        return {
            "samples": raw_data.to(self.device, dtype=self.dtype),
            # "samples": raw_data.to(self.device, dtype=torch.float32),
            "target": ray_data.reshape(-1).to(self.device, dtype=torch.float32),
            "w_data": get_gridX(self.sizes, dtype=self.dtype, device=self.device),
            "save_path": sp
        }

    @staticmethod
    def read_rawdata(filename, sizes, samples=8192, dtype=torch.float32, force_reload=False, verbose=False, device='cpu'):
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

        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        rawdata_np = np.fromfile(filename, dtype=np_dtype)

        rawdata_np = rawdata_np.reshape(-1, 4)
        x = rawdata_np[:, 0]
        phi = rawdata_np[:, 1] - np.pi
        r = np.sqrt(1 - x*x)
        y = r * np.cos(phi)
        z = r * np.sin(phi)

        # Create histogram edges
        x_edges = np.linspace(-1, 1, sizes[0] + 1, dtype=np_dtype)
        phi_edges = np.linspace(-np.pi, np.pi, sizes[1] + 1, dtype=np_dtype)

        # Statistics of ray data
        H, _, _ = np.histogram2d(x, phi, bins=[x_edges, phi_edges])
        raydata_tensor = torch.tensor(H, dtype=torch.float32, device=device).reshape(-1, 1)
        if raydata_tensor.shape[0] != sizes[0] * sizes[1]:
            print("Error: ray data shape mismatch!")
            sys.exit(1)
        area = 4 * math.pi / raydata_tensor.shape[0]
        raydata_tensor = raydata_tensor / torch.sum(raydata_tensor) / area
        
        raw_X = np.stack((x, y, z), axis=1)
        raw_X = np.random.permutation(raw_X)
        raw_num = min(samples, raw_X.shape[0])
        raw_X = raw_X[:raw_num, :]
        rawdata_tensor = torch.tensor(raw_X, dtype=dtype, device=device)
        
        if verbose:
            print("ray data shape:", raydata_tensor.shape)
            print("raw data shape:", rawdata_tensor.shape)

        torch.save({"raw": rawdata_tensor, "ray": raydata_tensor}, save_path)

        return rawdata_tensor, raydata_tensor

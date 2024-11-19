# Install required packages.
import os
import json
import argparse

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.utils.data import TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

os.environ['TORCH'] = torch.__version__


# Helper function for visualization.
import matplotlib.pyplot as plt
import numpy as np
import time


from models.model import *
from models.utils import *

def train_model(model, num_epochs, data_loader, device, optimizer, scheduler=None):
    loss_history = []
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            total_loss = 0
            count = 0
            # Iterate over pre-loaded data with tqdm for progress visualization
            for batch_data in data_loader:
                batch_data = batch_data.to(device)

                optimizer.zero_grad()

                # Forward pass
                predictions = model(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.pos,
                    batch_data.batch
                ).reshape(-1)
                # Compute the loss
                loss = criterion(predictions, batch_data.y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
            if scheduler is not None:
                scheduler.step()
            average_loss = total_loss/count
            loss_history.append(average_loss)
            pbar.set_postfix({
                'Loss': f"{average_loss:.6f}"
            })
            pbar.update(1)
        tqdm.write("Training complete!")
    return loss_history

def generate_filenames(folder_path, target_file, num_files):
    return [os.path.join(folder_path, f"{i}", target_file) for i in range(num_files)]

def load_dataset_config(base_path):
    config_path = os.path.join(base_path, "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def main(args):
    # Load dataset configuration
    
    dataset_config = load_dataset_config(args.base_path)
    device = args.device
    sizes = [64, 64]  # Default grid size
    X = get_gridX(sizes, device=device)

    # Initialize datasets
    datasets = []
    ray_datasets = []

    # Loop over datasets in the configuration
    for dataset_info in dataset_config["datasets"]:
        foam_name = dataset_info["foam_name"]
        file_num = dataset_info["file_num"]

        folder_path = os.path.join(args.base_path, foam_name)
        
        # Generate file paths
        graphs_path = generate_filenames(folder_path, "cells.txt", file_num)
        target_path = generate_filenames(folder_path, "vmf_parameters.pth", file_num)
        rays_path = generate_filenames(folder_path, "rawdataNonSpe.bin", file_num)
        
        # Load data for each file
        for i in range(file_num):
            # Load graph data
            parsed_cells = parse_cell_txt(graphs_path[i])
            graph_data = get_gnn_dataset(parsed_cells, device)
            
            # Load target data
            target_data = load_target_data(target_path[i], device)
            graph_data.y = target_data.reshape(-1)  
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

            # Load raw data
            _, ray_data, _ = load_rawdata(rays_path[i], sizes, device)
            ray_datasets.append(ray_data)
            
            datasets.append(graph_data)
    
    print(f"Loaded {len(datasets)} datasets.")

    # Define hyperparameters
    hyperparameters = {
        'num_features': datasets[0].x.size(1),
        'transformer_width': 256,
        'transformer_layers': 4,
        'transformer_heads': 8,
        'embedding_dim': 64,
        'pos_dim': datasets[0].pos.size(1),
        'dropout': 0.1,
        'hidden_dim': 128,
        'output_dim': 256,
        'num_epochs': args.num_epochs,
        'lr': args.learning_rate,
        'gamma': 0.9,
        'step_size': 500,
        'batch_size': 16
    }

    # Initialize model
    gcn_transformer = GCNTransformer(
        num_features=hyperparameters['num_features'],
        transformer_width=hyperparameters['transformer_width'],
        transformer_layers=hyperparameters['transformer_layers'],
        transformer_heads=hyperparameters['transformer_heads'],
        hidden_dim=hyperparameters['hidden_dim'],
        output_dim=hyperparameters['output_dim'],
        embedding_dim=hyperparameters['embedding_dim'],
        pos_dim=hyperparameters['pos_dim'],
        dropout=hyperparameters['dropout'],
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = Adam(gcn_transformer.parameters(), lr=hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=hyperparameters['gamma'],
        step_size=hyperparameters['step_size']
    )

    # Initialize data loader
    data_loader = DataLoader(
        datasets,
        batch_size=hyperparameters['batch_size'],
        shuffle=True
    )

    # Training starts from scratch
    loss_history = train_model(gcn_transformer, hyperparameters['num_epochs'], data_loader, device, optimizer, scheduler)
    time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    model_save_path = os.path.join(args.base_path, f"gnn_parameters_{time}.pth")

    checkpoint = {
        'epoch': 0,
        'model_state_dict': gcn_transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hyperparameters': hyperparameters,
        'loss_history': loss_history
    }

    torch.save(checkpoint, model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset configuration and train the GNN model.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path where the datasets and JSON configuration file are located.")


    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument("--device", type=str, default=default_device, help="Device to run the training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer.")

    args = parser.parse_args()
    main(args)
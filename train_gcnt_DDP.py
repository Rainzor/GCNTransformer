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
from torch.utils.data import random_split, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

from models.model import *
from models.utils import *


def generate_filenames(folder_path, target_file, num_files):
    return [os.path.join(folder_path, f"{i}", target_file) for i in range(num_files)]

def load_dataset_config(base_path):
    config_path = os.path.join(base_path, "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def log_sample_plots(model, datasets, X, sizes, writer, tag, num_samples=4, device='cpu'):
    """
    Logs 3D plots of predictions vs. references for random samples to TensorBoard.
    
    Parameters:
    - model: The trained model.
    - datasets: datasets (training or validation).
    - X: The grid for evaluation.
    - sizes: Shape of the output images.
    - writer: TensorBoard SummaryWriter instance.
    - tag: Tag to differentiate between train and validation.
    - num_samples: Number of random samples to visualize.
    - device: The device to run the inference on.
    """
    model.eval()
    random_samples = random.sample(list(datasets), min(num_samples, len(datasets)))

    for idx, sample in enumerate(random_samples):
        sample = sample.to(device)
        
        with torch.no_grad():
            # Get model predictions
            weights, mus, kappas = model.vmf_param(
                sample.x, sample.edge_index, sample.pos, sample.batch
            )
            img_predict = multi_vmf(weights.squeeze(), mus.squeeze(), kappas.squeeze(), X).cpu().numpy()
            img_predict = img_predict.reshape(sizes)

            # Get reference data
            tgt_w, tgt_m, tgt_k = extract_param(sample.y)
            img_reference = multi_vmf(tgt_w, tgt_m, tgt_k, X).cpu().numpy()
            img_reference = img_reference.reshape(sizes)

            # Plot and log to TensorBoard
            fig = plot_outputs_3d(img_reference, img_predict, sizes, return_fig=True)
            writer.add_figure(f'{tag}/Sample_{idx}', fig)

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure that all operations are deterministic on GPU (if possible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, hyperparameters, path, is_main):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler): The scheduler.
        epoch (int): Current epoch number.
        hyperparameters (dict): Dictionary of hyperparameters.
        path (str): Path to save the checkpoint.
        is_main (bool): Flag indicating if this is the main process.
    """
    if is_main:
        state_dict = model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'hyperparameters': hyperparameters
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")


def train_model(
    model, num_epochs, train_loader, val_loader, device, optimizer, scheduler=None, is_main=True, train_sampler=None):
    loss_history = {'train': [], 'val': []}

    with tqdm(total=num_epochs, desc="Training Progress", disable=not is_main) as pbar:
        for epoch in range(num_epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            model.train()
            total_train_loss = torch.zeros(2).to(device)

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                # Forward pass
                predictions = model(
                    data.x,
                    data.edge_index,
                    data.pos,
                    data.batch
                ).reshape(-1)

                # Compute loss
                loss = criterion(predictions, data.y)
                loss.backward()
                optimizer.step()

                total_train_loss[0] += loss.item()
                total_train_loss[1] += 1

            average_train_loss = float(total_train_loss[0] / total_train_loss[1])

            if scheduler:
                scheduler.step()

            # Validation phase
            if is_main:
                model.eval()
                total_val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        predictions = model(
                            data.x,
                            data.edge_index,
                            data.pos,
                            data.batch
                        ).reshape(-1)
                        loss = criterion(predictions, data.y)
                        total_val_loss += loss.item()
                        val_count += 1

                average_val_loss = total_val_loss / val_count
                loss_history['val'].append(average_val_loss)
                loss_history['train'].append(average_train_loss)
                pbar.set_postfix({'Train Loss': f"{average_train_loss:.6f}", 'Val Loss': f"{average_val_loss:.6f}"})
                pbar.update(1)

    return loss_history

def main():
    args = parse_args()

    # Initialize the process group
    init_process_group(backend='nccl')
    
    # Retrieve rank and world size from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])  # Typically used to assign the correct GPU

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Determine if this is the main process
    is_main = rank == 0

    if is_main:
        print(f"Running DDP on {world_size} GPU(s).")
        # Initialize TensorBoard SummaryWriter only in the main process
        timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        log_dir = os.path.join(args.base_path, 'tensorboard_logs', f'run_{timestamp}')
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None  # Non-main processes do not log

    # Load dataset configuration
    dataset_config = load_dataset_config(args.base_path)
    sizes = [64, 64, 64]  # Default grid size
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
            graph_data = get_gnn_dataset(parsed_cells)
            
            # Load target data
            target_data = load_target_data(target_path[i])
            graph_data.y = target_data.reshape(-1)  
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)

            # Load raw data
            _, ray_data, _ = load_rawdata(rays_path[i], sizes)
            ray_datasets.append(ray_data)
            
            datasets.append(graph_data)

    # Split dataset into training and validation sets (e.g., 80% train, 20% val)
    train_size = int(0.8 * len(datasets))
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    if is_main:
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

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
        'batch_size': args.batch_size  # Use batch_size from args
    }

    # Initialize model
    model = GCNTransformer(
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

    # Wrap the model with DDP, enable find_unused_parameters=True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=hyperparameters['gamma'],
        step_size=hyperparameters['step_size']
    )

    # Initialize DistributedSampler for training and validation
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        sampler=train_sampler,
        num_workers=4,  # Set to 0 for debugging if needed
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        sampler=val_sampler,
        num_workers=4,  # Set to 0 for debugging if needed
        pin_memory=True
    )

    # Training starts
    loss_history = train_model(
        model,
        hyperparameters['num_epochs'],
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler=scheduler,
        is_main=is_main,
        train_sampler=train_sampler
    )

    # Only the main process logs to TensorBoard and saves checkpoints
    if is_main:
        # Log losses to TensorBoard
        for epoch, train_loss in enumerate(loss_history['train']):
            writer.add_scalar('Loss/Train', train_loss, epoch)
        for epoch, val_loss in enumerate(loss_history['val']):
            writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Log final sample plots
        log_sample_plots(model.module, train_dataset, X, sizes, writer, tag='Train', num_samples=4, device=device, epoch=hyperparameters['num_epochs']-1)
        log_sample_plots(model.module, val_dataset, X, sizes, writer, tag='Validation', num_samples=4, device=device, epoch=hyperparameters['num_epochs']-1)

        # Save the model checkpoint
        timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        output_dir = os.path.join(args.base_path, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, f"gnn_parameters_{timestamp}.pth")

        save_checkpoint(model, optimizer, scheduler, hyperparameters['num_epochs'], hyperparameters, model_save_path, is_main=True)

        # Close the TensorBoard writer
        writer.close()

    # Clean up the process group
    destroy_process_group()
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training")
    parser.add_argument('--base_path', type=str, required=True, help='Base path for data and outputs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

if __name__ == "__main__":
    # Configure logging to display warnings
    logging.basicConfig(level=logging.WARNING)
    main()
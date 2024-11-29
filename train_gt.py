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

from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import random

os.environ['TORCH'] = torch.__version__


# Helper function for visualization.
import matplotlib.pyplot as plt
import numpy as np
import time


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

def save_checkpoint(model, optimizer, scheduler, epoch, hyperparameters, path):
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hyperparameters': hyperparameters
    }

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def train(model, iterator, optimizer, criterion, device):
    
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(
            batch.x,
            batch.edge_index,
            batch.pos,
            batch.batch
        ).reshape(-1)
        
        loss = criterion(predictions, batch.y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.pos,
                batch.batch
            ).reshape(-1)

            loss = criterion(predictions, batch.y)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(
    model, num_epochs, train_loader, val_loader, device, optimizer, scheduler=None,smooth_f= 0.5, diverge_th = 2.0):
    loss_history = {'train': [], 'val': []}
    best_loss = float('inf')
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, F.mse_loss, device)
            
            if scheduler:
                scheduler.step()

            val_loss = evaluate(model, val_loader, F.mse_loss, device)

            if epoch > 0:
                train_loss = smooth_f * train_loss + (1 - smooth_f) * loss_history['train'][-1]
                val_loss = smooth_f * val_loss + (1 - smooth_f) * loss_history['val'][-1]
            
            loss_history['train'].append(train_loss)
            loss_history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
            
            if val_loss > diverge_th * best_loss and epoch > 100:
                print("Stopping early as the val-loss diverged.")
                break

            if train_loss < 1e-4:
                print("Stopping early as the train-loss is too low.")
                break

            pbar.set_postfix({'Train Loss': f"{train_loss:.6f}", 'Val Loss': f"{val_loss:.6f}"})
            pbar.update(1)
            
    return loss_history

def main(args):
    # Load dataset configuration
    dataset_config = load_dataset_config(args.base_path)
    device = torch.device(args.device)
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
            graph_data = get_gnn_dataset(parsed_cells, device=device)
            
            # Load target data
            target_data = load_target_data(target_path[i], device)
            graph_data.y = target_data.reshape(-1)  
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)

            # Load raw data
            _, ray_data, _ = load_rawdata(rays_path[i], sizes, device)
            ray_datasets.append(ray_data)
            
            datasets.append(graph_data)
    
    # Split dataset into training and validation sets (e.g., 80% train, 20% val)
    train_size = int(0.8 * len(datasets))
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # Define hyperparameters
    hyperparameters = {
        'num_features': datasets[0].x.size(1),
        'gnn_dim': 128,
        'transformer_width': 256,
        'transformer_layers': 4,
        'transformer_heads': 8,
        'embedding_dim': 32,
        'pos_dim': datasets[0].pos.size(1),
        'dropout': 0.1,
        'hidden_dim': 512,
        'output_dim': 256,
        'num_epochs': args.num_epochs,
        'lr': args.learning_rate,
        'gamma': 0.9,
        'step_size': 500,
        'batch_size': args.batch_size
    }

    # Initialize model
    graph_transformer = GraphTransformer(
        num_features=hyperparameters['num_features'],
        gnn_dim=hyperparameters['gnn_dim'],
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
    optimizer = Adam(graph_transformer.parameters(), lr=hyperparameters['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=hyperparameters['gamma'],
        step_size=hyperparameters['step_size']
    )

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False
    )

    # Training starts from scratch
    loss_history = train_model(
        graph_transformer,
        hyperparameters['num_epochs'],
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler    
    )
    
    # Initialize TensorBoard SummaryWriter
    timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    if(args.model == ""):
        log_dir = os.path.join(args.base_path, 'tensorboard_logs', f'run_{timestamp}')
    else:
        log_dir = os.path.join(args.base_path, 'tensorboard_logs', f'{args.model}_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)

    # Log losses to TensorBoard
    for epoch, train_loss in enumerate(loss_history['train']):
        writer.add_scalar('Loss/Train', train_loss, epoch)
    for epoch, val_loss in enumerate(loss_history['val']):
        writer.add_scalar('Loss/Validation', val_loss, epoch)
    # Log final sample plots
    log_sample_plots(graph_transformer, train_dataset, X, sizes, writer, tag='Train', num_samples=4, device=device)
    log_sample_plots(graph_transformer, val_dataset, X, sizes, writer, tag='Validation', num_samples=4, device=device)

    # Save the model checkpoint
    timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    output_dir = os.path.join(args.base_path, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    if(args.model == ""):
        model_save_path = os.path.join(output_dir, f"gcn_transformer_{timestamp}.pth")
    else:
        model_save_path = os.path.join(output_dir, f"{args.model}_{timestamp}.pth")

    save_checkpoint(graph_transformer, optimizer, scheduler, hyperparameters['num_epochs'], hyperparameters, model_save_path)

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset configuration and train the GNN model.")
    parser.add_argument("-p", "--base_path", type=str, required=True, help="Base path where the datasets and JSON configuration file are located.")

    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument("--device", type=str, default=default_device, help="Device to run the training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("-n", "--num_epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer.")
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument("--model", type=str, default="", help="Name to train (e.g., 'graph_transformer').")
    args = parser.parse_args()
    main(args)
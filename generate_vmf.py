import threading
import os
import glob
import sys
import time
import argparse
import math
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from models.model import vMFMixtureModel
from models.loss import negative_log_likelihood
from models.data_utils.dataset import VMFDataset

def train_model(model_id, vmf, optimizer, scheduler, dataset, hyperparams, device, parent_pbar=None, verbose=False):
    kl_lambda = hyperparams['kl_lambda']
    l1_lambda = hyperparams['l1_lambda']
    l2_lambda = hyperparams['l2_lambda']
    epochs = hyperparams['epochs']

    # Load data and move it to the device
    samples = dataset["samples"].to(device, non_blocking=True)  # Shape: (bz, data_sizes, 3)
    target = dataset["target"].to(device, non_blocking=True)    # Shape: (bz, sizes)
    w_data = dataset["w_data"].to(device, non_blocking=True)    # Shape: (bz, sizes, 3)
    save_path = dataset.get("save_path", None)  # Save path for the trained model

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    def update_model():
        optimizer.zero_grad()

        # Use autocast for mixed precision during the forward pass
        with autocast('cuda'):  # Corrected to 'cuda'
            pi, mu, kappa = vmf()
            loss, loss_dict = negative_log_likelihood(
                samples, pi, mu, kappa,
                kl_lambda=kl_lambda, l1_lambda=l1_lambda, l2_lambda=l2_lambda,
                p=target, w=w_data
            )

        # Scale the loss and compute gradients in float32
        scaler.scale(loss).backward()

        # Step the optimizer with gradient scaling
        scaler.step(optimizer)
        scaler.update()

        # Update scheduler if loss is below a threshold
        if loss.item() < 4.0:
            scheduler.step()
        return loss, loss_dict

    vmf.train()
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 1500
    min_delta = 0.0001

    # Training loop with tqdm progress bar if verbose
    if verbose:
        loss_history = []
        with tqdm(total=epochs, desc=f'Model {model_id} Training', leave=False) as pbar:
            for epoch in range(epochs):
                loss, loss_dict = update_model()
                loss_history.append(loss.item())
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'KL': f'{loss_dict["NLL"].item():.4f}',
                    'Rec': f'{loss_dict["Rec"].item():.4f}',
                    'L2': f'{loss_dict["L2"].item():.4f}'
                })
                pbar.update(1)

                # Early stopping logic based on improvement in loss
                if loss.item() < best_loss - min_delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    break

            if parent_pbar is not None:
                parent_pbar.set_postfix({
                    'ID': f'{model_id}',
                    'Loss': f'{loss.item():.4f}',
                    'step': f'{epoch}'
                })
    else:
        for epoch in range(epochs):
            loss, loss_dict = update_model()

    # Save model parameters if save path is provided
    if save_path is not None:
        for i, sp in enumerate(save_path):
            state_dict = {
                "w_logits": vmf.w_logits[i].clone().detach().cpu(),
                "theta_phi": vmf.theta_phi[i].clone().detach().cpu(),
                "log_kappa": vmf.log_kappa[i].clone().detach().cpu(),
            }
            torch.save(state_dict, sp)


def train_process(model, optimizer, dataset, hyperparams, device):
    # Load the model and dataset
    model.to(device)
    dataset['samples'] = dataset['samples'].to(device) # Shape: (bz, data_sizes, 3)
    dataset['target'] = dataset['target'].to(device) # Shape: (bz, data_sizes)
    dataset['w_data'] = dataset['w_data'].to(device) # Shape: (bz, data_sizes, 3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    # Start training
    train_model(0, model, optimizer, scheduler, dataset, hyperparams, device)


def getpath(output_json_path):
    paths = []
    if os.path.isfile(output_json_path):
        subdir = os.path.dirname(output_json_path)
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                rawShellDataPath = os.path.join(subdir, entry["rawShell"])
                rawFirstDataPath = os.path.join(subdir, entry["rawFirst"])
                vmfShellPath = os.path.join(subdir, entry["vmfShell"])
                vmfFirstPath = os.path.join(subdir, entry["vmfFirst"])
                paths.append((rawFirstDataPath, vmfFirstPath))
                paths.append((rawShellDataPath, vmfShellPath))
    else:
        print(f"Warning: The file '{output_json_path}' does not exist, skipping this directory.")
    return paths

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Script to train vMF mixture models.')
    parser.add_argument('--base_path', type=str, default='',
                        help='Base path for data files. Example: datasets/')
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to the data file. Example: datasets/raw_data/foam0/0')
    parser.add_argument('--data_num', type=int, default=-1,
                        help='Number of data files to process. Default is -1 (all).')
    parser.add_argument('--sample_num', type=int, default=8192,
                        help='Number of samples in each dataset. Default is 8192.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('--components_num', type=int, default=64,
                        help='Number of components in the vMF mixture model. Default is 64.')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs. Default is 10000.')
    parser.add_argument('--kl', type=float, default=1.0,
                        help='Weight for KL divergence. Default is 1.')
    parser.add_argument('--l1', type=float, default=10.0,
                        help='Weight for L1 regularization. Default is 10.')
    parser.add_argument('--l2', type=float, default=0.5,
                        help='Weight for L2 regularization. Default is 0.5.')
    parser.add_argument('--lr', type=float, default=5e-2,
                        help='Learning rate. Default is 5e-2.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (for Adam optimizer). Default is 1e-5.')
    parser.add_argument('--sizes', type=str, default='64,64',
                        help='Sizes as a comma-separated list of integers. Example: "64,64"')
    parser.add_argument("--dtype", type=int, default="32",
                        help="Data type of raw data. Default is 32 (float32).")
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Whether using multi-GPU training.')

    parser.add_argument('--force_reload', action='store_true',
                        help='Whether to force reload the data.')   

    args = parser.parse_args()
    if args.data_path=="" and args.base_path=="":
        print("Error: Either --data_path or --base_path must be specified.")
        sys.exit(1)
    elif args.data_path=="":
        print(f"Training vMF mixture models with the datasets in {args.base_path}")
    else:
        print(f"Training vMF mixture models with the dataset in {args.data_path}")

    # Initialize hyperparameters
    base_path = args.base_path
    data_path = args.data_path
    batch_size = args.batch_size
    components_num = args.components_num
    epochs = args.epochs
    kl_lambda = args.kl
    l1_lambda = args.l1
    l2_lambda = args.l2
    print(f"kl_lambda: {kl_lambda}, l1_lambda: {l1_lambda}, l2_lambda: {l2_lambda}")
    learning_rate = args.lr
    weight_decay = args.weight_decay
    dtype = torch.float32 if args.dtype==32 else torch.float16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_gpu = max(min(torch.cuda.device_count()//2, 4), 1)

    available_devices = [torch.device(f'cuda:{i}') for i in range(max_gpu)]
    print(f"Available devices number: {len(available_devices)}")

    sizes = [int(s) for s in args.sizes.split(',')]

    hyperparams = {
        "kl_lambda": kl_lambda,
        "l1_lambda": l1_lambda,
        "l2_lambda": l2_lambda,
        "epochs": epochs,
    }


    # Get the paths of the raw data files
    rawdata_paths = [] 
    if data_path=="":
        if not os.path.isdir(base_path):
            print(f"Error: The base path '{base_path}' does not exist.")
            sys.exit(1)        
        pattern = os.path.join(base_path, "*", "*", "config.json")
        json_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(json_files)} config.json files.")
        for json_file in json_files:
            rawdata_paths.extend(getpath(json_file))


    elif base_path=="":
        if not os.path.isdir(data_path):
            print(f"Error: The data path '{data_path}' does not exist.")
            sys.exit(1)
        json_file = os.path.join(data_path, "config.json")
        rawdata_paths.extend(getpath(json_file))
    else:
        print("Error: Either --data_path or --base_path must be specified.")
        sys.exit(1)

    total_num = len(rawdata_paths)
    
    if args.data_num > 0:
        total_num = min(args.data_num, total_num)
        rawdata_paths = rawdata_paths[:total_num]
    
    if total_num == 0:
        print("Error: No data files found.")
        sys.exit(1)

    # Create the datasets
    print(f"Create {total_num} datasets......")     
    vmf_datasets = VMFDataset(rawdata_paths, sizes, dtype,samples=args.sample_num, force_reload=args.force_reload)
    datasets_loader = torch.utils.data.DataLoader(vmf_datasets,batch_size=batch_size, shuffle=False)
    model_num = len(datasets_loader)
    batch_size_list = np.ones(model_num, dtype=int) * batch_size
    if total_num % batch_size != 0:
        batch_size_list[-1] = total_num % batch_size
    
    assert sum(batch_size_list) == total_num, f"Batch size mismatch, total_num: {total_num}, batch_size_list: {batch_size_list}"
    # Train the models
    if args.multi_gpu:
        print(f"Using multi-GPU training.")
        start_time = time.time()

        models = []
        optimizers = []
        # schedulers = []
        for i in range(model_num):
            local_bz = batch_size_list[i]
            device_i = available_devices[i % len(available_devices)]  # Set device for each process
            vmf = vMFMixtureModel(components_num=components_num,
                                batch_size=local_bz
                                ).to(device_i)
            optimizer = torch.optim.Adam(vmf.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
            models.append(vmf)
            optimizers.append(optimizer)
            # schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5))

        max_processes = max_gpu
        processes = []
        finished_processes = 0
        with tqdm(total=model_num, desc="Training Models") as pbar:
            for i, dataset in enumerate(datasets_loader):
                device_i = available_devices[i % len(available_devices)]  # Set device for each process
                
                p = mp.Process(target=train_process, args=(models[i], optimizers[i], dataset, hyperparams, device_i))
                processes.append(p)
                p.start()
                if len(processes) >= max_processes:
                    for p in processes:
                        p.join()
                    processes = []
                    finished_processes += max_processes
                    # print(f"{finished_processes}/{model_num} models trained.")
                    pbar.update(max_processes)

            for p in processes:
                p.join()
            pbar.update(len(processes))
        
        cost_time = time.time() - start_time
        print(f"Training completed! Total time: {cost_time//60:.0f}m {cost_time%60:.0f}s")
    else:
        datasets = []
        optimizers = [] 
        models = []
        schedulers = []   

        print(f"Training {model_num} models......")
        with tqdm(total=model_num, desc='Training Models') as pbar:
            for i, dataset in enumerate(datasets_loader):
                # Create a new model for each dataset
                local_bz = batch_size_list[i]
                model = vMFMixtureModel(components_num=components_num, batch_size=local_bz).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*math.sqrt(local_bz), weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

                # Train the model
                train_model(i, model, optimizer, scheduler, dataset, hyperparams, device, parent_pbar=pbar, verbose=True)
                pbar.update(1)
        
        print(f"Training completed!")
    


if __name__ == "__main__":
    mp.set_start_method('spawn')  # 确保使用 'spawn' 启动方式
    main()


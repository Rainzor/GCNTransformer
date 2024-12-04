import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import glob
import torch.multiprocessing as mp


import numpy as np

import threading
import os
import sys
import time
import argparse
import math
import json

from models.model import vMFMixtureModel
from models.loss import negative_log_likelihood
from models.data_utils.utils import load_rawdata, get_gridX


def train_model(model_id, vmf, optimizer, scheduler,  dataset, hyperparams, device, parent_pbar=None, save_path=None, verbose=False):
    kl_lambda = hyperparams['kl_lambda']
    l1_lambda = hyperparams['l1_lambda']
    l2_lambda = hyperparams['l2_lambda']
    num_epochs = hyperparams['num_epochs']

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
        scheduler.step()
        return loss, loss_dict

    vmf.train()
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 1000
    min_delta = 0.0002
    if verbose:
        loss_history = []
        with tqdm(total=num_epochs, desc=f'Model {model_id} Training', leave=False) as pbar:
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
        for epoch in range(num_epochs):
            update_model()

    if save_path is not None:
        torch.save(vmf.state_dict(), save_path)

def train_process(model, optimizer, dataset, hyperparams, device, save_path):
    # 将模型、数据加载到指定设备
    model.to(device)
    dataset['samples'] = dataset['samples'].to(device)
    dataset['target'] = dataset['target'].to(device)
    dataset['w_data'] = dataset['w_data'].to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    # 开始训练
    train_model(0, model, optimizer, scheduler, dataset, hyperparams, device, save_path=save_path)


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
    parser.add_argument('--num_data', type=int, default=-1,
                        help='Number of data files to process. Default is -1 (all).')
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
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Learning rate. Default is 5e-3.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (for Adam optimizer). Default is 1e-5.')
    parser.add_argument('--sizes', type=str, default='64,64',
                        help='Sizes as a comma-separated list of integers. Example: "64,64"')
    parser.add_argument("--dtype", type=int, default="32",
                        help="Data type of raw data. Default is 32 (float32).")
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Whether using multi-GPU training.')
    args = parser.parse_args()
    if args.data_path=="" and args.base_path=="":
        print("Error: Either --data_path or --base_path must be specified.")
        sys.exit(1)
    elif args.data_path=="":
        print(f"Training vMF mixture models with the datasets in {args.base_path}")
    else:
        print(f"Training vMF mixture models with the dataset in {args.data_path}")
    # 初始化参数
    base_path = args.base_path
    data_path = args.data_path
    num_components = args.num_components
    num_epochs = args.num_epochs
    kl_lambda = args.kl_lambda
    l1_lambda = args.l1_lambda
    l2_lambda = args.l2_lambda
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    dtype = np.float32 if args.dtype==32 else np.float16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_gpu = min(torch.cuda.device_count(), 4)
    available_devices = [torch.device(f'cuda:{i}') for i in range(max_gpu)]
    print(f"Available devices number: {len(available_devices)}")

    sizes = [int(s) for s in args.sizes.split(',')]

    hyperparams = {
        "kl_lambda": kl_lambda,
        "l1_lambda": l1_lambda,
        "l2_lambda": l2_lambda,
        "num_epochs": num_epochs
    }
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

    num_models = len(rawdata_paths)
    
    if args.num_data > 0:
        num_models = min(args.num_data, num_models)
        rawdata_paths = rawdata_paths[:num_models]
    
    if num_models == 0:
        print("Error: No data files found.")
        sys.exit(1)
        
    X = get_gridX(sizes)
    print(f"Training {num_models} models......")
    if args.multi_gpu:
        print(f"Using multi-GPU training.")
        start_time = time.time()
        models = []
        optimizers = []
        for i in range(num_models):
            device_i = available_devices[i % len(available_devices)]  # Set device for each process
            vmf = vMFMixtureModel(num_components=num_components).to(device_i)
            optimizer = torch.optim.Adam(vmf.parameters(), lr=learning_rate, weight_decay=weight_decay)
            models.append(vmf)
            optimizers.append(optimizer)
            
        processes = []
        for i in range(num_models):
            device_i = available_devices[i % len(available_devices)]  # Set device for each process

            rp, sp = rawdata_paths[i]
            raw_data, ray_data = load_rawdata(rp, sizes, verbose=False, dtype=dtype)
            dataset = {
                "samples": raw_data.to(device_i),
                "target": ray_data.reshape(-1).to(device_i),
                "w_data": X.clone().to(device_i)
            }
            
            p = mp.Process(target=train_process, args=(
                models[i], optimizers[i], dataset, hyperparams, device_i, sp))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
        cost_time = time.time() - start_time
        print(f"Training completed! Total time: {cost_time//60:.0f}m {cost_time%60:.0f}s")
    else:
        save_paths = []
        datasets = []
        optimizers = [] 
        models = []
        schedulers = []   
        print(f"Loading {num_models} datasets......")     
        for i in range(num_models):
            # Create model, optimizer, scheduler
            vmf = vMFMixtureModel(num_components=num_components).to(device)
            optimizer = torch.optim.Adam(vmf.parameters(), lr=learning_rate, weight_decay=weight_decay)
            models.append(vmf)
            optimizers.append(optimizer)
            schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5))
            
            # Load raw data
            rp, sp = rawdata_paths[i]
            raw_data, ray_data = load_rawdata(rp, sizes, verbose=False, dtype=dtype)
            dataset = {
                "samples": raw_data.to(device),
                "target": ray_data.reshape(-1).to(device),
                "w_data": X.clone().to(device)
            }
            datasets.append(dataset)
            save_paths.append(sp)

        with tqdm(total=num_models, desc='Training Models') as pbar:
            for i in range(num_models):
                train_model(i, models[i], optimizers[i], schedulers[i], datasets[i], hyperparams, device, parent_pbar=pbar, save_path=save_paths[i], verbose=True)
                pbar.update(1)
        
        print(f"Training completed!")
    


if __name__ == "__main__":
    mp.set_start_method('spawn')  # 确保使用 'spawn' 启动方式
    main()


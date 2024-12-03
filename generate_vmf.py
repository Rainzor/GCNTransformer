import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


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
from models.data_utils.dataset import load_rawdata

# Set the device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model_id, vmf, optimizer, scheduler,  dataset, hyperparams, device, parent_pbar=None, save_path=None, verbose=False):
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

    # 保存模型参数
    if save_path is not None:
        torch.save(vmf.state_dict(), save_path)


def getpath(subdir):
    paths = []

    if not os.path.isdir(subdir):
        print(f"Error: The directory '{subdir}' does not exist.")
        sys.exit(1)
    output_json_path = os.path.join(subdir, "output.json")
    if os.path.isfile(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                # 获取 rawNonSpe 文件路径
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
    parser.add_argument('--num_files', type=int, default=1,
                        help='Number of data files. Default is 1.')
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to the data file. Example: datasets/raw_data/foam0/0')
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
                        help="Data type of raw data")
    # 默认设备设为所有可用GPU，如果没有GPU则为CPU
    if torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    parser.add_argument('--device', type=str, default=default_device,
                        help='Computation device. Defaults to CUDA if available, otherwise CPU.')

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
    num_files = args.num_files
    num_components = args.num_components
    num_epochs = args.num_epochs
    kl_lambda = args.kl_lambda
    l1_lambda = args.l1_lambda
    l2_lambda = args.l2_lambda
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    device = args.device
    dtype = np.float32 if args.dtype==32 else np.float16

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
        
        json_file = os.path.join(base_path, "data.json")
        if not os.path.isfile(json_file):
            print(f"Warning: The file '{json_file}' does not exist.")
            sys.exit(1)
        
        folder_num = 0
        folder_name =""
        rotatation = 0
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            folder_name = data["name"]
            folder_num = data["number"]
            rotation = data["rotation"]
        for i in range(folder_num):
            for j in range(rotation):
                subdir = os.path.join(base_path, folder_name+f"{i}", f"{j}")
                rawdata_paths.extend(getpath(subdir))


    else:
        if not os.path.isdir(data_path):
            print(f"Error: The data path '{data_path}' does not exist.")
            sys.exit(1)
        rawdata_paths.extend(getpath(data_path))

    num_models = len(rawdata_paths)

    datasets = []  
    optimizers = [] 
    models = []
    save_paths = []
    schedulers = []
    for i in range(num_models):
        vmf = vMFMixtureModel(num_components=num_components).to(device)
        optimizer = torch.optim.Adam(vmf.parameters(), lr=learning_rate, weight_decay=weight_decay)
        models.append(vmf)
        optimizers.append(optimizer)
        schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5))
        # 加载对应的数据集
        rp, sp = rawdata_paths[i]
        raw_data, ray_data, X = load_rawdata(rp, sizes, device, verbose=False, dtype=dtype)
        dataset = {
            "samples": raw_data.clone().detach(),
            "target": ray_data.reshape(-1).to(device),
            "w_data": X.clone().detach(),
        }
        save_paths.append(sp)
        datasets.append(dataset)

    print(f"Training {num_models} models......")
    with tqdm(total=num_models, desc='Training Models') as pbar:
        for i in range(num_models):
            train_model(i, models[i], optimizers[i],schedulers[i], datasets[i], hyperparams, device, parent_pbar=pbar,save_path=save_paths[i], verbose=True)
            pbar.update(1)
    
    print(f"Training completed!")

if __name__ == "__main__":
    main()


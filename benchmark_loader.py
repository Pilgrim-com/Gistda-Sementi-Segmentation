
import time
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from Tools import DatasetUtility
import argparse
from tqdm import tqdm

def benchmark():
    # Load config
    with open("cfg.json", 'r') as f:
        cfg = json.load(f)
    
    # Setup dataset (using DeepGlobe as default or try to find one)
    dataset_name = "DeepGlobe" 
    # Check if dir exists, if not try first available
    if not os.path.exists(cfg["Datasets"][dataset_name]["train_dir"] if os.path.isabs(cfg["Datasets"][dataset_name]["train_dir"]) else os.path.join(os.getcwd(), cfg["Datasets"][dataset_name]["train_dir"].strip("./"))):
        # simple fallback logic
        for k in cfg["Datasets"]:
            if k == "base_dir": continue
            dataset_name = k
            break
            
    print(f"Benchmarking Dataset: {dataset_name}")
    
    # Initialize dataset
    try:
        # Assuming LinkNet as model name just for init
        ds = DatasetUtility.DeepGlobe(cfg, "LinkNet", dataset_name, "training_settings")
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        # Try generic class if specific one fails or just stop
        return

    print(f"Dataset size: {len(ds)}")
    
    # Measure __getitem__ latency
    print("Uncached access (first 10 items):")
    total_time = 0
    for i in range(min(10, len(ds))):
        t0 = time.time()
        _ = ds[i]
        dt = time.time() - t0
        total_time += dt
        print(f"Item {i}: {dt:.4f} sec")
    
    avg = total_time / min(10, len(ds))
    print(f"Average time per item: {avg:.4f} sec")
    
    # Estimate epoch time
    batch_size = cfg["training_settings"]["batch_size"]
    num_workers = 4 # as per train_unet.py
    
    # Theoretical throughput (assuming perfect parallelization)
    # throughput = num_workers / avg_time_per_item
    throughput = num_workers / avg
    items_per_sec = throughput
    
    total_items = len(ds)
    estimated_epoch_time = total_items / items_per_sec
    
    print(f"\n--- Estimation ---")
    print(f"Avg processing time per image (CPU): {avg:.4f} sec")
    print(f"With {num_workers} workers, max throughput: {items_per_sec:.2f} img/sec")
    print(f"Estimated time per epoch ({total_items} images): {estimated_epoch_time/60:.2f} minutes ({estimated_epoch_time:.2f} sec)")
    print(f"Actual time will be higher due to overhead and GPU sync.")

if __name__ == "__main__":
    benchmark()

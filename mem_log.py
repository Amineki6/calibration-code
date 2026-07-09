import psutil
import os
import subprocess
import torch
import logging


def check_large_objects():
    import gc
    gc.collect()
    
    # Find largest objects
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    
    large_tensors = sorted(tensors, key=lambda x: x.numel() * x.element_size(), reverse=True)[:10]
    
    logging.info("Largest tensors in memory:")
    for i, t in enumerate(large_tensors):
        size_mb = t.numel() * t.element_size() / 1e6
        logging.info(f"  {i}: {t.shape} on {t.device}, {size_mb:.2f} MB")


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    logging.info(f"Process RSS: {mem_info.rss / 1e9:.2f} GB")
    logging.info(f"Process VMS: {mem_info.vms / 1e9:.2f} GB")
    
    # System memory
    vm = psutil.virtual_memory()
    logging.info(f"System RAM: {vm.used / 1e9:.2f} / {vm.total / 1e9:.2f} GB ({vm.percent}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logging.info(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")        


def check_shm_usage():
    result = subprocess.run(['df', '-h', '/dev/shm'], capture_output=True, text=True)
    logging.info(f"Shared memory usage:\n{result.stdout}")

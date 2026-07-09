import re
from tqdm import tqdm
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from typing import Any, cast
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import coolname
from lightning_module import CXPLightningModule

from dataset import CXP_dataset
from model import CXP_Model

class TeeStream:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
    def write(self, data):
        self.stream.write(data)
        clean_data = self.ansi_escape.sub('', data)
        self.log_file.write(clean_data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return hasattr(self.stream, 'isatty') and self.stream.isatty()

    def fileno(self):
        if hasattr(self.stream, 'fileno'):
            return self.stream.fileno()
        raise OSError("No fileno")

    def __getattr__(self, attr):
        return getattr(self.stream, attr)



def identify_error_set(model, method, loader, device, max_batches=None):
    """
    Evaluates the model on the loader and returns indices of misclassified examples.
    Used for JTT Stage 1.
    """
    model.eval()
    error_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Identifying Error Set", leave=False)):
            if max_batches is not None and i >= max_batches:
                break

            batch_data = batch
            inputs = batch_data.inputs.to(device)
            labels = batch_data.labels.to(device)
            
            # Forward pass
            model_output = model(inputs)
            logits = model_output[0]
            preds = (torch.sigmoid(logits) > 0.5).float()

            indices = batch_data.indices.view(-1)

            # Identify mismatches as a PyTorch boolean tensor
            mismatches = (preds.view(-1) != labels.view(-1)).detach().cpu()
            
            # Map robustly using the returned indices
            error_indices.extend(indices.cpu()[mismatches].tolist())

    return set(error_indices)

def get_jtt_loader(dataset, error_indices, config):
    """
    Creates a DataLoader with upweighted error set examples.
    """
    # Check if we have error indices
    if not error_indices:
        logging.warning("No errors found in identification stage! Returning standard loader.")
        return DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, 
            num_workers=config.num_workers, pin_memory=True
        )

    # Convert JTT behavior from dataset concatenation to sample weights to prevent GPU overflow.
    # Populate weights directly while keeping the batch shape stable.
    dataset.weights = np.ones(len(dataset), dtype=np.float32)
    
    # Upweight the error indices by lambda
    if len(error_indices) > 0:
        error_idx_array = list(error_indices)
        dataset.weights[error_idx_array] = float(config.jtt_lambda)
        
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    return loader

def run_jtt_stage1(config, method, datamodule, study_root, trial_number, run_name, wandb_group, debug):
    logging.info("--- JTT Stage 1: Identification Phase ---")
    
    if config.balance_val:
        train_csv = config.csv_dir / 'train_drain_shortcut_v2.csv'
    else:
        train_csv = config.csv_dir / 'train_drain_shortcut.csv'

    backbone = getattr(config, 'backbone', 'densenet')

    stage1_train_dataset = CXP_dataset(
        config.data_dir,
        train_csv,
        augment=False,
        compute_sample_weights=False,
        split_name='train_stage1',
        backbone=backbone,
    )

    id_dataset = CXP_dataset(
        config.data_dir,
        train_csv,
        augment=False,
        compute_sample_weights=False,
        split_name='train_identification',
        backbone=backbone,
    )

    if debug:
        from torch.utils.data import Subset
        stage1_train_dataset = Subset(stage1_train_dataset, range(min(len(stage1_train_dataset), 50)))
        id_dataset = Subset(id_dataset, range(min(len(id_dataset), 50)))

    stage1_train_loader = DataLoader(
        stage1_train_dataset,
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    # Init Stage 1 Model
    model_1 = CXP_Model(method) # .to(device) is handled by PL
    
    # Stage 1 Logger
    stage1_run_name = f"{run_name}_stage1" if run_name else f"trial_{trial_number}_stage1"
    coolname_any = cast(Any, coolname)
    if hasattr(coolname_any, "generate_slug"):
        run_id_suffix = coolname_any.generate_slug(2)
    else:
        run_id_suffix = "-".join(coolname_any.generate(2))

    wandb_logger_1 = WandbLogger(
        project="cxr_optuna_study", 
        group=wandb_group, 
        name=stage1_run_name,
        config=config.__dict__,
        save_dir=str(study_root),
        id=f"{stage1_run_name}_{run_id_suffix}" # Unique ID
    )

    # Stage 1 Module
    pl_module_1 = CXPLightningModule(model_1, method, config)
    
    # Stage 1 Checkpoint
    checkpoint_callback_1 = ModelCheckpoint(
        dirpath=study_root / "checkpoints",
        filename=f"trial_{trial_number}_jtt_stage1",
        save_top_k=0, # We don't really need to save best here, just train for Epochs
        save_last=True
    )

    # Trainer Stage 1
    trainer_1 = pl.Trainer(
        max_epochs=config.jtt_duration,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger_1,
        callbacks=[checkpoint_callback_1],
        enable_progress_bar=True,
        log_every_n_steps=10 if debug else 50,
        enable_checkpointing=True,
        inference_mode=False # Important for manual eval loop later if needed? No, PL handles it.
    )
    
    # Explicitly call setup to populate datasets (like val_dataset) before getting dataloaders
    datamodule.setup('fit')
    
    trainer_1.fit(
        pl_module_1, 
        train_dataloaders=stage1_train_loader,
        val_dataloaders=datamodule.val_dataloader()
    )
    
    # Identify Error Set
    # We need to run inference on training set strictly sequentially
    # Manual identification using the trained module
    logging.info("Identifying Error Set...")
    pl_module_1.eval()
    pl_module_1.freeze()
    
    # Use a raw dataloader for identification strictly sequentially/non-shuffled
    id_loader = DataLoader(
        id_dataset,
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    device = pl_module_1.device
    error_indices = identify_error_set(pl_module_1.model, method, id_loader, device)
    n_train_len = len(id_dataset)
    logging.info(f"JTT Stage 1 Complete. Found {len(error_indices)} errors out of {n_train_len} examples.")
    
    # Pass error indices to DataModule for Stage 2
    datamodule.set_jtt_error_indices(error_indices)
    
    # Clean up Stage 1
    wandb_logger_1.experiment.finish()
    del trainer_1, pl_module_1, model_1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logging.info("--- JTT Stage 2: Upweighted Training Phase ---")
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from typing import Optional

from config import ExperimentConfig
from dataset import CXP_dataset
from utils import get_jtt_loader # Retain these helpers if needed, or refactor

class CXPDataModule(pl.LightningDataModule):
    def __init__(self, config: ExperimentConfig, debug: bool = False):
        super().__init__()
        self.config = config
        self.debug = debug
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset_aligned = None
        self.test_dataset_misaligned = None
        self.train_sampler = None
        self.jtt_error_indices = None # For JTT Stage 2

    def setup(self, stage: Optional[str] = None):
        # Determine CSV filenames based on config
        # Logic copied from utils.py get_dataloaders
        
        if self.config.balance_val:
            train_csv = self.config.csv_dir / 'train_drain_shortcut_v2.csv'
            val_csv = self.config.csv_dir / 'val_drain_shortcut_v2.csv'
        else:
            train_csv = self.config.csv_dir / 'train_drain_shortcut.csv'
            val_csv = self.config.csv_dir / 'val_drain_shortcut.csv'

        backbone = getattr(self.config, 'backbone', 'densenet')

        self.train_dataset = CXP_dataset(
            self.config.data_dir,
            train_csv,
            augment=True,
            compute_sample_weights=False,
            split_name='train',
            backbone=backbone,
            use_cached_features=self.config.use_cached_features,
            features_dir=self.config.features_dir,
        )
        self.val_dataset = CXP_dataset(
            self.config.data_dir,
            val_csv,
            augment=False,
            compute_sample_weights=True,
            split_name='val',
            backbone=backbone,
            use_cached_features=self.config.use_cached_features,
            features_dir=self.config.features_dir,
        )

        self.test_dataset_aligned = CXP_dataset(
            self.config.data_dir,
            self.config.csv_dir / 'test_drain_shortcut_aligned.csv',
            augment=False,
            split_name='test_aligned',
            backbone=backbone,
            use_cached_features=self.config.use_cached_features,
            features_dir=self.config.features_dir,
        )
        self.test_dataset_misaligned = CXP_dataset(
            self.config.data_dir,
            self.config.csv_dir / 'test_drain_shortcut_misaligned.csv',
            augment=False,
            split_name='test_misaligned',
            backbone=backbone,
            use_cached_features=self.config.use_cached_features,
            features_dir=self.config.features_dir,
        )

        if self.debug:
            def fast_subset(ds, n=50):
                return torch.utils.data.Subset(ds, range(min(len(ds), n)))
            
            self.train_dataset = fast_subset(self.train_dataset)
            self.val_dataset = fast_subset(self.val_dataset, 20)
            self.test_dataset_aligned = fast_subset(self.test_dataset_aligned, 20)
            self.test_dataset_misaligned = fast_subset(self.test_dataset_misaligned, 20)
            
            self.config.balance_train = False # specific to debug mode logic in utils

        # --- SAMPLER LOGIC ---
        if self.config.balance_train and not self.jtt_error_indices:
            if self.train_dataset.drain.isna().any():
                raise ValueError(
                    "balance_train=True currently requires non-missing Drain labels, "
                    "but train split contains Drain=NaN rows."
                )
             # Basic balancing (not JTT stage 2)
            pneu_msk = self.train_dataset.labels == 1
            drain_counts_pneu = torch.bincount(torch.from_numpy(self.train_dataset.drain[pneu_msk].values))
            drain_weights_pneu = 1.0 / drain_counts_pneu.float()
            drain_counts_nopneu = torch.bincount(torch.from_numpy(self.train_dataset.drain[~pneu_msk].values))
            drain_weights_nopneu = 1.0 / drain_counts_nopneu.float()        
            
            sample_weights = torch.zeros_like(torch.from_numpy(self.train_dataset.labels.values), dtype=torch.float32)
            sample_weights[pneu_msk] = drain_weights_pneu[self.train_dataset.drain[pneu_msk].values]
            sample_weights[~pneu_msk] = drain_weights_nopneu[self.train_dataset.drain[~pneu_msk].values]

            self.train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            self.train_sampler = None

    def train_dataloader(self):
        # Handle JTT Stage 2
        if self.jtt_error_indices is not None:
             return get_jtt_loader(self.train_dataset, self.jtt_error_indices, self.config)
        
        prefetch_factor = 2 if self.config.num_workers > 0 else None
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=(self.train_sampler is None), 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            prefetch_factor=prefetch_factor,
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        prefetch_factor = 2 if self.config.num_workers > 0 else None
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            prefetch_factor=prefetch_factor
        )

    def test_dataloader(self):
        prefetch_factor = 2 if self.config.num_workers > 0 else None
        loader_aligned = DataLoader(
            self.test_dataset_aligned, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            prefetch_factor=prefetch_factor
        )
        loader_misaligned = DataLoader(
            self.test_dataset_misaligned, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            prefetch_factor=prefetch_factor
        )
        return [loader_aligned, loader_misaligned]

    def set_jtt_error_indices(self, error_indices):
        self.jtt_error_indices = error_indices
        # Reset sampler to None as JTT loader handles its own sampling
        self.train_sampler = None

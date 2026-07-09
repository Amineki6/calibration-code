from dataclasses import dataclass
from pathlib import Path
import torch
import argparse

@dataclass
class ExperimentConfig:
    """
    Central configuration object for the training pipeline.
    
    When using Optuna, the 'Hyperparameters' section will be populated 
    dynamically by the trial suggestions.
    """

    # -------------------------------------------------------------------------
    # 1. Method & Backbone Selection
    # -------------------------------------------------------------------------
    # Options: "standard", "supcon", "mmd", "score_matching"
    method_name: str = "standard"

    # Options: "densenet", "medsiglip", "medimageinsight"
    backbone: str = "densenet"

    # Options: True, False. To use pre-extracted foundation model features
    use_cached_features: bool = False
    
    select_chkpt_on: str = "fairness"  # Change default or add as option
    
    # -------------------------------------------------------------------------
    # 2. Hyperparameters
    # -------------------------------------------------------------------------
    # General Optimizer Params (not tuned)
    lr: float = 0.0001
    weight_decay: float = 0.005
    
    # EMA (Exponential Moving Average) Params (not tuned)
    ema_decay: float = 0.9
    
    # Method-Specific Params: Supervised Contrastive Learning
    # (Only used if method_name == "supcon")
    supcon_lambda: float = 0.50
    supcon_temperature: float = 0.10

    mmd_lambda: float = 1.0

    # Method-Specific Params: CDAN+E
    # (Only used if method_name == "cdan")
    cdan_lambda: float = 1.0
    cdan_entropy: bool = True

    # Method-Specific Params: Score Matching
    # (Only used if method_name == "score_matching")
    score_matching_lambda: float = 10.0
    score_matching_min_subgroup_count: int = 1
    dataset_score_matching_lambda: float = 10.0
    dataset_score_matching_min_subgroup_count: int = 10

    # Method-Specific Params: Soft Equalized Odds
    # (Only used if method_name == "soft_equalized_odds")
    soft_eo_lambda: float = 1.0
    soft_eo_mode: str = "both"  # one of: tpr, fpr, both
    soft_eo_tau: float = 0.5
    soft_eo_min_subgroup_count: int = 1
    soft_eo_temp_start: float = 0.0
    soft_eo_temp_end: float = 25.0
    soft_eo_temp_schedule_epochs: int = 25

    # Method-Specific Params: JTT
    # (Only used if method_name == "jtt")
    jtt_duration: int = 1   # Number of epochs for identification phase
    jtt_lambda: float = 4.0 # Upweighting factor for error set

    # -------------------------------------------------------------------------
    # 3. Data & Checkpointing
    # -------------------------------------------------------------------------
    # Data Balancing
    balance_train: bool = False  # activates weighted dataloader
    balance_val: bool = False  # selects whether _v2.csv files are used (yes if True, no if False)
    fairness_power: float = -1.0
    
    # Checkpoint Selection Metric
    # Options: "bce", "auroc"
    select_chkpt_on: str = "bce"

    # -------------------------------------------------------------------------
    # 4. Training Loop Constants
    # -------------------------------------------------------------------------
    epochs: int = 150
    batch_size: int = 64
    num_runs: int = 1  # Set to >1 for statistical significance testing (outside Optuna)
    num_workers: int = 12
    seed: int = 42

    # -------------------------------------------------------------------------
    # 5. System / Paths (Usually set via command line args)
    # -------------------------------------------------------------------------
    # These defaults can be overwritten by argparse in the main script
    data_dir: Path = Path('/data')
    csv_dir: Path = Path('.')
    out_dir: Path = Path('~/cxp_shortcut_out')
    features_dir: Path = Path('features_cache')
    
    comment: str = "No comment provided"
    
    @property
    def device(self):
        """Helper to automatically determine device (CUDA -> MPS -> CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps") # Uses Mac GPU!
        else:
            return torch.device("cpu")

def get_config(args: argparse.Namespace, trial=None) -> ExperimentConfig:
    """
    Constructs an ExperimentConfig object based on command-line arguments and Optuna trial.
    """
    config = ExperimentConfig()
    
    # Static paths from Args
    config.data_dir = args.data_dir
    config.csv_dir = args.csv_dir
    config.out_dir = args.out_dir
    config.balance_train = args.balance_train
    config.balance_val = args.balance_val
    config.fairness_power = args.fairness_power
    config.select_chkpt_on = args.select_chkpt_on
    config.method_name = args.method
    config.backbone = args.backbone
    if hasattr(args, 'use_cached_features'):
        config.use_cached_features = args.use_cached_features
    if hasattr(args, 'features_dir'):
        config.features_dir = args.features_dir
    
    
    if hasattr(args, 'comment'):
        config.comment = args.comment
   
    # --- HYPERPARAMETER & METHOD SELECTION ---
    if args.debug:
        config.epochs = 2
        config.batch_size = 4 
        config.num_workers = 0 
        
        if config.method_name == "supcon":
            config.supcon_lambda = 0.5
            config.supcon_temperature = 0.1
        elif config.method_name == "mmd":
            config.mmd_lambda = 1.0
        elif config.method_name == "cdan":
            config.cdan_lambda = 1.0
            config.cdan_entropy = True
        elif config.method_name == "score_matching":
            config.score_matching_lambda = 1.0
        elif config.method_name == "dataset_score_matching":
            config.dataset_score_matching_lambda = 1.0            
        elif config.method_name == "jtt":
            config.jtt_duration = 1
            config.jtt_lambda = 4.0
        elif config.method_name == "soft_equalized_odds":
            config.soft_eo_lambda = 1.0
    elif trial is not None:
        # Optuna Optimization
        if config.method_name == "supcon":
            config.supcon_lambda = trial.suggest_float("supcon_lambda", 100.0, 400.0, log=True)
            config.supcon_temperature = trial.suggest_float("supcon_temperature", 0.05, 0.5)
        elif config.method_name == "mmd":
            config.mmd_lambda = trial.suggest_float("mmd_lambda", 1e-4, 10.0, log=True)
        elif config.method_name == "cdan":
            config.cdan_lambda = trial.suggest_float("cdan_lambda", 1e-4, 100.0, log=True)
            config.cdan_entropy = True
        elif config.method_name == "score_matching":
            config.score_matching_lambda = trial.suggest_float("score_matching_lambda", 0.1, 1000.0, log=True)
        elif config.method_name == "dataset_score_matching":
            config.dataset_score_matching_lambda = trial.suggest_float("dataset_score_matching_lambda", 1e-1, 1e4, log=True)
        elif config.method_name == "jtt":
            config.jtt_duration = trial.suggest_int("jtt_duration", 1, max(1, config.epochs // 2))
            config.jtt_lambda = trial.suggest_int("jtt_lambda", 2, 100, log=True)
        elif config.method_name == "soft_equalized_odds":
            config.soft_eo_lambda = trial.suggest_float("soft_eo_lambda", 1e-3, 1e2, log=True)
            
    # CLI --epochs override (takes precedence over default and debug)
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.batch_size = args.batch_size

    # CLI Hyperparameter Overrides (takes absolute precedence)
    if hasattr(args, 'supcon_lambda') and args.supcon_lambda is not None:
        config.supcon_lambda = args.supcon_lambda
    if hasattr(args, 'supcon_temperature') and args.supcon_temperature is not None:
        config.supcon_temperature = args.supcon_temperature
    if hasattr(args, 'mmd_lambda') and args.mmd_lambda is not None:
        config.mmd_lambda = args.mmd_lambda
    if hasattr(args, 'cdan_lambda') and args.cdan_lambda is not None:
        config.cdan_lambda = args.cdan_lambda
    if hasattr(args, 'cdan_entropy') and args.cdan_entropy is not None:
        config.cdan_entropy = args.cdan_entropy
    if hasattr(args, 'jtt_lambda') and args.jtt_lambda is not None:
        config.jtt_lambda = args.jtt_lambda
    if hasattr(args, 'jtt_duration') and args.jtt_duration is not None:
        config.jtt_duration = args.jtt_duration
    if hasattr(args, 'soft_eo_lambda') and args.soft_eo_lambda is not None:
        config.soft_eo_lambda = args.soft_eo_lambda
    if hasattr(args, 'soft_eo_mode') and args.soft_eo_mode is not None:
        config.soft_eo_mode = args.soft_eo_mode
    if hasattr(args, 'soft_eo_tau') and args.soft_eo_tau is not None:
        config.soft_eo_tau = args.soft_eo_tau
    if hasattr(args, 'soft_eo_min_subgroup_count') and args.soft_eo_min_subgroup_count is not None:
        config.soft_eo_min_subgroup_count = args.soft_eo_min_subgroup_count
    if hasattr(args, 'soft_eo_temp_start') and args.soft_eo_temp_start is not None:
        config.soft_eo_temp_start = args.soft_eo_temp_start
    if hasattr(args, 'soft_eo_temp_end') and args.soft_eo_temp_end is not None:
        config.soft_eo_temp_end = args.soft_eo_temp_end
    if hasattr(args, 'soft_eo_temp_schedule_epochs') and args.soft_eo_temp_schedule_epochs is not None:
        config.soft_eo_temp_schedule_epochs = args.soft_eo_temp_schedule_epochs

    return config
import os
import sys
import logging
import argparse
from pathlib import Path
import json
import coolname
import datetime

import torch

# W&B Cluster Timeouts
os.environ["WANDB_HTTP_TIMEOUT"] = "120"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ["WANDB__SERVICE_WAIT"] = "300"

import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Local imports
from config import ExperimentConfig, get_config
from model import CXP_Model
import methods
from utils import run_jtt_stage1, TeeStream

from lightning_module import CXPLightningModule
from lightning_datamodule import CXPDataModule

# Global args placeholder
GLOBAL_ARGS: argparse.Namespace = argparse.Namespace()

# PyTorch Optimization Settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# Register safe globals for checkpoint loading.
# PyTorch 2.6 defaults to weights_only=True and may block pathlib classes.
import pathlib
torch.serialization.add_safe_globals([
    ExperimentConfig,
    pathlib.PosixPath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.WindowsPath,
])


def _infer_default_out_dir(data_dir: Path) -> Path:
    """Infer the run output root from the dataset root."""
    return data_dir / "runs"

def setup_logging(root_dir):
    log_path = root_dir / "optuna_training.log"
    
    # Create or append to the log file
    log_file = open(log_path, 'a', encoding='utf-8')
    
    # Overwrite stdout and stderr
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    
    # Use only StreamHandler, since Tee will capture its stderr output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ],
        force=True
    )

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler


def _stage_banner(title: str) -> None:
    logging.info("=" * 72)
    logging.info(title)
    logging.info("=" * 72)


def _get_method_lambda_items(config: ExperimentConfig) -> list[tuple[str, float | int | str]]:
    if config.method_name == "supcon":
        return [
            ("supcon_lambda", config.supcon_lambda),
            ("supcon_temperature", config.supcon_temperature),
        ]
    if config.method_name == "mmd":
        return [("mmd_lambda", config.mmd_lambda)]
    if config.method_name == "cdan":
        return [
            ("cdan_lambda", config.cdan_lambda),
            ("cdan_entropy", int(config.cdan_entropy)),
        ]
    if config.method_name == "score_matching":
        return [
            ("score_matching_lambda", config.score_matching_lambda),
            ("score_matching_min_subgroup_count", config.score_matching_min_subgroup_count),
        ]
    if config.method_name == "dataset_score_matching":
        return [
            ("dataset_score_matching_lambda", config.dataset_score_matching_lambda),
            (
                "dataset_score_matching_min_subgroup_count",
                config.dataset_score_matching_min_subgroup_count,
            ),
        ]
    if config.method_name == "jtt":
        return [
            ("jtt_lambda", config.jtt_lambda),
            ("jtt_duration", config.jtt_duration),
        ]
    if config.method_name == "soft_equalized_odds":
        return [
            ("soft_eo_lambda", config.soft_eo_lambda),
            ("soft_eo_mode", config.soft_eo_mode),
            ("soft_eo_tau", config.soft_eo_tau),
            ("soft_eo_min_subgroup_count", config.soft_eo_min_subgroup_count),
            ("soft_eo_temp_start", config.soft_eo_temp_start),
            ("soft_eo_temp_end", config.soft_eo_temp_end),
            ("soft_eo_temp_schedule_epochs", config.soft_eo_temp_schedule_epochs),
        ]
    return []

def run_lightning_training(config, trial_number, study_root, trial=None, wandb_group=None, run_name=None, run_phase: str = "unspecified"):
    """
    Helper function to run the Lightning training loop. 
    Used for both Optimization Objective and Final Evaluation.
    """
    
    logging.info(
        "[PHASE=%s] Starting run: trial_number=%s, run_name=%s, method=%s, monitor=%s",
        run_phase,
        trial_number,
        run_name if run_name is not None else f"trial_{trial_number}",
        config.method_name,
        config.select_chkpt_on,
    )
    lambda_items = _get_method_lambda_items(config)
    if lambda_items:
        logging.info("[PHASE=%s] Method coefficients: %s", run_phase, ", ".join([f"{k}={v}" for k, v in lambda_items]))

    # --- SETUP DATA ---
    datamodule = CXPDataModule(config, debug=GLOBAL_ARGS.debug)
    
    # --- METHOD ---
    method = methods.get_method(config.method_name, config)
    
    # --- JTT STAGE 1 ---
    if config.method_name == "jtt":
        run_jtt_stage1(
            config=config,
            method=method,
            datamodule=datamodule,
            study_root=study_root,
            trial_number=trial_number,
            run_name=run_name,
            wandb_group=wandb_group,
            debug=GLOBAL_ARGS.debug
        )

    # --- MAIN TRAINING ---
    
    # Init Model
    model = CXP_Model(method, backbone=config.backbone, use_cached_features=config.use_cached_features)
    pl_module = CXPLightningModule(model, method, config)
    
    # Logger
    final_run_name = run_name if run_name else f"{GLOBAL_ARGS.study_name}_trial_{trial_number}"
    wandb_logger = WandbLogger(
        project="cxr_optuna_study", 
        group=wandb_group, 
        name=final_run_name,
        config=config.__dict__,
        save_dir=str(study_root),
        id=f"{final_run_name}_{coolname.generate_slug(2)}"
    )
    
    # Checkpointing Logic
    monitor_metric = ""
    mode = ""
    
    if config.select_chkpt_on.upper() == "AUROC":
        monitor_metric = "val/auroc"
        mode = "max"
    elif config.select_chkpt_on.upper() == "WAUROC":
        monitor_metric = "val/wauroc"
        mode = "max"
    elif config.select_chkpt_on.upper() == "FAIRNESS":
        monitor_metric = "val/fairness_score"
        mode = "max"
    elif config.select_chkpt_on.upper() == "BCE":
        monitor_metric = "val/bce"
        mode = "min"
    elif config.select_chkpt_on.upper() == "WBCE":
        monitor_metric = "val/wbce"
        mode = "min"
    elif config.select_chkpt_on.upper() == "WORST_GROUP":
        monitor_metric = "val/worst_group_accuracy"
        mode = "max"
    
    checkpoint_filename = f"trial_{trial_number}_best"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=study_root / "checkpoints",
        filename=checkpoint_filename,
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1,
        save_last=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        precision="16-mixed",
    )
    
    logging.info("[PHASE=%s] Entering train/validation loop.", run_phase)

    # Fit
    trainer.fit(pl_module, datamodule=datamodule)
    
    # Get Best Metric Value
    best_score = checkpoint_callback.best_model_score
    if best_score is not None:
        best_score = best_score.item()
    else:
        best_score = 0.0 if mode == "max" else float('inf')

    # --- TESTING ---
    # Only run testing if this is a Final Eval run (trial is None or indicated)
    # Actually, for Optuna we usually don't run test set. 
    # But if `run_name` is provided (final eval), we do.
    test_results = None
    if run_name is not None:
        logging.info("[PHASE=%s] Entering test loop (best-checkpoint evaluation).", run_phase)
        # Load Best Model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            test_results = trainer.test(pl_module, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            logging.warning("[PHASE=%s] No best checkpoint found; testing current in-memory weights.", run_phase)
            test_results = trainer.test(pl_module, datamodule=datamodule)

    wandb_logger.experiment.finish()
    
    # Cleanup
    del trainer, pl_module, model, datamodule
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("[PHASE=%s] Run complete. best_score=%s", run_phase, best_score)
    return best_score, test_results

def run_final_evaluation_runs(args, best_params):
    final_config = get_config(args)
    
    # Apply Best Params
    for k, v in best_params:
        if hasattr(final_config, k):
            setattr(final_config, k, v)
            
    if args.debug:
        final_config.epochs = 2
        final_config.batch_size = 4
        final_config.num_workers = 0

    eval_root = GLOBAL_ARGS.study_root / "final_evaluation"
    eval_root.mkdir(exist_ok=True)

    lambda_items = _get_method_lambda_items(final_config)
    if lambda_items:
        logging.info(
            "Final-eval method coefficients: %s",
            ", ".join([f"{k}={v}" for k, v in lambda_items]),
        )

    import numpy as np
    from collections import defaultdict
    
    all_test_metrics = defaultdict(list)

    for i in range(args.n_eval_runs):
        run_name = f"{args.study_name}_final_run_{i}"
        _, test_results = run_lightning_training(
            config=final_config,
            trial_number=i,
            study_root=eval_root,
            wandb_group=f"{args.study_name}_final",
            run_name=run_name,
            run_phase=f"final_eval_run_{i}",
        )
        
        if test_results is not None and len(test_results) > 0:
            # test_results is a list of dicts (one for each dataloader, usually combined into one dict by lightning)
            res_dict = test_results[0]
            for k, v in res_dict.items():
                if k.startswith("test/") or k.startswith("test_"):
                    all_test_metrics[k].append(v)
                    
    if all_test_metrics:
        logging.info("===== FINAL RESULTS ACROSS RUNS =====")
        if lambda_items:
            for k, v in lambda_items:
                logging.info("estimated/%s: %s", k, v)
        lambda_from_best_params = [(k, v) for (k, v) in best_params if "lambda" in k]
        if lambda_from_best_params:
            for k, v in lambda_from_best_params:
                logging.info("estimated/%s (from best_params): %s", k, v)
        for k, v_list in all_test_metrics.items():
            mean_val = np.mean(v_list)
            std_val = np.std(v_list)
            logging.info(f"{k}: {mean_val:.4f} +/- {std_val:.4f}")
        logging.info("=====================================")

def objective(trial):
    study_root = GLOBAL_ARGS.study_root
    config = get_config(GLOBAL_ARGS, trial)

    best_score, _ = run_lightning_training(
        config=config, 
        trial_number=trial.number, 
        study_root=study_root,
        trial=trial,
        wandb_group=GLOBAL_ARGS.study_name,
        run_phase=f"optuna_trial_{trial.number}",
    )

    chkpt_path = study_root / "checkpoints" / f"trial_{trial.number}_best.ckpt"
    if chkpt_path.exists():
         chkpt_path.unlink()

    return best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path Arguments
    parser.add_argument('--data_dir', type=Path, default=Path('data'), help='Directory above /CheXpert-v1.0-small')
    parser.add_argument('--csv_dir', type=Path, default=Path('csv_data'), help='Directory containing CSV files')
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=None,
        help='Output directory for logs/checkpoints. If omitted, defaults to <data_dir>/runs.',
    )
    parser.add_argument('--features_dir', type=Path, default=Path('features_cache'), help='Directory containing cached foundation model features')
    
    # Study Arguments
    parser.add_argument('--study_name', type=str, default=None, help='Name of the study. If not provided, generates a random name.')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials to run')
    parser.add_argument('--n_eval_runs', type=int, default=0, help='Number of final evaluation runs using best params (default: 0)')
    parser.add_argument('--balance_train', type=lambda x: x.lower() == 'true', default=False, help='Use weighted sampler for training')
    parser.add_argument('--balance_val', type=lambda x: x.lower() == 'true', default=False, help='Use balanced validation set')
    parser.add_argument('--fairness_power', type=float, default=-1.0, help='Power-mean exponent for fairness aggregation across ordered group pairs; use 0 for geometric mean and -1 for harmonic mean (default).')
    parser.add_argument('--select_chkpt_on', type=str, default="fairness", choices=["bce", "wbce", "auroc", "wauroc", "fairness", "worst_group"], help='Metric to select best model')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (tiny data, 1 epoch, CPU/MPS friendly)')
    
    # Methods
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'supcon', 'mmd', 'cdan', 'score_matching', 'dataset_score_matching', 'jtt', 'soft_equalized_odds'], help='Method to use for training (default: standard)')

    # Backbone
    parser.add_argument('--backbone', type=str, default='densenet', choices=['densenet', 'medsiglip', 'medimageinsight'], help='Feature extractor backbone. Foundation model backbones are frozen; only the clf head is trained.')
    parser.add_argument('--use_cached_features', action='store_true', help='If True, bypass frozen backbones and load features directly from --features_dir. Only applicable for foundational models.')

    # Training
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config default of 150)')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size (overrides config default of 64)')

    # Method-Specific Hyperparameters (Manual Overrides)
    parser.add_argument('--supcon_lambda', type=float, default=None)
    parser.add_argument('--supcon_temperature', type=float, default=None)
    parser.add_argument('--mmd_lambda', type=float, default=None)
    parser.add_argument('--cdan_lambda', type=float, default=None)
    parser.add_argument('--cdan_entropy', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--jtt_lambda', type=float, default=None)
    parser.add_argument('--jtt_duration', type=int, default=None)
    parser.add_argument('--soft_eo_lambda', type=float, default=None)
    parser.add_argument('--soft_eo_mode', type=str, choices=['tpr', 'fpr', 'both'], default=None)
    parser.add_argument('--soft_eo_tau', type=float, default=None)
    parser.add_argument('--soft_eo_min_subgroup_count', type=int, default=None)
    parser.add_argument('--soft_eo_temp_start', type=float, default=None)
    parser.add_argument('--soft_eo_temp_end', type=float, default=None)
    parser.add_argument('--soft_eo_temp_schedule_epochs', type=int, default=None)

    # Run Comment
    parser.add_argument('--comment', type=str, default="No comment provided", help='Comment for the run and what it represents')

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = _infer_default_out_dir(args.data_dir)

    # Automatically override select_chkpt_on for JTT
    #if args.method == "jtt" :
        #logging.info("Auto-overriding select_chkpt_on to 'worst_group' because method is 'jtt'.")
        #args.select_chkpt_on = "worst_group"

    if args.study_name is None:
        args.study_name = coolname.generate_slug(2) + '_' + args.backbone + '_' + args.method 
    
    study_root = args.out_dir / args.study_name
    study_root.mkdir(parents=True, exist_ok=True)
    
    GLOBAL_ARGS = args
    GLOBAL_ARGS.study_root = study_root

    (study_root / "checkpoints").mkdir(exist_ok=True)
    (study_root / "predictions").mkdir(exist_ok=True)

    setup_logging(study_root)
    _stage_banner(f"RUN START: study={args.study_name}, method={args.method}, n_trials={args.n_trials}, n_eval_runs={args.n_eval_runs}")

    # Optuna Setup
    if args.n_trials == 0:
        optimize = False
        logging.info("Skipping optimization stage: n_trials == 0")
    elif args.method == 'standard':
        optimize = False
        logging.info("Skipping optimization stage: method == 'standard'")
    else:
        optimize = True

    best_params = []
    
    if optimize:
        _stage_banner("STAGE 1/3: HYPERPARAMETER OPTIMIZATION (OPTUNA)")
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        config_dict = vars(args)
        config_dict["start_time"] = start_time_str
        
        with open(study_root / "experiment_config.json", "w") as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        logging.info(f"Starting Study: {args.study_name}")
        logging.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        if 'AUROC' in args.select_chkpt_on.upper() or args.select_chkpt_on.upper() in ["FAIRNESS", "WORST_GROUP"]:
            direction = "maximize"
        else:
            direction = "minimize"
        
        study = optuna.create_study(
            sampler=optuna.samplers.GPSampler(),
            direction=direction, 
            study_name=args.study_name,
            load_if_exists=True
        )
        
        # ========================== RUN OPTIMIZATION ===============================================

        from wandb.errors import CommError
        study.optimize(
            objective, 
            n_trials=args.n_trials, 
            gc_after_trial=True,
            catch=(CommError, TimeoutError, ConnectionError)
        )
        best_params = list(study.best_trial.params.items())
        
        logging.info("===== STUDY COMPLETED =====")
        logging.info(f"Best Trial Number: {study.best_trial.number}")
        logging.info(f"Best Value ({args.select_chkpt_on}): {study.best_trial.value}")
        logging.info("Best Params:")
        for k, v in best_params:
            logging.info(f"  {k}: {v}")
    
    else:
        _stage_banner("STAGE 1/3: INITIAL TRAIN+EVAL (NO OPTIMIZATION)")
        logging.info("Optimization disabled for this run configuration.")
        
        if args.n_eval_runs > 0:
            logging.info("Skipping initial run because n_eval_runs > 0. We will proceed directly to final evaluation runs.")
        else:
            # Prepare Config
            config = get_config(args)

            # Run Single Training
            best_score, _ = run_lightning_training(
                config=config,
                trial_number=0,
                study_root=study_root,
                wandb_group=args.study_name,
                run_name=f"{args.study_name}_initial",
                run_phase="initial_eval",
            )
            logging.info(f"Run Completed. Best Score: {best_score}")

    # ========================== FINAL EVALUATION RUNS ==========================================
    _stage_banner("STAGE 2/3: TRANSITION TO FINAL EVALUATION")
    if args.n_eval_runs > 0:
        _stage_banner(f"STAGE 3/3: FINAL EVALUATION RUNS ({args.n_eval_runs} seeds)")
        run_final_evaluation_runs(args, best_params)
    else:
        _stage_banner("STAGE 3/3: FINAL EVALUATION RUNS (SKIPPED)")
        logging.info("No final evaluation runs requested (n_eval_runs == 0).")

    _stage_banner("RUN COMPLETE")
# Prevalence-aware calibration is all you need for shortcut learning mitigation

<p align="center">
  <img src="images/thumbnail.jpg" alt="Thumbnail">
</p>

This repository contains the codebase for research on **Shortcut Learning Mitigation**, specifically tailored for medical imaging (e.g., Chest X-Rays). The framework is designed to train, evaluate, and tune models robust to shortcut features (spurious correlations).

It includes integrations for automated hyperparameter tuning via [Optuna](https://optuna.org/) and experiment tracking via [Weights & Biases (W&B)](https://wandb.ai/).


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Amineki6/calibration-code.git
cd calibration-code
```

2. Create a virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

*Note: Ensure you have a compatible version of PyTorch installed for your hardware (CUDA/MPS).*

## Project Structure

- `train.py`: The main entry point for running training, Optuna sweeps, and final evaluations.
- `config.py`: Centralized configuration dataclasses handling hyperparameters and file paths.
- `dataset.py` / `lightning_datamodule.py`: Handling of data loading, caching, and batching.
- `model.py` / `lightning_module.py`: PyTorch Lightning model definitions and backbone integrations.
- `methods/`: Directory containing all shortcut mitigation algorithms (e.g., `supcon.py`, `mmd.py`, `jtt.py`).
- `eval_*.py`: Standalone scripts for evaluating reliability and recalibrating runs.
- `extract_features.py`: Script to extract and cache foundation model features.

## Implemented Methods

You can select a shortcut mitigation algorithm via the `--method` argument. Supported methods include:
- `standard`: Standard Empirical Risk Minimization (ERM)
- `supcon`: Supervised Contrastive Learning
- `mmd`: Maximum Mean Discrepancy
- `cdan`: Conditional Domain Adversarial Networks (CDAN+E)
- `score_matching` & `dataset_score_matching`: Score Matching objectives
- `soft_equalized_odds`: Soft Equalized Odds penalty
- `jtt`: Just Train Twice

## Usage

### Basic Training

To run a standard baseline with no hyperparameter optimization (using default config parameters):

```bash
python train.py \
    --method standard \
    --backbone densenet \
    --data_dir /path/to/data \
    --csv_dir /path/to/csv \
    --epochs 50 \
    --batch_size 64
```

### Hyperparameter Optimization

The framework is highly optimized for running Optuna trials to find the best mitigation hyperparameters. You can specify the number of trials and the evaluation metric for model selection. 

```bash
python train.py \
    --method supcon \
    --backbone medsiglip \
    --n_trials 20 \
    --select_chkpt_on fairness \
    --use_cached_features
```
*Tip: When using foundation models, pass `--use_cached_features` to load pre-extracted features instead of running images through the heavy backbone.*

## Configuration

Default hyperparameters and training parameters are managed in `config.py` using `ExperimentConfig`. You can override most of these from the command line.

For example, to manually override method-specific hyperparams instead of using Optuna:
```bash
python train.py --method mmd --mmd_lambda 2.5 --epochs 100
```

## Logging and Tracking

The project natively integrates with **Weights & Biases (W&B)**. 
- Ensure you have run `wandb login` before starting a distributed sweep.
- A comprehensive `optuna_training.log` is also generated automatically inside the output directory.

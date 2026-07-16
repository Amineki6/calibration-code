import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shortcut Learning Mitigation Training"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the YAML configuration file",
    )

    # Study Arguments that dictate high-level script behavior (not model hyperparameters)
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Name of the study. If not provided, generates a random name.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--n_eval_runs",
        type=int,
        default=0,
        help="Number of final evaluation runs using best params (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with config/debug.yaml (tiny data, 1 epoch)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Allow dynamic omegaconf overrides
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options using omegaconf dotlist style (e.g. epochs=50 method=jtt)",
    )

    return parser.parse_args()

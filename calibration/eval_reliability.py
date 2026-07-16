import argparse
import logging
import pathlib
import sys
from pathlib import Path

# Add project root to sys.path so we can import config, model, etc.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

import methods
from config import ExperimentConfig
from dataset import CXP_dataset
from lightning_module import CXPLightningModule
from meval.diags import rel_diag
from model import CXP_Model


DEFAULT_CHECKPOINT = Path(
    r"\\gaia\imageData\deep_learning\output\petersen\cxr_small_data\unbiased-moose_densenet_standard\final_evaluation\checkpoints\trial_0_best.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone checkpoint evaluation and reliability plotting with meval."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing images referenced by test CSV Path columns.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("csv_data"),
        help="Directory containing test_drain_shortcut_aligned.csv and test_drain_shortcut_misaligned.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "standalone_eval",
        help="Directory for merged predictions and reliability plots.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


def _windows_safe_torch_load(checkpoint_path: Path) -> dict[str, Any]:
    import os

    torch.serialization.add_safe_globals(
        [
            ExperimentConfig,
            pathlib.PosixPath,
            pathlib.PurePosixPath,
            pathlib.PureWindowsPath,
            pathlib.WindowsPath,
        ]
    )

    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path
        pathlib.WindowsPath = original_windows_path

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
    return checkpoint


def _resolve_config(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> ExperimentConfig:
    config = ExperimentConfig()
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    ckpt_config = (
        hyper_parameters.get("config") if isinstance(hyper_parameters, dict) else None
    )

    if isinstance(ckpt_config, ExperimentConfig):
        config = ckpt_config

    config.data_dir = args.data_dir
    config.csv_dir = args.csv_dir
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    return config


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_test_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    aligned_csv = config.csv_dir / "test_drain_shortcut_aligned.csv"
    misaligned_csv = config.csv_dir / "test_drain_shortcut_misaligned.csv"

    if not aligned_csv.exists() or not misaligned_csv.exists():
        raise FileNotFoundError(
            "Expected test CSV files missing in csv_dir: "
            f"{aligned_csv} and/or {misaligned_csv}"
        )

    common_kwargs = {
        "augment": False,
        "backbone": config.backbone,
        "use_cached_features": config.use_cached_features,
        "features_dir": config.features_dir,
    }

    ds_aligned = CXP_dataset(
        config.data_dir,
        aligned_csv,
        split_name="test_aligned",
        compute_sample_weights=False,
        **common_kwargs,
    )
    ds_misaligned = CXP_dataset(
        config.data_dir,
        misaligned_csv,
        split_name="test_misaligned",
        compute_sample_weights=False,
        **common_kwargs,
    )

    prefetch_factor = 2 if config.num_workers > 0 else None
    loader_aligned = DataLoader(
        ds_aligned,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    loader_misaligned = DataLoader(
        ds_misaligned,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    return loader_aligned, loader_misaligned


@torch.inference_mode()
def _predict(
    module: CXPLightningModule,
    dataloader: DataLoader,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    enable_autocast = device.type == "cuda"
    for batch in dataloader:
        with torch.cuda.amp.autocast(enabled=enable_autocast):
            logits_tensor, _ = module.ema_model(
                batch.inputs.to(device, non_blocking=True)
            )
        logits_fp32 = logits_tensor.reshape(-1).float()
        probs = torch.sigmoid(logits_fp32).cpu()
        logits_cpu = logits_fp32.cpu()

        labels = batch.labels.cpu()
        drains = batch.drains.cpu()

        for logit_val, prob, label, drain in zip(logits_cpu, probs, labels, drains):
            rows.append(
                {
                    "split": split_name,
                    "logit": logit_val.item(),
                    "y_prob": prob.item(),
                    "label": int(label),
                    "drain": float(drain),
                }
            )

    return pd.DataFrame(rows)


def _run_reliability_plot(merged_df: pd.DataFrame, output_dir: Path) -> None:
    finite_drain = pd.to_numeric(merged_df["drain"], errors="coerce")
    plot_df = merged_df[finite_drain.isin([0.0, 1.0])].copy()
    if plot_df.empty:
        raise RuntimeError(
            "No samples with drain in {0,1}; cannot make drain-group reliability diagram"
        )

    plot_df["drain"] = plot_df["drain"].astype(int)
    plot_df["label"] = plot_df["label"].astype(bool)

    prevalence = float(plot_df["label"].mean())
    fig, _, _ = rel_diag(
        plot_df,
        plot_groups=["drain=0", "drain=1"],
        fig_title="Reliability by drain group (merged aligned+misaligned test sets)",
        legend=True,
        add_risk_density=True,
        threshold=prevalence,
    )

    html_path = output_dir / "reliability_drain_groups.html"
    fig.write_html(html_path)
    logging.info("Saved reliability diagram HTML to %s", html_path)

    png_path = output_dir / "reliability_drain_groups.png"
    try:
        fig.write_image(png_path)
        logging.info("Saved reliability diagram PNG to %s", png_path)
    except Exception as error:
        logging.warning("PNG export skipped (%s). HTML export succeeded.", error)


def main() -> None:
    # Set identical PyTorch optimizations as train.py to ensure precision parity
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    checkpoint = _windows_safe_torch_load(args.checkpoint)
    config = _resolve_config(checkpoint, args)

    logging.info(
        "Checkpoint loaded. method=%s backbone=%s",
        config.method,
        config.backbone,
    )

    method = methods.get_method(config.method, config)
    model = CXP_Model(
        method_strategy=method,
        backbone=config.backbone,
        use_cached_features=config.use_cached_features,
    )
    module = CXPLightningModule(model=model, method=method, config=config)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(
            "Missing state_dict keys (%d), sample=%s", len(missing), missing[:5]
        )
    if unexpected:
        logging.warning(
            "Unexpected state_dict keys (%d), sample=%s",
            len(unexpected),
            unexpected[:5],
        )

    device = _resolve_device(args.device)
    module.to(device)
    module.eval()
    module.ema_model.eval()
    logging.info("Running inference on device=%s", device)

    loader_aligned, loader_misaligned = _build_test_loaders(config)
    df_aligned = _predict(module, loader_aligned, split_name="aligned", device=device)
    df_misaligned = _predict(
        module, loader_misaligned, split_name="misaligned", device=device
    )
    merged_df = pd.concat([df_aligned, df_misaligned], ignore_index=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / "merged_test_predictions.csv"
    merged_df.to_csv(merged_path, index=False)
    logging.info("Saved merged predictions to %s (n=%d)", merged_path, len(merged_df))
    logging.info(
        "Merged split counts: aligned=%d misaligned=%d",
        len(df_aligned),
        len(df_misaligned),
    )

    _run_reliability_plot(merged_df, output_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()

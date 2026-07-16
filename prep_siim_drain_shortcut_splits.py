import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_pool(siim_root: Path, jpg_root: Path) -> pd.DataFrame:
    """Build train pool from train-rle (PTX known), with drain labels where available."""
    train_rle_path = siim_root / "train-rle.csv"
    if not train_rle_path.exists():
        train_rle_path = jpg_root / "train-rle.csv"

    tube_dict_path = siim_root / "cxr_tube_dict.pkl"
    if not tube_dict_path.exists():
        tube_dict_path = jpg_root / "cxr_tube_dict.pkl"

    # Read with ImageId as index, mirroring notebook logic.
    rle = pd.read_csv(train_rle_path, index_col=0)
    tube_dict = pickle.load(open(tube_dict_path, "rb"))

    encoded_col = (
        "EncodedPixels" if "EncodedPixels" in rle.columns else " EncodedPixels"
    )

    # Whitespace-safe PTX parsing and collapse to one row per ImageId.
    # Any non -1 mask for an ImageId marks PTX=1.
    ptx = (rle[encoded_col].astype(str).str.strip() != "-1").astype(int)
    ptx_by_image = ptx.groupby(level=0).max()

    base = ptx_by_image.rename("Pneumothorax").to_frame().reset_index()
    base = base.rename(columns={base.columns[0]: "ImageId"})

    df = base.copy()
    df["Drain"] = df["ImageId"].map(tube_dict)
    # Keep drain as nullable float for compatibility with NaNs in unlabeled rows.
    df["Drain"] = pd.to_numeric(df["Drain"], errors="coerce").astype(float)
    df["Path"] = df["ImageId"].map(lambda x: f"train/{x}.jpg")

    # Keep only rows where converted JPG exists.
    exists_mask = df["Path"].map(lambda p: (jpg_root / str(p)).exists())
    missing = int((~exists_mask).sum())
    if missing > 0:
        print(f"Warning: dropping {missing} rows without JPG files under {jpg_root}.")
    df = df[exists_mask].copy()

    df["strat_col"] = df.apply(
        lambda r: (
            "unlabeled"
            if pd.isna(r["Drain"])
            else f"{int(r['Pneumothorax'])}_{int(r['Drain'])}"
        ),
        axis=1,
    )
    return df[["Path", "Pneumothorax", "Drain", "strat_col"]]


def load_test_unlabeled_pool(jpg_root: Path) -> pd.DataFrame:
    """Build original SIIM test pool (PTX unknown, no drain labels)."""
    test_dir = jpg_root / "test"
    test_files = sorted(test_dir.glob("*.jpg"))
    return pd.DataFrame(
        {
            "Path": [f"test/{p.name}" for p in test_files],
            "Pneumothorax": [-1] * len(test_files),
            "Drain": [np.nan] * len(test_files),
            "strat_col": ["test_unknown_ptx"] * len(test_files),
        }
    )


def sample_group(
    df: pd.DataFrame, ptx: int, drain: int, n: int, seed: int
) -> pd.DataFrame:
    grp = df[(df["Pneumothorax"] == ptx) & (df["Drain"] == drain)]
    if len(grp) < n:
        raise ValueError(
            f"Not enough samples for group (Pneumothorax={ptx}, Drain={drain}). "
            f"Need {n}, found {len(grp)}"
        )
    return grp.sample(n=n, random_state=seed)


def print_counts(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: n={len(df)}")
    print(df["strat_col"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SIIM drain-shortcut splits for cxr-shortcut."
    )
    parser.add_argument(
        "--siim_root",
        type=Path,
        default=Path("../data/siim-acr-pneumothorax"),
        help="Path containing train-rle.csv and cxr_tube_dict.pkl",
    )
    parser.add_argument(
        "--jpg_root",
        type=Path,
        default=Path("../data/siim-acr-pneumothorax/jpg-images"),
        help="Root of converted JPG data with train/<ImageId>.jpg",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("csv_data"),
        help="Output folder for CXP-compatible CSVs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_per_group", type=int, default=25)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_pool = load_train_pool(args.siim_root, args.jpg_root)
    test_unlabeled_pool = load_test_unlabeled_pool(args.jpg_root)
    pooled_all = pd.concat([train_pool, test_unlabeled_pool], ignore_index=True)

    labeled_pool = train_pool[train_pool["Drain"].notna()].copy()
    unlabeled_pool = train_pool[train_pool["Drain"].isna()].copy()

    print_counts("Train pool (from train-rle)", train_pool)
    print_counts("Original test pool (PTX unknown)", test_unlabeled_pool)
    print_counts("Pooled all available JPGs (train+test)", pooled_all)
    print_counts("Labeled subset", labeled_pool)
    print_counts("Unlabeled subset", unlabeled_pool)

    # Total test set: 25 per drain x ptx cell => 100 by default.
    # aligned = (ptx=1,drain=1) + (ptx=0,drain=0)
    # misaligned = (ptx=1,drain=0) + (ptx=0,drain=1)
    n = args.test_per_group
    test_aligned = pd.concat(
        [
            sample_group(labeled_pool, ptx=1, drain=1, n=n, seed=args.seed + 1),
            sample_group(labeled_pool, ptx=0, drain=0, n=n, seed=args.seed + 2),
        ],
        ignore_index=False,
    )

    used_mask_labeled = labeled_pool.index.isin(test_aligned.index)
    rem_labeled = labeled_pool[~used_mask_labeled]

    test_misaligned = pd.concat(
        [
            sample_group(rem_labeled, ptx=1, drain=0, n=n, seed=args.seed + 3),
            sample_group(rem_labeled, ptx=0, drain=1, n=n, seed=args.seed + 4),
        ],
        ignore_index=False,
    )

    used_mask_labeled = used_mask_labeled | labeled_pool.index.isin(
        test_misaligned.index
    )
    labeled_remaining = labeled_pool[~used_mask_labeled].copy()

    # Symmetric train/val split over all remaining data (labeled + unlabeled).
    # Only test is special (constructed from labeled groups).
    full_remaining = pd.concat([labeled_remaining, unlabeled_pool], ignore_index=False)
    train_df, val_df = train_test_split(
        full_remaining,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=full_remaining["strat_col"],
    )

    # Ensure all labeled groups appear in both train and val.
    expected = {"0_0", "0_1", "1_0", "1_1"}
    train_labeled = train_df[train_df["Drain"].notna()]
    val_labeled = val_df[val_df["Drain"].notna()]
    missing_train = expected - set(train_labeled["strat_col"].unique())
    missing_val = expected - set(val_labeled["strat_col"].unique())
    if missing_train:
        raise ValueError(f"Train split missing groups: {sorted(missing_train)}")
    if missing_val:
        raise ValueError(f"Val split missing groups: {sorted(missing_val)}")

    if len(val_df) < 100:
        raise ValueError(
            f"Validation set has only {len(val_df)} samples (<100). "
            "Adjust split sizes or test sampling."
        )

    # Save main files expected by existing datamodule.
    train_df.to_csv(args.out_dir / "train_drain_shortcut.csv", index=False)
    val_df.to_csv(args.out_dir / "val_drain_shortcut.csv", index=False)
    test_aligned.to_csv(args.out_dir / "test_drain_shortcut_aligned.csv", index=False)
    test_misaligned.to_csv(
        args.out_dir / "test_drain_shortcut_misaligned.csv", index=False
    )

    print_counts("Train", train_df)
    print_counts("Train labeled subset", train_labeled)
    print_counts("Val", val_df)
    print_counts("Val labeled subset", val_labeled)
    print_counts("Test aligned", test_aligned)
    print_counts("Test misaligned", test_misaligned)
    print("Wrote split CSVs to", args.out_dir)


if __name__ == "__main__":
    main()

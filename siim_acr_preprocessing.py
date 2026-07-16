#!/usr/bin/env python3
"""
Convert SIIM-ACR Pneumothorax DICOM files to JPGs.
Preserves the final ID as the filename, flattens the nested structure.

Notes on conversion
Window/level: The script does a simple global min-max normalisation per image. This is what most Kaggle notebooks for this challenge did. Proper DICOM windowing (using WindowCenter/WindowWidth tags) would be more faithful radiologically but typically gives very similar results for pneumothorax CXRs since they're already well-exposed. You can swap in dcm.WindowCenter/dcm.WindowWidth if you prefer.
MONOCHROME1 inversion: A small subset of the SIIM images have inverted polarity (bright = air); the script handles that automatically.
JPG quality: 100 should be lossless? Could also easily do png though.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import pydicom
from tqdm import tqdm
import os


def _to_windows_long_path(path: Path) -> str:
    """Return a long-path-safe string on Windows; unchanged elsewhere."""
    path_str = str(path)
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str

    abs_path = str(path.resolve())
    if abs_path.startswith("\\\\"):
        # UNC path: \\server\share -> \\?\UNC\server\share
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def dicom_to_jpg(dcm_path: Path, out_path: Path, jpg_quality: int = 95) -> None:
    dcm = pydicom.dcmread(_to_windows_long_path(dcm_path))
    arr = dcm.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Normalise to [0, 255]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    else:
        arr = np.zeros_like(arr)

    img = Image.fromarray(arr.astype(np.uint8))

    # CXRs are monochrome; invert if MONOCHROME1 (bright = air)
    if getattr(dcm, "PhotometricInterpretation", "").strip() == "MONOCHROME1":
        img = Image.fromarray(255 - np.array(img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=jpg_quality)


def convert_split(dicom_root: Path, jpg_root: Path, jpg_quality: int = 95) -> None:
    dcm_files = sorted(dicom_root.rglob("*.dcm"))
    if not dcm_files:
        print(f"No .dcm files found under {dicom_root}")
        return

    print(f"Converting {len(dcm_files)} files from {dicom_root.name} ...")
    errors = []
    for dcm_path in tqdm(dcm_files):
        # Use the leaf stem (the actual image ID) as filename
        out_path = jpg_root / f"{dcm_path.stem}.jpg"
        try:
            dicom_to_jpg(dcm_path, out_path, jpg_quality)
        except Exception as e:
            errors.append((dcm_path, e))

    if errors:
        print(f"\n{len(errors)} errors:")
        for p, e in errors:
            print(f"  {p}: {e}")
    else:
        print("Done, no errors.")


def check_multi_subdir(dicom_root: Path) -> None:
    """Warn if any first-level directory contains more than one subdirectory."""
    first_level = [p for p in dicom_root.iterdir() if p.is_dir()]
    multi = {
        p: subdirs
        for p in first_level
        if len(subdirs := [s for s in p.iterdir() if s.is_dir()]) > 1
    }

    if not multi:
        print(
            f"[{dicom_root.name}] All first-level dirs have exactly one subdirectory."
        )
    else:
        print(
            f"[{dicom_root.name}] {len(multi)} first-level dirs with >1 subdirectory:"
        )
        for parent, subdirs in sorted(multi.items()):
            print(f"  {parent.name}/")
            for s in subdirs:
                n_dcm = len(list(s.rglob("*.dcm")))
                print(f"    {s.name}/  ({n_dcm} .dcm file{'s' if n_dcm != 1 else ''})")


if __name__ == "__main__":
    # python sii_acr_preprocessing.py --data_dir /path/to/dataset --out_dir /path/to/jpg-images
    import argparse

    parser = argparse.ArgumentParser(description="SIIM-ACR DICOM → JPG converter")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("."),
        help="Directory containing dicom-images-train / dicom-images-test",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("jpg-images"),
        help="Output root (subdirs train/ and test/ will be created)",
    )
    parser.add_argument(
        "--quality", type=int, default=100, help="JPEG quality (default: 100)"
    )
    args = parser.parse_args()

    for split in ("train", "test"):
        dicom_root = args.data_dir / f"dicom-images-{split}"
        check_multi_subdir(dicom_root)
        convert_split(
            dicom_root=args.data_dir / f"dicom-images-{split}",
            jpg_root=args.out_dir / split,
            jpg_quality=args.quality,
        )

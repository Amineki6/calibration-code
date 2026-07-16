import pandas as pd
import numpy as np
from betacal import BetaCalibration

CALIBRATION_WEIGHT_COL = "calibration_weight"


def fit_group_calibrators(
    val_df: pd.DataFrame, beta_params: str
) -> dict[int, BetaCalibration]:
    """
    Fits separate BetaCalibration models for each subpopulation (drain group).
    """
    calibrators: dict[int, BetaCalibration] = {}
    for group in (0, 1):
        group_df = val_df[val_df["drain"] == float(group)].copy()
        if group_df.empty:
            raise RuntimeError(f"Validation set has no samples for drain={group}")
        if group_df["label"].nunique() < 2:
            raise RuntimeError(f"Validation set for drain={group} lacks both classes")

        calibrator = BetaCalibration(parameters=beta_params)
        calibrator.fit(
            group_df["y_prob"].to_numpy(),
            group_df["label"].to_numpy(),
            sample_weight=group_df[CALIBRATION_WEIGHT_COL].to_numpy(),
        )
        calibrators[group] = calibrator
    return calibrators


def add_calibration_weights(val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds drain-stratified class-balanced weights for calibration fitting.
    """
    weighted_df = val_df.copy()
    weighted_df[CALIBRATION_WEIGHT_COL] = np.nan

    for group in (0.0, 1.0):
        group_mask = weighted_df["drain"] == group
        group_df = weighted_df[group_mask]
        if group_df.empty:
            raise RuntimeError(f"Validation set has no samples for drain={int(group)}")

        counts = group_df["label"].value_counts()
        if 0 not in counts or 1 not in counts:
            raise RuntimeError(
                f"Validation set for drain={int(group)} lacks both classes"
            )

        n_group = float(len(group_df))
        class_weights = {
            0: n_group / (2.0 * float(counts[0])),
            1: n_group / (2.0 * float(counts[1])),
        }
        weighted_df.loc[group_mask, CALIBRATION_WEIGHT_COL] = (
            group_df["label"].map(class_weights).to_numpy()
        )

    valid_group_mask = weighted_df["drain"].isin([0.0, 1.0])
    assert weighted_df.loc[valid_group_mask, CALIBRATION_WEIGHT_COL].notna().all()
    return weighted_df


def apply_group_calibration(
    df: pd.DataFrame, calibrators: dict[int, BetaCalibration]
) -> pd.DataFrame:
    """
    Applies the fitted group-specific calibrators to the dataframe probabilities.
    """
    calibrated = df.copy()
    calibrated["y_prob_calibrated"] = calibrated["y_prob"]
    for group, calibrator in calibrators.items():
        mask = calibrated["drain"] == float(group)
        if mask.any():
            calibrated.loc[mask, "y_prob_calibrated"] = calibrator.predict(
                calibrated.loc[mask, "y_prob"].to_numpy()
            )
    return calibrated

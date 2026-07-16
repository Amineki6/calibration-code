import pytest
import pandas as pd
import numpy as np
from calibration.core import fit_group_calibrators, add_calibration_weights, apply_group_calibration

def test_add_calibration_weights():
    # Create dummy data: 
    # Group 0: 3 positive, 1 negative (n=4)
    # Group 1: 1 positive, 3 negative (n=4)
    df = pd.DataFrame({
        "drain": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "label": [1, 1, 1, 0, 1, 0, 0, 0],
        "y_prob": [0.9, 0.8, 0.7, 0.1, 0.9, 0.2, 0.3, 0.1]
    })
    
    weighted_df = add_calibration_weights(df)
    
    assert "calibration_weight" in weighted_df.columns
    # Group 0 label 1 weight = 4 / (2 * 3) = 0.666...
    # Group 0 label 0 weight = 4 / (2 * 1) = 2.0
    
    g0_l1 = weighted_df[(weighted_df["drain"] == 0.0) & (weighted_df["label"] == 1)]["calibration_weight"].iloc[0]
    g0_l0 = weighted_df[(weighted_df["drain"] == 0.0) & (weighted_df["label"] == 0)]["calibration_weight"].iloc[0]
    
    assert np.isclose(g0_l1, 2/3)
    assert np.isclose(g0_l0, 2.0)

def test_fit_and_apply_calibration():
    # Dummy data
    df = pd.DataFrame({
        "drain": [0.0, 0.0, 1.0, 1.0],
        "label": [1, 0, 1, 0],
        "y_prob": [0.9, 0.1, 0.8, 0.2]
    })
    
    weighted_df = add_calibration_weights(df)
    calibrators = fit_group_calibrators(weighted_df, beta_params="abm")
    
    assert 0 in calibrators
    assert 1 in calibrators
    
    calibrated_df = apply_group_calibration(weighted_df, calibrators)
    assert "y_prob_calibrated" in calibrated_df.columns
    assert len(calibrated_df) == 4
    
    # Check predictions are roughly bounded [0, 1]
    assert calibrated_df["y_prob_calibrated"].between(0, 1).all()

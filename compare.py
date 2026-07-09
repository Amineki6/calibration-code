## COMMAND
## python compare.py --studies output/satisfied-bug output/rational-bullfinch --names "Standard" "JTT"

import argparse
import logging
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import permutation_test

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Suppress specific deepsig warnings that occur with identical data
warnings.filterwarnings("ignore", category=UserWarning, module="deepsig")

# --- 1. Import Deep Significance ---
from deepsig import aso, multi_aso


# --- 2. Data Loading Helpers ---

def get_csv_columns(df, filepath):
    """Auto-detect target and prediction columns."""
    target_candidates = ['label', 'target', 'gt', 'y_true', 'ground_truth', 'class', 'actual']
    pred_candidates = ['y_prob', 'pred', 'prediction', 'prob', 'y_score', 'probability', 'score', 'output', 'y_pred']
    
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): original for c, original in zip(df.columns, df.columns)}

    target_col = next((col_map[c] for c in target_candidates if c in cols), None)
    pred_col = next((col_map[c] for c in pred_candidates if c in cols), None)
    
    if not target_col or not pred_col:
        logging.warning(f"Skipping {filepath.name}: Could not find target/pred columns.")
        logging.warning(f"    Available columns: {list(df.columns)}")
        logging.warning(f"    Looking for one of: {target_candidates} AND one of: {pred_candidates}")
        return None, None
    
    return target_col, pred_col

def load_study_scores(study_path, study_name):
    """
    Extracts AUROCs for Aligned and Misaligned sets from all runs in a study.
    Returns a dict with lists of scores.
    """
    study_path = Path(study_path)
    eval_dir = study_path / "final_evaluation"
    
    if not eval_dir.exists():
        logging.warning(f"Missing final_evaluation for {study_name} at {eval_dir}")
        return None

    scores = {
        "name": study_name,
        "aligned": [],
        "misaligned": []
    }

    # Gather valid runs
    run_dirs = sorted(list(eval_dir.glob("run_*")))
    
    if not run_dirs:
        logging.warning(f"No runs found in {eval_dir}")
        return None

    valid_runs_count = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        
        # Define paths
        aligned_path = run_dir / f"{run_id}_aligned.csv"
        misaligned_path = run_dir / f"{run_id}_misaligned.csv"
        
        # Helper to calc AUC
        def calc_auc(csv_path):
            if not csv_path.exists(): return np.nan
            try:
                df = pd.read_csv(csv_path)
                t, p = get_csv_columns(df, csv_path) 
                if not t: return np.nan
                return roc_auc_score(df[t], df[p])
            except Exception as e:
                logging.error(f"Error reading {csv_path.name}: {e}")
                return np.nan

        # We calculate both.
        a_score = calc_auc(aligned_path)
        m_score = calc_auc(misaligned_path)
        
        scores["aligned"].append(a_score)
        scores["misaligned"].append(m_score)
        
        if not np.isnan(a_score) or not np.isnan(m_score):
            valid_runs_count += 1

    # Filter out NaNs (failed runs)
    scores["aligned"] = [s for s in scores["aligned"] if not np.isnan(s)]
    scores["misaligned"] = [s for s in scores["misaligned"] if not np.isnan(s)]
    
    if valid_runs_count == 0:
        logging.warning(f"{study_name}: No valid CSVs could be parsed.")
    
    return scores

# --- 3. Statistical Methods ---

def print_summary_stats(studies_data):
    """Prints Mean/Std table."""
    print("\n" + "="*95)
    print(f"{'METHOD':<20} | {'N':<3} | {'ALIGNED AUROC (Mean +/- Std)':<30} | {'MISALIGNED AUROC (Mean +/- Std)':<30}")
    print("-" * 95)
    
    for s in studies_data:
        n = len(s['aligned'])
        if n > 0:
            a_mean, a_std = np.mean(s['aligned']), np.std(s['aligned'])
            m_mean, m_std = np.mean(s['misaligned']), np.std(s['misaligned'])
            print(f"{s['name']:<20} | {n:<3} | {a_mean:.4f} +/- {a_std:.4f}{'':<14} | {m_mean:.4f} +/- {m_std:.4f}")
        else:
            print(f"{s['name']:<20} | 0   | {'NO DATA':<30} | {'NO DATA':<30}")
            
    print("="*95 + "\n")

def run_deep_significance_comparison(studies_data, metric_key):
    """
    Uses multi_aso (Scenario 4) to compare all models at once.
    """
    # Filter out studies with no data for this metric
    valid_studies = [s for s in studies_data if len(s[metric_key]) > 0]
    
    if len(valid_studies) < 2:
        print(f"Skipping {metric_key.upper()} analysis: Need at least 2 methods with data.")
        return

    names = [s['name'] for s in valid_studies]
    scores_dict = {s['name']: s[metric_key] for s in valid_studies}
    
    # Pass list of arrays to multi_aso
    scores_list = [scores_dict[n] for n in names]
    
    print(f"Deep Significance (ASO) Analysis: {metric_key.upper()} Set")
    print("-" * 65)
    
    try:
        # seed=1234 ensures reproducibility as per docs
        # We suppress warnings here specifically for clean output on identical data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps_matrix = multi_aso(scores_list, confidence_level=0.95, seed=1234)
        
        # Print Matrix Legend
        print(f"{'Row (A) vs Col (B)':<25} | {'eps_min':<10} | Interpretation")
        print("-" * 65)
        
        # Iterate through the matrix
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i == j: continue
                
                eps = eps_matrix[i, j]
                
                # Interpretation logic from docs
                if eps < 0.2:
                    conclusion = f"{name_a} > {name_b}"
                elif eps < 0.5:
                    conclusion = f"{name_a} > {name_b}"
                else:
                    conclusion = f"{name_a} ~ {name_b} (Inconclusive)"
                
                print(f"{name_a:<12} vs {name_b:<12} | {eps:.4f}     | {conclusion}")
    except Exception as e:
        print(f"Error during ASO calculation: {e}")
        
    print("\n")

def run_fallback_comparison(studies_data, metric_key):
    """Fallback using Permutation Test if deepsig is missing."""
    import itertools
    print(f"Permutation Test Analysis (Fallback): {metric_key.upper()} Set")
    
    combos = list(itertools.combinations(studies_data, 2))
    
    for s1, s2 in combos:
        x, y = s1[metric_key], s2[metric_key]
        if len(x) < 2 or len(y) < 2: 
            continue
        
        # Test statistic: mean(x) - mean(y)
        res = permutation_test((x, y), lambda x, y: np.mean(x) - np.mean(y), 
                               alternative='two-sided', n_resamples=9999)
        
        p_val = res.pvalue
        diff = np.mean(x) - np.mean(y)
        better = s1['name'] if diff > 0 else s2['name']
        
        sig = "*" if p_val < 0.05 else ""
        print(f"{s1['name']} vs {s2['name']}: p={p_val:.4f} {sig} (Best: {better})")
    print("\n")

# --- 4. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare study performance using ASO/Deep Significance.")
    parser.add_argument('--studies', nargs='+', required=True, type=Path, 
                        help='List of study folders (e.g. output/JTT output/Standard)')
    parser.add_argument('--names', nargs='+', help='Custom names for the studies (optional)')
    
    args = parser.parse_args()

    # 1. Load Data
    studies_data = []
    for i, path in enumerate(args.studies):
        name = args.names[i] if args.names else path.name
        data = load_study_scores(path, name)
        if data:
            studies_data.append(data)
            
    if not studies_data:
        print("No valid data loaded.")
        sys.exit(1)

    # 2. Print Summary
    print_summary_stats(studies_data)

    # 3. Run Comparisons
    for metric in ['aligned', 'misaligned']:
        run_deep_significance_comparison(studies_data, metric)

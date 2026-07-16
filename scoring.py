import torch
import numpy as np
from torcheval.metrics import BinaryAUROC
import logging
from typing import Any, Dict, Tuple, Optional


def _power_mean(values: np.ndarray, power: float) -> float:
    """Compute a power mean on [0, 1] values with a stable geometric special-case."""
    assert values.ndim == 1 and values.size > 0, "values must be a non-empty 1D array"
    assert np.all((0.0 <= values) & (values <= 1.0)), (
        "power mean expects values in [0, 1]"
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if np.isclose(power, 0.0):
            return float(np.exp(np.mean(np.log(values))))

        return float(np.mean(np.power(values, power)) ** (1.0 / power))


def _power_mean_name(power: float) -> str:
    if np.isclose(power, -1.0):
        return "harmonic_mean"
    if np.isclose(power, 0.0):
        return "geometric_mean"
    if np.isclose(power, 1.0):
        return "arithmetic_mean"
    return f"power_mean_p{power:g}"


def compute_group_fairness_score(
    logits,
    labels,
    groups,
    scope: str = "unknown",
    power: float = -1.0,
) -> Tuple[float, Dict]:
    """
    Compute a fairness-aware score from ordered pairwise group AUROCs.

    For each ordered pair of distinct groups (i, j), define
      - A_ij = AUROC(scores for y=1, g=i  vs  scores for y=0, g=j)

    The returned fairness score is the power mean over all valid ordered-pair
    AUROCs A_ij:
      - power = -1: harmonic mean
      - power = 0: geometric mean
      - power = 1: arithmetic mean
      - more negative power: stronger emphasis on the weakest pairwise AUROCs

    In the binary-group case, the two ordered pairs correspond exactly to the
    aligned and misaligned AUROCs, so this score reduces to the chosen power mean
    of those two quantities.

    Samples with missing group info (e.g., NaN, -1) are excluded from computation.

    Args:
        logits: Model scores or logits (tensor, 1D)
        labels: Ground truth labels (tensor, 1D, binary 0/1)
        groups: Group assignments (tensor, 1D, integer or may contain NaN/-1 for missing)
        scope: Human-readable context string used in warnings/debug logs to indicate
            where this fairness computation is happening (for example:
            "validation_epoch_aggregated", "utils_validation_epoch", "test_aligned").
            This does not affect metric values; it only improves log interpretability.
        power: Power-mean exponent used to aggregate ordered-pair AUROCs.
            Use 0 for the geometric mean. Default: -1.0 (harmonic mean).

    Returns:
        score: The primary fairness score returned by the function. This is the
            power mean of the valid ordered-pair AUROCs.
        details: Dictionary with summary, pairwise AUROCs, and per-group support.
    """
    # Input validation
    assert torch.is_tensor(logits), "logits must be a torch.Tensor"
    assert torch.is_tensor(labels), "labels must be a torch.Tensor"
    assert torch.is_tensor(groups), "groups must be a torch.Tensor"

    # Move to CPU for computation
    logits = logits.cpu()
    labels = labels.cpu()
    groups = groups.cpu()

    # Ensure 1D tensors
    assert logits.dim() == 1, f"logits must be 1D, got shape {logits.shape}"
    assert labels.dim() == 1, f"labels must be 1D, got shape {labels.shape}"
    assert groups.dim() == 1, f"groups must be 1D, got shape {groups.shape}"

    # Check matching sizes
    n_samples = logits.size(0)
    assert labels.size(0) == n_samples, (
        f"labels size {labels.size(0)} != logits size {n_samples}"
    )
    assert groups.size(0) == n_samples, (
        f"groups size {groups.size(0)} != logits size {n_samples}"
    )

    # Check label values are binary
    unique_labels = torch.unique(labels)
    assert torch.all((unique_labels == 0) | (unique_labels == 1)), (
        f"labels must be binary (0/1), got unique values: {unique_labels.tolist()}"
    )

    # Filter out samples with missing group info
    # Handle both NaN and negative values (e.g., -1) as missing
    if groups.dtype.is_floating_point:
        valid_mask = ~torch.isnan(groups)
    else:
        # For integer groups, treat negative values as missing
        valid_mask = groups >= 0

    n_valid = valid_mask.sum().item()
    n_missing = (~valid_mask).sum().item()

    if n_missing > 0:
        logging.debug(
            "[%s] Excluding %d/%d samples with missing group info before fairness computation.",
            scope,
            n_missing,
            n_samples,
        )

    if n_valid == 0:
        logging.warning(
            "[%s] Fairness score fallback: 0 valid group-labeled samples (%d total, %d missing). "
            "Returning fairness_score=0.0 for this scope; training/evaluation continues.",
            scope,
            n_samples,
            n_missing,
        )
        return 0.0, {"error": "no_valid_groups", "n_missing": n_missing}

    # Apply mask to filter valid samples only
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    groups = groups[valid_mask]

    # For floating point groups, convert to int after filtering
    if groups.dtype.is_floating_point:
        groups = groups.long()

    assert groups.dtype in [torch.int32, torch.int64], (
        f"groups must be integer type after conversion, got {groups.dtype}"
    )

    unique_groups = torch.unique(groups, sorted=True)
    n_groups = len(unique_groups)

    assert n_groups > 0, "No groups found after filtering"
    total_pairs = n_groups * (n_groups - 1)
    logging.debug(
        "[%s] Computing fairness score across %d groups with %d valid samples using %s.",
        scope,
        n_groups,
        n_valid,
        _power_mean_name(power),
    )

    details: Dict[str, Any] = {
        "groups": {},
        "pairwise": {},
    }

    if n_groups < 2:
        logging.warning(
            "[%s] Fairness score fallback: need at least 2 groups after filtering, got %d. "
            "Returning fairness_score=0.0 for this scope; "
            "training/evaluation continues. If checkpointing on fairness, this run will be penalized.",
            scope,
            n_groups,
        )
        details["error"] = "insufficient_groups"
        details["summary"] = {
            "fairness_score": 0.0,
            "fairness_power": float(power),
            "aggregation": _power_mean_name(power),
            "mean_pairwise_auroc": 0.0,
            "min_pairwise_auroc": 0.0,
            "n_valid_pairs": 0,
            "n_total_pairs": int(total_pairs),
            "n_groups": int(n_groups),
            "n_valid_samples": int(n_valid),
            "n_missing_samples": int(n_missing),
            "arithmetic_mean": 0.0,
            "geometric_mean": 0.0,
            "harmonic_mean": 0.0,
        }
        return 0.0, details

    positive_mask = labels == 1
    negative_mask = labels == 0
    pairwise_scores: list[float] = []

    for g in unique_groups.tolist():
        group_mask = groups == g
        details["groups"][f"group_{int(g)}"] = {
            "n_total": int(group_mask.sum().item()),
            "n_pos": int((group_mask & positive_mask).sum().item()),
            "n_neg": int((group_mask & negative_mask).sum().item()),
        }

    for pos_group in unique_groups.tolist():
        pos_mask = (groups == pos_group) & positive_mask
        n_pos = int(pos_mask.sum().item())

        for neg_group in unique_groups.tolist():
            if pos_group == neg_group:
                continue

            neg_mask = (groups == neg_group) & negative_mask
            n_neg = int(neg_mask.sum().item())
            pair_key = f"pos_g{int(pos_group)}_neg_g{int(neg_group)}"

            if n_pos > 0 and n_neg > 0:
                pair_metric = BinaryAUROC()
                pair_scores = torch.cat([logits[pos_mask], logits[neg_mask]])
                pair_labels = torch.cat(
                    [
                        torch.ones(n_pos),
                        torch.zeros(n_neg),
                    ]
                )
                pair_metric.update(pair_scores, pair_labels)
                pair_auroc = float(pair_metric.compute().item())
                assert 0.0 <= pair_auroc <= 1.0, f"pair_auroc={pair_auroc} out of range"

                details["pairwise"][pair_key] = {
                    "auroc": pair_auroc,
                    "pos_group": int(pos_group),
                    "neg_group": int(neg_group),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
                pairwise_scores.append(pair_auroc)
            else:
                details["pairwise"][pair_key] = {
                    "note": "insufficient_samples",
                    "pos_group": int(pos_group),
                    "neg_group": int(neg_group),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }

    if not pairwise_scores:
        logging.warning(
            "[%s] Fairness score fallback: no valid ordered group pairs with >=1 positive and >=1 negative "
            "samples after filtering (valid_samples=%d, groups=%d). Returning fairness_score=0.0 for this "
            "scope; training/evaluation continues. If checkpointing on fairness, this run will be penalized.",
            scope,
            n_valid,
            n_groups,
        )
        details["error"] = "no_valid_pairwise_aurocs"
        details["summary"] = {
            "fairness_score": 0.0,
            "fairness_power": float(power),
            "aggregation": _power_mean_name(power),
            "mean_pairwise_auroc": 0.0,
            "min_pairwise_auroc": 0.0,
            "n_valid_pairs": 0,
            "n_total_pairs": int(total_pairs),
            "n_groups": int(n_groups),
            "n_valid_samples": int(n_valid),
            "n_missing_samples": int(n_missing),
            "arithmetic_mean": 0.0,
            "geometric_mean": 0.0,
            "harmonic_mean": 0.0,
        }
        return 0.0, details

    pairwise_scores_np = np.asarray(pairwise_scores, dtype=np.float64)
    mean_pairwise_auroc = float(np.mean(pairwise_scores_np))
    min_pairwise_auroc = float(np.min(pairwise_scores_np))
    score = _power_mean(pairwise_scores_np, power)

    arithmetic_mean = _power_mean(pairwise_scores_np, 1.0)
    geometric_mean = _power_mean(pairwise_scores_np, 0.0)
    harmonic_mean = _power_mean(pairwise_scores_np, -1.0)

    assert 0.0 <= mean_pairwise_auroc <= 1.0, (
        f"mean_pairwise_auroc={mean_pairwise_auroc} out of range"
    )
    assert 0.0 <= min_pairwise_auroc <= 1.0, (
        f"min_pairwise_auroc={min_pairwise_auroc} out of range"
    )
    assert 0.0 <= score <= 1.0, f"score={score} out of range"

    invalid_pairs = total_pairs - len(pairwise_scores)
    if invalid_pairs > 0:
        logging.debug(
            "[%s] Excluded %d/%d ordered group pairs from fairness score due to insufficient positive/negative support.",
            scope,
            invalid_pairs,
            total_pairs,
        )

    details["summary"] = {
        "fairness_score": float(score),
        "fairness_power": float(power),
        "aggregation": _power_mean_name(power),
        "mean_pairwise_auroc": mean_pairwise_auroc,
        "min_pairwise_auroc": min_pairwise_auroc,
        "n_valid_pairs": int(len(pairwise_scores)),
        "n_total_pairs": int(total_pairs),
        "n_groups": int(n_groups),
        "n_valid_samples": int(n_valid),
        "n_missing_samples": int(n_missing),
        "arithmetic_mean": float(arithmetic_mean),
        "geometric_mean": float(geometric_mean),
        "harmonic_mean": float(harmonic_mean),
    }

    logging.debug(
        "[%s] Fairness Score (%s): %.4f | mean_pairwise_auroc: %.4f | min_pairwise_auroc: %.4f | "
        "valid_pairs=%d/%d",
        scope,
        _power_mean_name(power),
        score,
        mean_pairwise_auroc,
        min_pairwise_auroc,
        len(pairwise_scores),
        total_pairs,
    )

    return float(score), details


def compute_worst_group_accuracy(logits, labels, group_ids) -> Tuple[float, Dict]:
    """
    Compute JTT-style worst-group accuracy over intersectional groups.

    Args:
        logits:    1D tensor of raw model logits
        labels:    1D tensor of binary labels (0/1)
        group_ids: 1D integer tensor of group assignments (e.g. label*2 + attr for 4 groups)

    Returns:
        worst_acc: float — minimum per-group accuracy across all groups
        details:   per-group breakdown
    """
    assert (
        torch.is_tensor(logits)
        and torch.is_tensor(labels)
        and torch.is_tensor(group_ids)
    )
    logits = logits.view(-1).cpu()
    labels = labels.view(-1).cpu()
    group_ids = group_ids.view(-1).cpu()

    if group_ids.dtype.is_floating_point:
        valid_mask = ~torch.isnan(group_ids)
        group_ids = group_ids[valid_mask].long()
        logits = logits[valid_mask]
        labels = labels[valid_mask]
    else:
        valid_mask = group_ids >= 0
        group_ids = group_ids[valid_mask]
        logits = logits[valid_mask]
        labels = labels[valid_mask]

    preds = (logits > 0.0).long()
    details = {}
    per_group_acc = []

    for g in torch.unique(group_ids):
        mask = group_ids == g
        acc = (preds[mask] == labels[mask].long()).float().mean().item()
        per_group_acc.append(acc)
        details[f"group_{g.item()}"] = {"n": int(mask.sum()), "accuracy": acc}

    worst_acc = float(min(per_group_acc)) if per_group_acc else 0.0
    details["worst_group_accuracy"] = worst_acc
    return worst_acc, details


def compute_saab_robust_auroc(preds, labels, drains) -> Tuple[Optional[float], Dict]:
    """
    Compute robust AUROC as used in Saab et al. CXR evaluation.

    Robust subset mask:
      - positives without tube (label=1, drain=0)
      - negatives with tube (label=0, drain=1)

    Args:
        preds: 1D tensor of model scores/probabilities (higher => more positive)
        labels: 1D tensor of binary labels (0/1)
        drains: 1D tensor of binary tube labels (0/1)

    Returns:
        robust_auroc: float in [0, 1] if computable, else None
        details: metadata and failure reason (if any)
    """
    assert torch.is_tensor(preds), "preds must be a torch.Tensor"
    assert torch.is_tensor(labels), "labels must be a torch.Tensor"
    assert torch.is_tensor(drains), "drains must be a torch.Tensor"

    preds = preds.view(-1).cpu()
    labels = labels.view(-1).cpu().long()
    drains = drains.view(-1).cpu().long()

    assert preds.size(0) == labels.size(0) == drains.size(0), (
        f"Size mismatch: preds={preds.size(0)}, labels={labels.size(0)}, drains={drains.size(0)}"
    )

    robust_mask = ((labels == 1) & (drains == 0)) | ((labels == 0) & (drains == 1))
    robust_preds = preds[robust_mask]
    robust_labels = labels[robust_mask]

    n_total = int(labels.numel())
    n_robust_subset = int(robust_mask.sum().item())
    n_pos = int((robust_labels == 1).sum().item())
    n_neg = int((robust_labels == 0).sum().item())

    details: Dict[str, object] = {
        "n_total": n_total,
        "n_robust_subset": n_robust_subset,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    # Need both classes in robust subset for AUROC.
    if n_pos <= 3 or n_neg <= 3:
        details["error"] = "insufficient_class_diversity"
        return None, details

    metric = BinaryAUROC()
    metric.update(robust_preds, robust_labels)
    robust_auroc = float(metric.compute().item())
    details["robust_auroc"] = robust_auroc
    return robust_auroc, details

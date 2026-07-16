import torch
import torch.nn as nn
from typing import Optional
import logging

from .base import BaseMethod
from config import ExperimentConfig


class ScoreMatchingLoss(nn.Module):
    def __init__(self, min_subgroup_count: int = 1):
        super(ScoreMatchingLoss, self).__init__()
        self.min_subgroup_count = min_subgroup_count
        self._inactive_warning_count = 0

    def forward(
        self, probs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalizes standard deviation in average predicted scores across groups, separately
        for positive and negative class samples.

        For each group g:
            - Compute E[pred | g, y=1] (avg score for positive samples) if sufficient samples
            - Compute E[pred | g, y=0] (avg score for negative samples) if sufficient samples

        Loss: avg(std(E[pred | g, y=1]), std(E[pred | g, y=0]))
            (average of positive-class standard deviation and negative-class standard deviation)
            Returns only available standard deviation if one class has <2 valid groups.

        Args:
            probs: (B,) or (B, 1) predicted probabilities for positive class [0, 1]
            labels: (B,) true binary labels (0 or 1)
            groups: (B,) group indicators (any integer values)
            min_subgroup_count: Minimum number of examples required for each
                            (label, group) combination. A group-label subgroup
                            is only included if it meets this threshold.
                            Default: 1

        Returns:
            Scalar tensor. Returns:
            - Average of pos and neg standard deviation if both have >=2 valid groups
            - Only pos standard deviation if neg has <2 valid groups
            - Only neg standard deviation if pos has <2 valid groups
            - Detached 0.0 if both have <2 valid groups (no gradients)

        Edge cases:
            - If min_subgroup_count < 1: treated as 1 (no filtering)
            - If a (group, label) has <min_subgroup_count examples: that subgroup excluded
            - Subgroups are evaluated independently per class
        """

        # 1. Shape validation and normalization
        assert probs.dim() in [1, 2], (
            f"Probs must be 1D (B,) or 2D (B, 1), received {probs.shape}"
        )
        if probs.dim() == 2:
            assert probs.shape[1] == 1, (
                f"2D Probs must have shape (B, 1), received {probs.shape}"
            )
            probs = probs.squeeze(1)

        assert labels.dim() == 1 and labels.shape[0] == probs.shape[0], (
            f"Labels shape {labels.shape} must match probs {probs.shape}"
        )
        assert groups.dim() == 1 and groups.shape[0] == probs.shape[0], (
            f"Groups shape {groups.shape} must match probs {probs.shape}"
        )

        # 2. Value validation
        assert labels.dtype in [torch.int64, torch.int32, torch.uint8, torch.bool], (
            f"Labels must be integer/bool type, received {labels.dtype}"
        )
        assert labels.max() <= 1 and labels.min() >= 0, "Labels must be binary (0 or 1)"
        assert probs.min() >= 0.0 and probs.max() <= 1.0, (
            f"Probs must be in [0, 1], got [{probs.min():.3f}, {probs.max():.3f}]"
        )

        # Exclude missing group labels before subgroup accounting.
        if groups.dtype.is_floating_point:
            valid_group_mask = (~torch.isnan(groups)) & (groups >= 0)
        else:
            valid_group_mask = groups >= 0

        n_valid_groups = int(valid_group_mask.sum().item())
        n_missing_groups = int((~valid_group_mask).sum().item())

        if n_valid_groups == 0:
            if self._inactive_warning_count < 3:
                logging.warning(
                    "[score_matching_train_loss] Regularizer inactive for this batch: "
                    "all group labels are missing/invalid (missing=%d, batch_size=%d). "
                    "Returning score_matching_loss=0.0.",
                    n_missing_groups,
                    int(probs.shape[0]),
                )
                self._inactive_warning_count += 1
                if self._inactive_warning_count == 3:
                    logging.info(
                        "[score_matching_train_loss] Further 'regularizer inactive' warnings are suppressed "
                        "for this run to reduce log spam."
                    )
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype).detach()

        probs = probs[valid_group_mask]
        labels = labels[valid_group_mask]
        groups = groups[valid_group_mask]
        if groups.dtype.is_floating_point:
            groups = groups.long()

        # 3. Clamp min_subgroup_count to valid range (no error, just silent correction)
        min_subgroup_count = max(1, int(self.min_subgroup_count))

        # 4. Compute average scores per group per class (with independent subgroup filtering)
        unique_groups = groups.unique()
        group_pos_avgs = []
        group_neg_avgs = []

        for g in unique_groups:
            mask = groups == g
            group_labels = labels[mask]
            group_probs = probs[mask]

            # Check positive subgroup count
            pos_mask = group_labels == 1
            n_positive = pos_mask.sum().item()

            if n_positive >= min_subgroup_count:
                mean_pos_score = group_probs[pos_mask].mean()
                assert mean_pos_score.dim() == 0, "Mean positive score must be scalar"
                group_pos_avgs.append(mean_pos_score)

            # Check negative subgroup count (independently)
            neg_mask = group_labels == 0
            n_negative = neg_mask.sum().item()

            if n_negative >= min_subgroup_count:
                mean_neg_score = group_probs[neg_mask].mean()
                assert mean_neg_score.dim() == 0, "Mean negative score must be scalar"
                group_neg_avgs.append(mean_neg_score)

        # 5. Compute standard deviations based on available valid subgroups
        pos_std = None
        neg_std = None

        if len(group_pos_avgs) >= 2:
            group_pos_avgs = torch.stack(group_pos_avgs)
            pos_std = group_pos_avgs.std()
            assert pos_std.dim() == 0, "Positive standard deviation must be scalar"

        if len(group_neg_avgs) >= 2:
            group_neg_avgs = torch.stack(group_neg_avgs)
            neg_std = group_neg_avgs.std()
            assert neg_std.dim() == 0, "Negative standard deviation must be scalar"

        # 6. Return based on what's available
        if pos_std is not None and neg_std is not None:
            # Both available: return average
            total_loss = (pos_std + neg_std) / 2.0
        elif pos_std is not None:
            # Only positive standard deviation available
            total_loss = pos_std
        elif neg_std is not None:
            # Only negative standard deviation available
            total_loss = neg_std
        else:
            # Neither available: return detached zero (no gradients)
            if self._inactive_warning_count < 3:
                logging.warning(
                    "[score_matching_train_loss] Regularizer inactive for this batch: "
                    "eligible_pos_groups=%d, eligible_neg_groups=%d, unique_groups=%d, "
                    "min_subgroup_count=%d, valid_group_samples=%d, missing_group_samples=%d. "
                    "Returning score_matching_loss=0.0.",
                    len(group_pos_avgs),
                    len(group_neg_avgs),
                    len(unique_groups),
                    min_subgroup_count,
                    int(probs.shape[0]),
                    n_missing_groups,
                )
                self._inactive_warning_count += 1
                if self._inactive_warning_count == 3:
                    logging.info(
                        "[score_matching_train_loss] Further 'regularizer inactive' warnings are suppressed "
                        "for this run to reduce log spam."
                    )
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype).detach()

        assert total_loss.dim() == 0, "Final loss must be scalar"

        return total_loss


class ScoreMatchingMethod(BaseMethod):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.score_matching_loss = ScoreMatchingLoss(
            min_subgroup_count=getattr(config, "score_matching_min_subgroup_count", 1)
        )

        # Default lambda is 1.0 if not specified in config
        self.lambda_val = getattr(config, "score_matching_lambda", 1.0)

    def get_model_components(
        self, num_features: int
    ) -> tuple[nn.Module, Optional[nn.Module]]:
        # Score matching only needs a classification head.
        clf = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))
        # We return None for the projection head because we don't use it.
        return clf, None

    def compute_loss(
        self,
        model_output: tuple[torch.Tensor, Optional[torch.Tensor]],
        targets: torch.Tensor,
        extra_info: Optional[dict] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates Total Loss = BCE + Lambda * ScoreMatchingLoss
        Uses 'extra_info' to access the Drain labels.
        """
        assert extra_info is not None
        assert "drain" in extra_info.keys()

        logits, _ = model_output

        # 1. Classification Loss (Standard)
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        cls_loss = bce_loss  # Always use unweighted BCE as the primary optimization objective to match training

        # 2. Score matching loss
        score_matching_val = self.score_matching_loss(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
        )

        total_loss = cls_loss + self.lambda_val * score_matching_val

        return total_loss, {
            "bce": bce_loss.item(),
            "wbce": wbce_loss.item(),
            "score_matching_loss": score_matching_val.item(),
        }

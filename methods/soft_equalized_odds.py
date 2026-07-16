import torch
import torch.nn as nn
from typing import Optional

from .base import BaseMethod
from config import ExperimentConfig


class SoftEqualizedOddsLoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.5,
        temperature: float = 0.0,
        mode: str = "both",
        min_subgroup_count: int = 1,
    ):
        super().__init__()
        if mode not in {"tpr", "fpr", "both"}:
            raise ValueError(f"Unsupported soft EO mode: {mode}")
        self.tau = tau
        self.temperature = temperature
        self.mode = mode
        self.min_subgroup_count = max(1, int(min_subgroup_count))

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, sens: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits.view(-1))
        labels = labels.view(-1)
        sens = sens.view(-1)

        if sens.dtype.is_floating_point:
            valid = (~torch.isnan(sens)) & (sens >= 0)
        else:
            valid = sens >= 0

        if not valid.any():
            return torch.zeros((), device=probs.device, dtype=probs.dtype).detach()

        probs = probs[valid]
        labels = labels[valid]
        sens = sens[valid]
        sens = sens.long() if sens.dtype.is_floating_point else sens

        soft_dec = torch.sigmoid(self.temperature * (probs - self.tau))
        pos_mask = labels == 1
        neg_mask = labels == 0

        losses = []
        unique_groups = sens.unique()

        if self.mode in {"tpr", "both"}:
            tpr_vals = []
            for g in unique_groups:
                group_mask = sens == g
                subgroup_mask = group_mask & pos_mask
                if int(subgroup_mask.sum().item()) < self.min_subgroup_count:
                    continue
                tpr_vals.append(soft_dec[subgroup_mask].mean())
            if len(tpr_vals) >= 2:
                losses.append(torch.stack(tpr_vals).var(unbiased=False))

        if self.mode in {"fpr", "both"}:
            fpr_vals = []
            for g in unique_groups:
                group_mask = sens == g
                subgroup_mask = group_mask & neg_mask
                if int(subgroup_mask.sum().item()) < self.min_subgroup_count:
                    continue
                fpr_vals.append(soft_dec[subgroup_mask].mean())
            if len(fpr_vals) >= 2:
                losses.append(torch.stack(fpr_vals).var(unbiased=False))

        if not losses:
            return torch.zeros((), device=probs.device, dtype=probs.dtype).detach()
        if len(losses) == 1:
            return losses[0]
        return (losses[0] + losses[1]) / 2.0


class SoftEqualizedOddsMethod(BaseMethod):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.lambda_val = getattr(config, "soft_eo_lambda", 1.0)
        self.temp_start = getattr(config, "soft_eo_temp_start", 0.0)
        self.temp_end = getattr(config, "soft_eo_temp_end", 25.0)
        self.temp_schedule_epochs = max(
            1, int(getattr(config, "soft_eo_temp_schedule_epochs", 25))
        )

        self.soft_eo_loss = SoftEqualizedOddsLoss(
            tau=getattr(config, "soft_eo_tau", 0.5),
            temperature=self.temp_start,
            mode=getattr(config, "soft_eo_mode", "both"),
            min_subgroup_count=getattr(config, "soft_eo_min_subgroup_count", 1),
        )

    def set_epoch(self, epoch: int) -> None:
        progress = min(max(epoch, 0), self.temp_schedule_epochs) / float(
            self.temp_schedule_epochs
        )
        self.soft_eo_loss.temperature = (
            self.temp_start + (self.temp_end - self.temp_start) * progress
        )

    def get_model_components(
        self, num_features: int
    ) -> tuple[nn.Module, Optional[nn.Module]]:
        clf = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        return clf, None

    def compute_loss(
        self,
        model_output: tuple[torch.Tensor, Optional[torch.Tensor]],
        targets: torch.Tensor,
        extra_info: Optional[dict] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        assert extra_info is not None
        assert "drain" in extra_info

        logits, _ = model_output
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        soft_eo_val = self.soft_eo_loss(
            logits=logits, labels=targets, sens=extra_info["drain"]
        )
        total_loss = bce_loss + self.lambda_val * soft_eo_val

        return total_loss, {
            "bce": bce_loss.item(),
            "wbce": wbce_loss.item(),
            "soft_eo_loss": soft_eo_val.item(),
            "soft_eo_temperature": float(self.soft_eo_loss.temperature),
        }

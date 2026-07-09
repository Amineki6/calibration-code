import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseMethod
from config import ExperimentConfig


# Single-GPU version adapted by Claude from https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py
# (or at least I told it to)
# https://claude.ai/share/ebb8e3c5-ae51-447d-b1e9-034e80dabdea

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for binary/multi-class classification.
    Adapted from StableRep's MultiPosConLoss:
    https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py
    
    This is a single-GPU version (no distributed training).
    """
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: Feature embeddings, shape [B, D] (should be from a projection head)
            labels: Class labels, shape [B] (0 or 1 for binary classification)
        
        Returns:
            loss: Scalar supervised contrastive loss
        """
        device = feats.device
        batch_size = feats.size(0)
        
        # L2 normalize features
        feats = F.normalize(feats, dim=-1, p=2)
        
        # Compute similarity matrix: [B, B]
        similarity_matrix = torch.matmul(feats, feats.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        # mask[i, j] = 1 if labels[i] == labels[j], else 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrasts (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss
    

class SupConMethod(BaseMethod):
    """
    Training strategy using Supervised Contrastive Loss + BCE.
    """
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # Initialize the custom loss using the config parameter
        self.supcon_loss = SupervisedContrastiveLoss(temperature=config.supcon_temperature)

    def get_model_components(self, num_features: int) -> tuple[nn.Module, Optional[nn.Module]]:
        # 1. Classification Head
        clf = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 2. Projection Head (Specific to Contrastive Learning)
        # 2-layer MLP as commonly used in contrastive learning
        proj = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        return clf, proj

    def compute_loss(self, 
                     model_output: tuple[torch.Tensor, Optional[torch.Tensor]], 
                     targets: torch.Tensor, 
                     extra_info: Optional[dict] = None,
                     weight: Optional[torch.Tensor] = None
                     ) -> tuple[torch.Tensor, dict]:
        
        logits, projections = model_output

        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        cls_loss = bce_loss # Always use unweighted BCE as the primary optimization objective to match training
        
        if projections is not None:
            supcon_loss = self.supcon_loss(projections, targets)
        else:
            supcon_loss = torch.tensor(0.0, device=logits.device)

        total_loss = cls_loss + (self.config.supcon_lambda * supcon_loss)

        return total_loss, {"bce": bce_loss.item(), "wbce": wbce_loss.item(), "supcon": supcon_loss.item()}
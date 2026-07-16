import torch.nn as nn
from .base import BaseMethod


class StandardMethod(BaseMethod):
    """
    Standard supervised training using only Binary Cross Entropy (BCE) loss.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_model_components(self, num_features: int):
        # Standard training only needs a classification head.
        clf = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))
        # We return None for the projection head because we don't use it.
        return clf, None

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        logits, _ = model_output

        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        loss = bce_loss  # Always use unweighted BCE as the primary optimization objective to match training

        return loss, {"bce": bce_loss.item(), "wbce": wbce_loss.item()}

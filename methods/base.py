import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn.functional as F
import copy

from config import ExperimentConfig


class BaseMethod(ABC):
    """
    Abstract base class for training methods.
    
    This class enforces a standard interface so the training loop 
    doesn't need to know the details of the specific algorithm (Standard vs SupCon).
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config

    @abstractmethod
    def get_model_components(self, num_features: int) -> tuple[nn.Module, Optional[nn.Module]]:
        """
        Constructs and returns the specific neural network heads required by the method.
        
        Args:
            num_features: The output dimension of the encoder (e.g., 1024 for DenseNet121).
            
        Returns:
            Tuple[nn.Module, Optional[nn.Module]]: (classifier, projection_head)
        """
        pass

    @abstractmethod
    def compute_loss(self, 
                     model_output: tuple[torch.Tensor, Optional[torch.Tensor]], 
                     targets: torch.Tensor, 
                     extra_info: Optional[dict] = None,
                     weight: Optional[torch.Tensor] = None
                     ) -> tuple[torch.Tensor, dict]:
        """
        Calculates the total loss for the batch.
        
        Args:
            model_output: The tuple returned by the model forward pass (logits, projections).
            targets: The ground truth labels.
            extra_info: Any additional information (e.g., drain) to consider for loss computation.
            weight: Optional sample weights.

        Returns:
            torch.Tensor: The final scalar loss to backward on.
            dict: Loss components. (At least "bce" is always present.)
        """
        pass

    def clone(self, dataset_size: Optional[int] = None):
        """
        Creates a deep copy of the method instance.
        
        This creates a new instance with fresh loss modules and buffers.
        Subclasses can override this to handle special reinitialization logic.
        
        Args:
            dataset_size: Optional dataset size for methods that need to 
                         reinitialize buffers (e.g., DatasetScoreMatchingMethod).
                         Default implementation ignores this parameter.
        
        Returns:
            BaseMethod: A new instance of the same method class.
        """
        return copy.deepcopy(self)

    def compute_bce_terms(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute unweighted BCE and weighted BCE (wBCE).

        Returns:
            bce: Unweighted BCE over all samples.
            wbce: Weighted BCE over valid (non-NaN weight) samples, or NaN if
                no weights are provided / no valid weighted samples exist.
        """
        logits_flat = logits.view(-1)
        targets_flat = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='mean')
        wbce = torch.tensor(float('nan'), device=logits_flat.device, dtype=bce.dtype)

        if weight is not None:
            weight_flat = weight.float().view(-1).to(logits_flat.device)
            valid = ~torch.isnan(weight_flat)
            if valid.any():
                wbce = F.binary_cross_entropy_with_logits(
                    logits_flat[valid],
                    targets_flat[valid],
                    weight=weight_flat[valid],
                    reduction='mean',
                )

        return bce, wbce
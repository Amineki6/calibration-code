import torch
import torch.nn as nn
from typing import Optional

from .base import BaseMethod
from config import ExperimentConfig


class MMDLoss(nn.Module):
    """
    Self-contained implementation of Maximum Mean Discrepancy (MMD) 
    using Multi-Kernel Gaussian RBF, similar to the logic in pytorch-adapt.
    """
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: Optional[float] = None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self,
                        source: torch.Tensor, 
                        target: torch.Tensor):
        
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        kernel_mul = self.kernel_mul
        kernel_num = self.kernel_num
        fix_sigma = self.fix_sigma
        
        # 1. Compute L2 distance matrix (Equivalent to c_f.LpDistance)
        L2_distance_squared = torch.cdist(total, total, p=2.0) ** 2
        
        # 2. Bandwidth selection (Heuristic to find optimal sigma)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance_squared.detach()) / (n_samples**2 - n_samples)
            
        # 3. Multi-Kernel: Create a list of bandwidths
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        # 4. Compute Kernel Matrix (Sum of multiple Gaussians)
        kernel_val = [torch.exp(-L2_distance_squared / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)

        batch_size_source = int(source.size()[0])
        
        kernels = self.gaussian_kernel(source, target)
        
        # Break kernel matrix into blocks: XX, YY, XY, YX
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        
        # MMD Statistic: Mean(XX) + Mean(YY) - 2 * Mean(XY)
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

        del kernels
        
        return loss

class MMDMethod(BaseMethod):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.mmd_loss = MMDLoss()
        # Default lambda is 1.0 if not specified in config
        self.lambda_val = getattr(config, 'mmd_lambda', 1.0)

    def get_model_components(self, num_features: int) -> tuple[nn.Module, Optional[nn.Module]]:
        """
        Returns the Classifier and an Identity Projection.
        We use Identity because MMD must be applied to the same features 
        used for classification to enforce invariance.
        """
        # 1. Classifier
        clf = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # 2. Projection (Identity passes raw features)
        proj = nn.Identity()
        
        return clf, proj

    def compute_loss(self, 
                     model_output: tuple[torch.Tensor, Optional[torch.Tensor]], 
                     targets: torch.Tensor, 
                     extra_info: Optional[dict] = None,
                     weight: Optional[torch.Tensor] = None
                     ) -> tuple[torch.Tensor, dict]:
        """
        Calculates Total Loss = BCE + Lambda * Conditional MMD
        Conditioned on Class: 
          - Align Healthy (No Drain) <-> Healthy (Drain)
          - Align Pneu (No Drain) <-> Pneu (Drain)
        """
        assert extra_info is not None and 'drain' in extra_info.keys()
        
        logits, features = model_output
        assert features is not None

        # Handling targets which might be (labels, weights) tuple or just labels
        # Also respect weight kwarg
        sample_weights = weight
        if isinstance(targets, tuple):
            labels, tuple_weights = targets
            if sample_weights is None:
                 sample_weights = tuple_weights
        else:
            labels = targets

        # 1. Classification Loss (Standard)
        bce_loss, wbce_loss = self.compute_bce_terms(logits, labels, weight=sample_weights)
        cls_loss = bce_loss # Always use unweighted BCE as the primary optimization objective to match training
        
        # 2. Conditional MMD Loss
        drain = extra_info['drain']
        
        # Masks for Classes
        class0_mask = (labels == 0)
        class1_mask = (labels == 1)
        
        # --- Class 0 (Healthy) Alignment ---
        input_c0 = features[class0_mask]
        drain_c0 = drain[class0_mask]
        
        c0_source = input_c0[drain_c0 == 0] # No Drain
        c0_target = input_c0[drain_c0 == 1] # Drain
        
        if len(c0_source) > 0 and len(c0_target) > 0:
            mmd_class0 = self.mmd_loss(c0_source, c0_target)
        else:
            mmd_class0 = torch.tensor(0.0, device=features.device)
            
        # --- Class 1 (Pneumothorax) Alignment ---
        input_c1 = features[class1_mask]
        drain_c1 = drain[class1_mask]
        
        c1_source = input_c1[drain_c1 == 0] # No Drain
        c1_target = input_c1[drain_c1 == 1] # Drain
        
        if len(c1_source) > 0 and len(c1_target) > 0:
            mmd_class1 = self.mmd_loss(c1_source, c1_target)
        else:
            mmd_class1 = torch.tensor(0.0, device=features.device)

        # Total MMD
        total_mmd = mmd_class0 + mmd_class1

        total_loss = cls_loss + (self.lambda_val * total_mmd)
        
        return total_loss, {
            "bce": bce_loss.item(),
            "wbce": wbce_loss.item(),
            "mmd": total_mmd.item()
        }

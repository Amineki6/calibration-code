import torch
import torch.nn as nn
from torchvision.models import densenet121
from typing import Optional
from methods.base import BaseMethod


def _get_num_features(backbone: str) -> int:
    if backbone == "densenet":
        return 1024
    elif backbone == "medimageinsight":
        return 1024
    elif backbone == "medsiglip":
        return 1152
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'.")


def _build_encoder(backbone: str) -> tuple[nn.Module, int]:
    """
    Load a backbone encoder and return (encoder, num_features).
    Foundation model encoders are returned with all parameters frozen.
    """
    if backbone == "densenet":
        encoder = densenet121(weights='IMAGENET1K_V1')
        num_features = encoder.classifier.in_features
        encoder.classifier = nn.Identity()
        return encoder, num_features

    elif backbone == "medimageinsight":
        import os
        import sys
        from pathlib import Path

        repo_root = Path(os.environ["MEDIMAGEINSIGHT_REPO"]).resolve()
        sys.path.insert(0, str(repo_root))

        encoder = _LionMedImageInsightEncoder(repo_root)
        num_features = 1024
        _freeze(encoder)
        return encoder, num_features

    elif backbone == "medsiglip":
        from transformers import AutoModel
        hf_id = "google/medsiglip-448"
        model = AutoModel.from_pretrained(hf_id)
        # SiglipModel has no visual_projection; image_embeds are the L2-normalised
        # pooled vision outputs — dimension is the vision encoder's hidden size.
        num_features = model.config.vision_config.hidden_size

        encoder = _MedSigLIPEncoder(model)
        _freeze(encoder)
        return encoder, num_features

    else:
        raise ValueError(f"Unknown backbone: '{backbone}'. "
                         f"Choose from: 'densenet', 'medsiglip', 'medimageinsight'.")


def _freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class _LionMedImageInsightEncoder(nn.Module):
    def __init__(self, repo_root):
        super().__init__()
        import sys
        from pathlib import Path

        repo_root = Path(repo_root).resolve()
        sys.path.insert(0, str(repo_root))

        from MedImageInsight.UniCLModel import build_unicl_model
        from MedImageInsight.Utils.Arguments import load_opt_from_config_files

        model_dir = repo_root / "2024.09.27"

        opt = load_opt_from_config_files([str(model_dir / "config.yaml")])
        opt["LANG_ENCODER"]["PRETRAINED_TOKENIZER"] = str(
            model_dir / "language_model" / "clip_tokenizer_4.16.2"
        )
        opt["UNICL_MODEL"]["PRETRAINED"] = str(
            model_dir / "vision_model" / "medimageinsigt-v1.0.0.pt"
        )

        self.model = build_unicl_model(opt).eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.encode_image(x)


class _MedSigLIPEncoder(nn.Module):
    """Wraps google/medsiglip-448 to expose image embeddings."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                outputs = self.model.get_image_features(pixel_values=x)
            else:
                outputs = self.model(pixel_values=x)

            if isinstance(outputs, torch.Tensor):
                return outputs
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                return outputs.image_embeds
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output

            raise RuntimeError(f"Could not extract MedSigLIP image features. Got output type: {type(outputs)}")


class CXP_Model(nn.Module):
    """
    Dynamic model wrapper for CheXpert Pneumothorax detection.

    The backbone is selected via config.backbone; foundation model backbones
    are frozen (only the clf head is trained).  The classifier/projection
    head architecture is determined by the method_strategy.
    """
    def __init__(self, method_strategy: BaseMethod, backbone: str = "densenet", use_cached_features: bool = False):
        super().__init__()
        
        self.use_cached_features = use_cached_features and backbone != "densenet"
        self.backbone = backbone

        if self.use_cached_features:
            self.encoder = nn.Identity()
            num_features = _get_num_features(backbone)
        else:
            self.encoder, num_features = _build_encoder(backbone)
            
        self.clf, self.projection_head = method_strategy.get_model_components(num_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_cached_features:
            features = x
        else:
            features = self.encoder(x)
            
        logits = self.clf(features)

        if self.projection_head is not None:
            if getattr(self.projection_head, 'requires_logits', False):
                projections = self.projection_head((features, logits))
            else:
                projections = self.projection_head(features)
        else:
            projections = None

        return logits, projections

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return torch.sigmoid(logits)
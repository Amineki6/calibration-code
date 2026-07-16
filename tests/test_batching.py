from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig
from dataset import BatchData, CXP_dataset
from lightning_module import CXPLightningModule
from utils import identify_error_set


class DummyMethod:
    def __init__(self) -> None:
        self.loss_weights: list[torch.Tensor | None] = []
        self.update_weights: list[torch.Tensor | None] = []

    def clone(self, dataset_size=None):
        return self

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        assert extra_info is not None
        assert "indices" in extra_info and "drain" in extra_info
        self.loss_weights.append(weight)

        logits, _ = model_output
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1),
            targets.float().view(-1),
        )
        return loss, {"bce": float(loss.detach().cpu()), "wbce": float("nan")}

    def update_loss(self, model_output, targets, extra_info=None, weight=None):
        assert extra_info is not None
        assert "indices" in extra_info and "drain" in extra_info
        assert isinstance(targets, torch.Tensor)
        self.update_weights.append(weight)


class DummyLightningModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.clf = nn.Linear(2, 1)
        self.projection_head = None

    def forward(self, inputs: torch.Tensor):
        return self.clf(inputs), None


class DummyIdentifyModel(nn.Module):
    def forward(self, inputs: torch.Tensor):
        return inputs[:, :1], None


class MappingDataset(Dataset[BatchData]):
    def __init__(self, samples: list[BatchData]) -> None:
        self.samples = samples

    def __getitem__(self, index: int) -> BatchData:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def test_named_batch_collates_cleanly_and_hides_nan_weights():
    samples = [
        BatchData(
            indices=torch.tensor(0, dtype=torch.long),
            inputs=torch.tensor([1.0, 2.0]),
            labels=torch.tensor(0, dtype=torch.long),
            drains=torch.tensor(0.0, dtype=torch.float32),
            weights=torch.tensor(float("nan"), dtype=torch.float32),
        ),
        BatchData(
            indices=torch.tensor(1, dtype=torch.long),
            inputs=torch.tensor([3.0, 4.0]),
            labels=torch.tensor(1, dtype=torch.long),
            drains=torch.tensor(1.0, dtype=torch.float32),
            weights=torch.tensor(float("nan"), dtype=torch.float32),
        ),
    ]

    batch = next(iter(DataLoader(MappingDataset(samples), batch_size=2, shuffle=False)))

    assert isinstance(batch, BatchData)
    assert torch.equal(batch.indices, torch.tensor([0, 1]))
    assert batch.usable_weights is None


def test_dataset_returns_mapping_with_weight_field():
    dataset = object.__new__(CXP_dataset)
    dataset.path = pd.Series(["sample_a", "sample_b"])
    dataset.use_cached_features = True
    dataset.features_dict = {
        "sample_a": torch.tensor([1.0, 2.0]),
        "sample_b": torch.tensor([3.0, 4.0]),
    }
    dataset.labels = pd.Series([0, 1])
    dataset.drain = pd.Series([0.0, 1.0], dtype=np.float32)
    dataset.weights = np.array([np.nan, 2.0], dtype=np.float32)

    sample = dataset[0]

    assert isinstance(sample, BatchData)
    assert sample.indices.item() == 0
    assert torch.equal(sample.inputs, torch.tensor([1.0, 2.0]))
    assert torch.isnan(sample.weights)


def test_lightning_module_uses_normalized_batch_data():
    method = DummyMethod()
    module = CXPLightningModule(
        cast(Any, DummyLightningModel()), method, ExperimentConfig(backbone="medsiglip")
    )
    module.log = lambda *args, **kwargs: None
    module.test_method_aligned = method
    module.test_method_misaligned = method

    batch4 = BatchData(
        indices=torch.tensor([0, 1]),
        inputs=torch.randn(2, 2),
        labels=torch.tensor([0, 1]),
        drains=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([float("nan"), float("nan")]),
    )
    batch5 = BatchData(
        indices=torch.tensor([0, 1]),
        inputs=torch.randn(2, 2),
        labels=torch.tensor([0, 1]),
        drains=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([1.0, 3.0]),
    )

    module.training_step(batch4, 0)
    module.on_train_batch_end(None, batch5, 0)
    module.validation_step(batch5, 0)
    module.test_step(batch4, 0, 0)
    module.test_step(batch4, 0, 1)

    assert method.loss_weights[0] is None
    assert method.update_weights[0] is not None
    assert method.loss_weights[1] is not None
    assert method.loss_weights[2] is None
    assert method.loss_weights[3] is None


def test_identify_error_set_accepts_weighted_and_unweighted_batches():
    batch4 = BatchData(
        indices=torch.tensor([0, 1]),
        inputs=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        labels=torch.tensor([1.0, 1.0]),
        drains=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([float("nan"), float("nan")]),
    )
    batch5 = BatchData(
        indices=torch.tensor([0, 1]),
        inputs=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        labels=torch.tensor([1.0, 1.0]),
        drains=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([1.0, 3.0]),
    )

    errors_unweighted = identify_error_set(
        DummyIdentifyModel(), None, [batch4], device=torch.device("cpu")
    )
    errors_weighted = identify_error_set(
        DummyIdentifyModel(), None, [batch5], device=torch.device("cpu")
    )

    assert errors_unweighted == {0}
    assert errors_weighted == {0}

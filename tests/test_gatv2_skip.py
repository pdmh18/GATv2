import pytest
import torch
from torch import nn

from fraud_pipeline.models.gatv2 import GATv2
from fraud_pipeline.models.variants import build_model


def base_config():
    return {
        "in_dim": 4,
        "edge_dim": 2,
        "hidden_dim": 4,
        "heads": 1,
        "num_layers": 2,
        "dropout": 0.0,
    }


def test_gatv2_baseline_has_skip_disabled():
    model = build_model(
        {
            "kind": "gatv2",
            **base_config(),
        }
    )

    assert isinstance(model, GATv2)
    assert model.use_skip_connection is False


def test_gatv2_skip_variant_has_skip_enabled():
    model = build_model(
        {
            "kind": "gatv2_skip",
            **base_config(),
        }
    )

    assert isinstance(model, GATv2)
    assert model.use_skip_connection is True


def test_gatv2_skip_requires_at_least_two_layers():
    config = {
        "kind": "gatv2_skip",
        **base_config(),
    }
    config["num_layers"] = 1

    with pytest.raises(
        ValueError,
        match="use_skip_connection=True",
    ):
        build_model(config)


class AddOne(nn.Module):
    def forward(self, x, edge_index):
        del edge_index
        return x + 1.0


class TimesTwo(nn.Module):
    def forward(self, x, edge_index):
        del edge_index
        return x * 2.0


def test_skip_connection_adds_previous_layer_output():
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
    )

    edge_index = torch.tensor(
        [
            [0],
            [1],
        ],
        dtype=torch.long,
    )

    baseline = GATv2(
        **base_config(),
        use_skip_connection=False,
    )

    skip_model = GATv2(
        **base_config(),
        use_skip_connection=True,
    )

    # Thay GATv2Conv bằng các layer đơn giản, xác định được kết quả
    # để kiểm tra riêng logic residual/skip.
    baseline.convs = nn.ModuleList(
        [
            AddOne(),
            TimesTwo(),
        ]
    )

    skip_model.convs = nn.ModuleList(
        [
            AddOne(),
            TimesTwo(),
        ]
    )

    baseline_output = baseline.encode(x, edge_index)
    skip_output = skip_model.encode(x, edge_index)

    # Layer 1:
    # h1 = x + 1
    expected_h1 = x + 1.0

    # Baseline:
    # h2 = h1 * 2
    expected_baseline = expected_h1 * 2.0

    # Skip variant:
    # output = h2 + h1
    expected_skip = expected_baseline + expected_h1

    assert torch.allclose(
        baseline_output,
        expected_baseline,
    )

    assert torch.allclose(
        skip_output,
        expected_skip,
    )
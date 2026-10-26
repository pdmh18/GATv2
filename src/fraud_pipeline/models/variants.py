"""Controlled baselines sharing the transaction classifier interface."""
import torch
from torch import nn
from .gatv2 import GATv2


class FeatureMLP(nn.Module):
    def __init__(self, *, in_dim, edge_dim, hidden_dim, dropout, endpoints=False):
        super().__init__()
        self.endpoints = endpoints
        self.head = nn.Sequential(nn.Linear(edge_dim + (2*in_dim if endpoints else 0), hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_index, target_edge_index, edge_features):
        parts = [edge_features]
        if self.endpoints:
            parts.extend([x[target_edge_index[0]], x[target_edge_index[1]]])
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class SelfLoopGAT(GATv2):
    def encode(self, x, edge_index, edge_attr=None):
        return super().encode(x, edge_index[:, :0], edge_attr)


def build_model(config):
    options = dict(config)
    kind = options.pop('kind', 'gatv2')
    if kind in ('gatv2', 'gatv2_self_loops'):
        return (GATv2 if kind == 'gatv2' else SelfLoopGAT)(**options)
    if kind in ('feature_mlp', 'endpoint_mlp'):
        return FeatureMLP(**{k: options[k] for k in ('in_dim','edge_dim','hidden_dim','dropout')},
                          endpoints=kind == 'endpoint_mlp')
    raise ValueError(f'Unknown model kind: {kind}')

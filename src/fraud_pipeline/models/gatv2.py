from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2(nn.Module):
    """
    GATv2 encoder + transaction/edge classifier.

    Thiết kế của đồ án:
        Node = Account
        Edge = Transaction

    Luồng xử lý:
        Account node features
            -> GATv2Conv layers
            -> account embeddings

        [source_account_embedding
         + destination_account_embedding
         + transaction/behavioral/temporal features]
            -> MLP
            -> 1 logit / transaction
            -> sigmoid
            -> risk score in [0, 1]

    Lưu ý:
    - `edge_index` dùng để mô tả graph giữa các account.
    - `edge_features` là feature của transaction, ví dụ:
      amount, transaction_type, channel, authentication,
      avg_amount_7d, tx_count_1h, tx_count_24h,
      time_since_last_tx, new_receiver, new_device, ...
    - Theo thiết kế hiện tại, transaction features KHÔNG được đưa trực tiếp
      vào GATv2Conv. Chúng được ghép với source/destination embeddings ở
      prediction head (MLP).
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        hidden_dim: int = 64,
        edge_dim: int = 0,
        out_dim: int = 1,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        *,
        in_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Hỗ trợ cả tên cũ/new phổ biến: in_dim hoặc in_channels.
        if in_dim is None:
            in_dim = in_channels
        if in_dim is None:
            raise ValueError("Phải truyền `in_dim` hoặc `in_channels`.")

        if num_layers < 1:
            raise ValueError("`num_layers` phải >= 1.")
        if heads < 1:
            raise ValueError("`heads` phải >= 1.")
        if hidden_dim % heads != 0:
            raise ValueError(
                "`hidden_dim` phải chia hết cho `heads` "
                "để giữ đúng kích thước embedding."
            )
        if edge_dim < 0:
            raise ValueError("`edge_dim` phải >= 0.")
        if out_dim != 1:
            raise ValueError(
                "Bài toán hiện tại là binary transaction classification, "
                "nên `out_dim` phải bằng 1."
            )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Mỗi layer xuất tổng cộng hidden_dim sau khi concat các attention heads.
        per_head_dim = hidden_dim // heads

        self.convs = nn.ModuleList()

        # Layer đầu: account raw features -> account hidden embedding.
        self.convs.append(
            GATv2Conv(
                in_channels=in_dim,
                out_channels=per_head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )

        # Các layer tiếp theo: hidden_dim -> hidden_dim.
        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=per_head_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )

        # Prediction head cho transaction/edge.
        classifier_in_dim = (2 * hidden_dim) + edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Tạo embedding cho từng account node.

        Parameters
        ----------
        x:
            Node feature matrix, shape [num_nodes, in_dim].
        edge_index:
            Graph connectivity, shape [2, num_edges].
        edge_attr:
            Giữ tham số này để tương thích với code cũ.
            Theo thiết kế hiện tại, edge_attr không đi trực tiếp vào GATv2Conv.

        Returns
        -------
        Tensor
            Account embeddings, shape [num_nodes, hidden_dim].
        """
        del edge_attr  # intentionally unused in the current architecture

        h = x

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            # Activation + dropout sau mỗi GATv2 layer.
            # Với 2 layers, node có thể tổng hợp thông tin tới 2-hop.
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def classify_edges(
        self,
        node_embeddings: Tensor,
        target_edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Phân loại từng transaction/edge.

        Parameters
        ----------
        node_embeddings:
            Output của `encode`, shape [num_nodes, hidden_dim].
        target_edge_index:
            Các transaction cần dự đoán, shape [2, num_target_edges].
            Row 0 = source account indices.
            Row 1 = destination account indices.
        edge_features:
            Feature của từng transaction,
            shape [num_target_edges, edge_dim].

        Returns
        -------
        Tensor
            Logit cho mỗi transaction, shape [num_target_edges]
            khi out_dim = 1.
        """
        src = target_edge_index[0]
        dst = target_edge_index[1]

        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]

        parts = [src_emb, dst_emb]

        if self.edge_dim > 0:
            if edge_features is None:
                raise ValueError(
                    "`edge_features` không được None khi `edge_dim > 0`."
                )
            if edge_features.dim() != 2:
                raise ValueError(
                    "`edge_features` phải có shape "
                    "[num_target_edges, edge_dim]."
                )
            if edge_features.size(0) != target_edge_index.size(1):
                raise ValueError(
                    "Số dòng của `edge_features` phải bằng số target edges."
                )
            if edge_features.size(1) != self.edge_dim:
                raise ValueError(
                    f"`edge_features` có {edge_features.size(1)} cột, "
                    f"nhưng model được tạo với edge_dim={self.edge_dim}."
                )

            parts.append(edge_features)
        elif edge_features is not None and edge_features.numel() > 0:
            raise ValueError(
                "Model được tạo với edge_dim=0 nhưng lại nhận `edge_features`."
            )

        edge_input = torch.cat(parts, dim=-1)
        logits = self.edge_mlp(edge_input)

        # Binary classification: [E, 1] -> [E]
        return logits.squeeze(-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        target_edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward cho transaction classification.

        Nếu `target_edge_index` không truyền vào, model sẽ dự đoán cho chính
        các edge trong `edge_index`.

        `edge_attr` được hỗ trợ như alias tương thích cho `edge_features`.
        Nếu code cũ gọi `model(x, edge_index, edge_attr=...)`, dữ liệu đó sẽ
        được dùng ở MLP cuối, KHÔNG đưa trực tiếp vào GATv2Conv.
        """
        if edge_features is not None and edge_attr is not None:
            raise ValueError(
                "Chỉ truyền một trong hai: `edge_features` hoặc `edge_attr`."
            )

        if edge_features is None:
            edge_features = edge_attr

        if target_edge_index is None:
            target_edge_index = edge_index

        node_embeddings = self.encode(x, edge_index)

        return self.classify_edges(
            node_embeddings=node_embeddings,
            target_edge_index=target_edge_index,
            edge_features=edge_features,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        target_edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Trả về risk score / xác suất bất thường trong khoảng [0, 1].
        """
        was_training = self.training
        self.eval()

        logits = self.forward(
            x=x,
            edge_index=edge_index,
            edge_features=edge_features,
            target_edge_index=target_edge_index,
            edge_attr=edge_attr,
        )
        probs = torch.sigmoid(logits)

        if was_training:
            self.train()

        return probs


# Tên rõ nghĩa hơn nếu sau này muốn import theo vai trò của model.
GATv2EdgeClassifier = GATv2

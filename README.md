# Fraud Detection ML Pipeline - GATv2

Dự án xây dựng pipeline Machine Learning phục vụ bài toán **phát hiện giao dịch tài chính bất thường/gian lận theo thời gian thực**, kết hợp dữ liệu giao dịch, lịch sử hành vi và cấu trúc graph.

Thuật toán GNN chính đang sử dụng trong hướng nghiên cứu hiện tại là **GATv2 (Graph Attention Network v2)**.

> Trạng thái hiện tại: đã dựng cấu trúc project và thêm model GATv2 theo kiến trúc của đồ án. Các bước xử lý dữ liệu, Feature Engineering, training, tuning, evaluation và inference sẽ được hoàn thiện theo từng giai đoạn của pipeline.

---

## 1. Mục tiêu

Pipeline hướng đến các bước chính:

1. Thu thập event và dữ liệu giao dịch.
2. Tiền xử lý và làm sạch dữ liệu.
3. Feature Engineering.
4. Chia Train / Validation / Test.
5. Xây dựng và huấn luyện model.
6. Hyperparameter Tuning.
7. Đánh giá bằng Precision, Recall, F1-score, AUC-ROC.
8. So sánh với phương pháp truyền thống.
9. Lưu model.
10. Inference / dự đoán giao dịch mới.

---

## 2. Kiến trúc dữ liệu dự kiến

### Graph representation

Trong đồ án, graph được biểu diễn theo hướng:

- **Node = Account (Tài khoản)**
- **Edge = Transaction (Giao dịch)**

Mỗi tài khoản có thể chứa các đặc trưng của account và các đặc trưng hành vi đã tổng hợp.

Mỗi giao dịch có thể chứa các đặc trưng như:

- amount
- transaction_type
- channel
- authentication
- platform
- thời gian giao dịch
- behavioral features
- temporal features

Ví dụ một số feature được tạo ở bước Feature Engineering:

- `avg_amount_7d`: số tiền giao dịch trung bình trong 7 ngày gần nhất.
- `tx_count_1h`: số giao dịch trong 1 giờ gần nhất.
- `tx_count_24h`: số giao dịch trong 24 giờ gần nhất.
- `time_since_last_tx`: thời gian kể từ giao dịch gần nhất.
- `new_receiver`: người nhận có phải người nhận mới hay không.
- `new_device`: thiết bị có phải thiết bị mới hay không.

---

## 3. Luồng GATv2 trong đồ án

Kiến trúc hiện tại:

```text
Account Node Features
        ↓
GATv2Conv Layer 1
        ↓
Activation + Dropout
        ↓
GATv2Conv Layer 2
        ↓
Account Embeddings
        ↓

Source Account Embedding
        +
Destination Account Embedding
        +
Transaction / Behavioral / Temporal Features
        ↓
MLP
        ↓
Logit
        ↓
Sigmoid
        ↓
Risk Score (0 - 1)
        ↓
Threshold
        ↓
Normal / Fraud-Anomaly
```

Hai lớp GATv2 được dùng làm cấu hình khởi đầu. `num_layers` vẫn là một hyperparameter và có thể được thử nghiệm thêm trong giai đoạn tuning.

---

## 4. Cấu trúc project

```text
GATv2/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
│
├── models/
├── notebooks/
├── reports/
│
├── scripts/
│   └── run_pipeline.py
│
├── src/
│   └── fraud_pipeline/
│       ├── ingestion/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       │   ├── base.py
│       │   ├── gatv2.py
│       │   └── trainer.py
│       ├── tuning/
│       ├── evaluation/
│       ├── registry/
│       ├── inference/
│       ├── data_split.py
│       └── pipeline.py
│
├── tests/
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

---

## 5. Model GATv2 hiện tại

File chính:

```text
src/fraud_pipeline/models/gatv2.py
```

Model hiện tại được thiết kế theo bài toán **transaction/edge classification**.

GATv2 trước tiên tạo embedding cho các account node. Sau đó model lấy:

```text
[source_embedding, destination_embedding, edge_features]
```

và ghép thành một vector đầu vào cho MLP để dự đoán mức độ bất thường của từng giao dịch.

Đầu ra trước Sigmoid là **logit** cho mỗi transaction.

Sau Sigmoid:

```text
Risk Score ∈ [0, 1]
```

Ví dụ:

```text
0.05 → nguy cơ thấp
0.91 → nguy cơ cao
```

Threshold để chuyển Risk Score thành nhãn 0/1 sẽ được xác định trong giai đoạn đánh giá/tuning, thay vì mặc định coi một ngưỡng cố định là tối ưu.

---

## 6. Training label

Bài toán dự kiến là binary classification:

```text
0 = Normal
1 = Fraud / Anomaly
```

Label là **ground truth dùng để train**, không phải feature đầu vào của model.

Các trường do model hoặc rule engine tạo ra như `risk_score`, `risk_status` không được dùng ngược lại làm feature đầu vào cho chính model nếu điều đó gây data leakage.

---

## 7. Tài liệu tham khảo GATv2

### Bài báo gốc

- **GATv2 - How Attentive are Graph Attention Networks? (ICLR 2022)**  
  https://arxiv.org/abs/2105.14491

### Code chính thức của tác giả

- **Official GATv2 repository**  
  https://github.com/tech-srl/how_attentive_are_gats

### Tài liệu thư viện dùng để triển khai

- **GATv2Conv - PyTorch Geometric**  
  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html

- **GATConv - phiên bản GAT gốc để đối chiếu**  
  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html

- **PyTorch Geometric**  
  https://github.com/pyg-team/pytorch_geometric

---

# Ghi chú: GATv2 gốc và cách áp dụng trong đồ án

## GATv2 gốc

Bài báo **How Attentive are Graph Attention Networks?** đề xuất GATv2 nhằm khắc phục hạn chế **static attention** của GAT bằng cơ chế **dynamic attention**.

GATv2 gốc là một kiến trúc/layer GNN tổng quát. Nó nhận node features và cấu trúc graph để học representation cho các node. Bài báo gốc không được thiết kế riêng cho bài toán phát hiện gian lận giao dịch tài chính và cũng không quy định:

```text
Node = Account
Edge = Transaction
```

Ngoài ra, bài báo gốc không quy định prediction head riêng theo dạng:

```text
source_embedding
+ destination_embedding
+ transaction_features
→ MLP
→ Fraud / Normal
```

Đó là phần được thiết kế thêm cho bài toán của đồ án.

## Phiên bản áp dụng trong đồ án

Đồ án **không thay đổi công thức attention cốt lõi của GATv2**. Phần GATv2Conv vẫn sử dụng implementation của PyTorch Geometric.

Những phần được **điều chỉnh / bổ sung để phù hợp với bài toán** gồm:

1. **Thay đổi cách biểu diễn graph**

```text
GATv2 gốc:
Graph tổng quát, node/edge phụ thuộc từng dataset và bài toán.

Đồ án:
Node = Account
Edge = Transaction
```

2. **GATv2 được dùng làm encoder cho Account**

```text
Account Features
→ GATv2Conv
→ Account Embedding
```

Mục tiêu của hai GATv2Conv layers là tổng hợp thông tin từ neighborhood và tạo vector embedding cho mỗi tài khoản.

3. **Chuyển từ node representation sang transaction/edge classification**

Model không dự đoán trực tiếp một account là fraud hay normal.

Đối tượng cần đánh giá trong đồ án là **giao dịch**.

Vì vậy sau GATv2, model lấy embedding của tài khoản nguồn và tài khoản đích:

```text
emb_source + emb_destination
```

sau đó ghép thêm các feature của chính giao dịch:

```text
[emb_source
 + emb_destination
 + edge_features]
```

4. **Bổ sung transaction, behavioral và temporal features**

`edge_features` có thể bao gồm dữ liệu giao dịch gốc và Feature Engineering, ví dụ:

```text
amount
transaction_type
channel
authentication
platform
avg_amount_7d
tx_count_1h
tx_count_24h
time_since_last_tx
new_receiver
new_device
...
```

Các feature này được bổ sung để model không chỉ nhìn cấu trúc graph mà còn có thông tin về giao dịch hiện tại và hành vi lịch sử gần đây.

5. **Bổ sung MLP prediction head**

Sau khi ghép:

```text
emb_source + emb_destination + edge_features
```

vector này được đưa qua **MLP** để tạo một logit cho từng transaction.

```text
GATv2 Encoder
      ↓
Account Embeddings
      ↓
[emb_source + emb_destination + edge_features]
      ↓
MLP
      ↓
Transaction Logit
```

6. **Bổ sung Sigmoid và Risk Score**

Trong inference:

```text
Transaction Logit
      ↓
Sigmoid
      ↓
Risk Score từ 0 đến 1
```

Risk Score sau đó có thể được so sánh với một threshold để phân loại:

```text
0 = Normal
1 = Fraud / Anomaly
```

Threshold cuối cùng sẽ được chọn dựa trên kết quả validation/evaluation.

7. **Không đưa transaction feature trực tiếp vào GATv2Conv trong thiết kế hiện tại**

PyTorch Geometric `GATv2Conv` có hỗ trợ edge features thông qua `edge_dim`.

Tuy nhiên, trong kiến trúc hiện tại của đồ án, transaction/behavioral/temporal features được giữ lại để ghép với `emb_source` và `emb_destination` tại MLP prediction head.

Thiết kế này giúp tách rõ:

```text
GATv2
→ học representation và quan hệ giữa các Account

MLP
→ sử dụng Account Embeddings + Transaction Features
→ dự đoán mức độ bất thường của Transaction
```

Đây là **quyết định thiết kế của đồ án**, không phải thay đổi thuật toán attention bên trong GATv2.

## Tóm tắt khác biệt

```text
GATv2 gốc
────────────────────────────────
GATv2 là graph attention layer tổng quát
→ học node representation
→ dùng dynamic attention


GATv2 trong đồ án
────────────────────────────────
Giữ nguyên GATv2Conv / dynamic attention
        ↓
Node = Account
Edge = Transaction
        ↓
2 GATv2Conv layers
        ↓
Account Embeddings
        ↓
emb_source
+ emb_destination
+ transaction/behavioral/temporal features
        ↓
MLP
        ↓
Sigmoid
        ↓
Risk Score (0 - 1)
        ↓
Normal / Fraud-Anomaly
```

**Kết luận:** đồ án không sửa cơ chế GATv2 cốt lõi mà **áp dụng GATv2 như một graph encoder và bổ sung transaction classification head phù hợp với bài toán phát hiện giao dịch bất thường**.


## Dataset pipeline và output

Context/phases: [docs/TRAINING_PIPELINE_PLAN.md](docs/TRAINING_PIPELINE_PLAN.md). Mọi artifact của pipeline dataset được gom theo phiên bản và seed:

```text
outputs/
├── contextual-v4/
│   ├── multiseed.json
│   ├── hour-distribution.json
│   └── seed-20261002/        # seed-20261003, seed-20261004 tương tự
│       ├── raw/            # JSONL nguồn + manifest generator
│       ├── validation/     # validation.json, validation.md
│       ├── splits/         # train/validation/test.jsonl, manifest.json, summary.md
│       ├── features/       # causal NumPy tensors, node mapping, preprocessing, graph timeline
│       ├── baseline/       # assessment.json: rule metrics
│       └── eda/            # PNG, PDF và eda_summary.json
├── contextual-v3/          # phiên bản đối chiếu trước sửa giờ
└── archive/                # thử nghiệm, output trùng và backup cũ
```

`outputs/` bị gitignore. Generator vẫn nằm ở repo cùng cấp `../anomaly-data-pipeline`; GATv2 dùng snapshot `raw/` của từng run để tránh dữ liệu mặc định của generator thay đổi làm lệch provenance. Không tự sinh lại dữ liệu khi phân tích.

`configs/dataset.json` xác định run hiện hành, seed, source, validation report, split output và tỷ lệ 50/30/20. Defaults của các script đọc từ config này; đường dẫn config tính từ root GATv2, override CLI tính từ working directory.

### Validator, split và baseline

```bash
python3 scripts/validate_dataset.py
python3 scripts/run_pipeline.py --stage split
python3 scripts/assess_dataset.py
python3 scripts/check_hour_distribution.py
```

Validator chỉ cần standard library; kiểm tra contracts, label join, ledger và causal scenario witnesses. Loader kiểm tra source SHA256 và report đã pass trước khi split. Một mẫu cho mỗi TransferCreated/TransactionRequested; native transfer export không được tính thêm giao dịch. Split giữ ngày nguyên theo thời gian, không resample: với 181 ngày, train trước 2025-04-01, validation trước 2025-05-25, test trước 2025-07-01.

Split từ chối output đã tồn tại; `--overwrite` giữ bản trước ở `splits.previous-<id>` cùng seed. Split chỉ chứa request/target và metadata truy vết; terminal status, posted_at, risk, after balances, scenario/profile/identity/loan không là input model. Phase 3 cần replay source events theo thời gian để tạo lịch sử và graph. `--stage all` chưa triển khai và bị từ chối.

Baseline score trước khi quan sát current request, dùng lịch sử qua các tập, chọn threshold trên validation rồi giữ nguyên trên test. Precision/recall/F1/AUC/AP là đánh giá các rules amount/behavior/graph, chưa phải kết quả GATv2 hoặc chứng minh dataset sát ngân hàng thực.

### Sinh và kiểm chứng nhiều seed

Dùng environment của generator có Mimesis/Pydantic/PyYAML, chọn root mới để giữ nguyên các run đã có:

```bash
PYTHONDONTWRITEBYTECODE=1 ../anomaly-data-pipeline/.venv/bin/python scripts/check_dataset_seeds.py --output-root outputs/contextual-v4-rerun --seeds 20261002 20261003 20261004
python3 scripts/check_hour_distribution.py --output-root outputs/contextual-v4-rerun
```

Runner thực hiện generation → validation → split → baseline, đọc YAML/TOML generator và tỷ lệ split trong config GATv2. Từ chối seed directory đã tồn tại. Các report validation và baseline có directory riêng, tổng hợp nằm ở run root.

### EDA bằng Matplotlib

Trong environment có Matplotlib và NumPy:

```bash
python3 scripts/plot_dataset.py
python3 scripts/plot_dataset.py --data outputs/contextual-v4-rerun/seed-20261002/raw --validation-report outputs/contextual-v4-rerun/seed-20261002/validation/validation.json --multiseed-report outputs/contextual-v4-rerun/multiseed.json --output outputs/contextual-v4-rerun/seed-20261002/eda
```

Script kiểm tra SHA256 trước khi đọc nguồn. PNG/PDF gồm phân bố ngày/type/channel/giờ, amount/scenario, activity/failure và graph; trang baseline được ghi rõ là diagnostic rules. Normal/anomaly chuẩn hóa riêng từng class. Graph gồm cả failed requests, đảo CASH_IN thành cash → account; không phải settled graph. Histogram số request mỗi cặp dùng log10 ở trục X và thang log ở trục Y để thấy phần ít cặp hơn.

Contextual-v4 dùng anomaly off-hours probability 0.35, normal 0.05; contextual amount vẫn lệch amount/profile và channel, không bắt buộc lệch giờ. Profile/labels chỉ dùng audit offline. Overlap giờ không chứng minh ngân hàng thực đã được mô phỏng đúng; funding/failure, clipping activity và cold start còn cần xử lý.

Các artifact trước đã được di chuyển; `outputs/archive/migration-originals/` giữ metadata trước sửa provenance. Source JSONL và split JSONL giữ nguyên checksum. Một split amount-only cũ thiếu audit gốc được ghi rõ trong manifest, chỉ giữ làm lưu trữ; cần validator mới trước khi đánh giá lại.

### Phase 3: feature và graph causal

Trong environment có NumPy, chạy từ GATv2:

```bash
python3 scripts/run_pipeline.py --stage features
python3 scripts/check_feature_artifacts.py
```

Output mặc định `outputs/contextual-v4/seed-20261002/features/` theo `dataset.features_output`. Cửa sổ request 120s/1h/24h, median completed theo account+type 30 ngày, graph/node history 24h, neighborhood 600s và cold-start threshold nằm trong `configs/dataset.json`. Output đã tồn tại bị từ chối; `--overwrite` giữ backup `features.previous-<id>`.

Feature engine replay full chronological `events.jsonl`. Các request cùng timestamp chấm trên history trước nhóm đó; sau đó mới quan sát request và terminal. Failed requests góp vào request counts/receiver novelty nhưng không vào completed amount mean hoặc settled graph. Lịch sử tiếp tục qua validation/test, không cập nhật từ label. Metadata profile/credit/loan/identity/balance không là feature.

Artifact schema v2 có `transaction_features.npy` [N,35] đã scale log numeric bằng train mean/std; `preprocessing.json` và manifest ghi thứ tự cột, train fit count, cold-start/missing policy. Type/channel dùng vocabulary contract cố định có unknown bucket. Có node type one-hot nền [1041,4], node inputs tại mỗi cutoff [1041,12], `nodes.json`, timestamps, labels riêng, train/validation/test masks, raw-role target edges và money-flow target edges. ID chỉ để mapping/trace; `requests.jsonl` truy về source event line.

Graph lưu timeline sparse gồm attempt/settled, type, amount và thời gian; `graph_prefix` chỉ chứa timestamp nhỏ hơn request. `TemporalDataset.context(i)` dựng graph trong `[t-window,t)`, mặc định settled, deduplicate pairs và thêm reverse edges. CASH_IN cash → account; target classifier vẫn giữ raw source/destination. Không xuất self-loops; GATv2Conv hiện có xử lý self-loops. Node inputs gồm type one-hot và 8 feature incoming/outgoing settled count, unique partners, median amount và has-history flags trong 24h. Loader tính lại tại cutoff, không giữ history đã hết hạn của node ít hoạt động. Node scaler fit trên 256 contexts train có sẵn, không dùng validation/test.

```python
from fraud_pipeline.features.dataset import TemporalDataset

dataset = TemporalDataset("outputs/contextual-v4/seed-20261002/features")
inputs = dataset.context(100)  # x, edge_index, target_edge_index, edge_features
label = dataset.arrays["labels"][100]  # target riêng, không có trong inputs
```

Loader chưa chuyển sang Torch. Phase 4 phải giữ graph context đúng từng cutoff; không ghép request có timestamp khác nhau vào một graph muộn hơn. `check_feature_artifacts.py` xác minh checksum, shape, mask, mọi graph cutoff và sample contexts; report nằm ở `validation/features.json`. Chưa train GATv2; các module dataframe cũ không được gọi trong luồng Phase 3 này.

### Đối chiếu feature/window với dataset trước khi train

```bash
python3 scripts/analyze_feature_suitability.py
```

Environment cần NumPy/Matplotlib. Output `analysis/feature-suitability/` cùng seed gồm JSON, diagnostic arrays, PNG và PDF. Audit replay lịch sử hoàn tất theo account và account+type ở 1/7/30 ngày, mean/median, độ phủ settled bidirectional graph và phân bố feature. Options nằm trong `feature_audit` của JSON config. Train support dùng để đề xuất sửa input; held-out chỉ là diagnostics.

**Audit trước revision:** 7d cùng type chưa đủ support cho nhiều request, 600s graph cô lập đa số source và type-only node features hạn chế GATv2 phân biệt account. Candidates: median account+type 30d với relative amount/support flags, message-passing graph 24h tách 600s scalar features, bổ sung node-specific causal history. Revision đã thực hiện ở schema v2, xem mục 16 trong plan; chưa train hoặc khẳng định tăng hiệu quả.


Revision đã xuất lại seed chính: 35 transaction features, 12 node input features. `log_fee` bị loại khỏi active input do constant; source contract vẫn giữ fee. Amount reference dùng median 30d theo account+type, `amount_min_history=5`: relative amount chỉ xuất khi đủ mẫu; thiếu thì zero cùng cờ reliability, median/count quan sát vẫn giữ. Không lọc history bằng label. Node features tính từ settled money-flow timeline; scaler log numeric của node fit trên available known nodes ở sampled train contexts, flags/type không scale. `node_features.npy` chỉ lưu static one-hot, phần dynamic được reconstruct qua loader.

Kiểm tra model thật trên CPU trong environment có Torch/PyG:

```bash
python3 scripts/check_model_inputs.py
```

Lệnh chỉ forward/backward hai train rows, derive dimensions từ schema, không optimizer step và không tính evaluation metrics. Report `validation/model-inputs.json` ghi versions đã kiểm tra. Graph 24h lớn hơn bản trước: audit samples đạt tối đa 7.638 directed edges; chi phí replay/node reconstruction/batching cần benchmark ở Phase 4.

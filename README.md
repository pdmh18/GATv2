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

Split từ chối output đã tồn tại; `--overwrite` giữ bản trước ở `splits.previous-<id>` cùng seed. Split chỉ chứa request/target và metadata truy vết; terminal status, posted_at, risk, after balances, scenario/profile/identity/loan không là input model. Phase 3 cần replay source events theo thời gian để tạo lịch sử và graph. `--stage all` chạy validate → split → features → train; reuse artifact chỉ khi checksum/policy khớp.

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

Loader chưa chuyển sang Torch. Phase 4 phải giữ graph context đúng từng cutoff; không ghép request có timestamp khác nhau vào một graph muộn hơn. `check_feature_artifacts.py` xác minh checksum, shape, mask, mọi graph cutoff và sample contexts; report nằm ở `validation/features.json`. Trainer thật được mô tả ở Phase 4 bên dưới; các module dataframe cũ không được gọi trong luồng causal này.

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

### Phase 4: train GATv2 thật

CPU dependencies đã dùng được pin trực tiếp trong `requirements-training.txt` (chưa là lockfile toàn bộ transitive dependencies). Trong Python environment phù hợp:

```bash
make setup-training
make train
# Từ raw dataset tới checkpoint, reuse artifact khớp provenance:
make all
```

Hoặc dùng uv với CPU index:

```bash
uv run --no-project --index https://download.pytorch.org/whl/cpu --with-requirements requirements-training.txt python scripts/run_pipeline.py --stage train
```

Nix dùng native wheels cần runtime libstdc++/zlib trong library path của dev shell. Run đã kiểm tra với Python 3.13, Torch 2.14.1+cpu, PyG 2.8.0.post1, NumPy 2.5.2, Matplotlib 3.11.2. `make train CONFIG=... PYTHON=...` cho phép chọn config/environment. Các tham số nằm trong `configs/dataset.json.training`; không dùng `model.fit()` hoặc vòng lặp đạt target giả.

Trainer dùng BCEWithLogitsLoss, pos_weight=negative/positive chỉ từ train, AdamW, gradient clipping, CPU deterministic seed, validation mỗi epoch và early stopping theo weighted validation loss. Score sigmoid là score của objective có class weight, chưa là probability đã calibration. Threshold report 0.5 là cấu hình cố định; chọn threshold/tuning là Phase 5. Validation thiếu một class thì AUC/AP ghi null, không dựng điểm giả.

Batch chia theo nhóm thời gian 1h, lấy graph/node context tại request đầu nhóm rồi dùng lại cho các request sau; graph không có timestamp mới hơn target. Feature giao dịch vẫn exact per-request. Microbatch tối đa 128 targets. Node xuất hiện sau cutoff ánh xạ unknown tới lần refresh sau. Snapshot conservative có thể bỏ thông tin graph mới trong giờ đó; đây là tradeoff cần ablation, không tương đương exact graph per-request. Cache có feature/policy identity và checksum, không chứa label.

Output `outputs/contextual-v4/seed-20261002/training/gatv2-v1/`:

- `best.pt`, `last.pt`: model architecture/state, optimizer state, epoch và feature identity.
- `config.json`, `provenance.json`, bản sao feature manifest/schema/preprocessing/node mapping.
- `history.json`, `summary.json`, validation predictions, PNG/PDF learning curves.
- `contexts/`: cached causal snapshots; run đầu khoảng 403 MiB với 3.456 snapshots.

Run thật: train 251.481, validation 148.653, 3 epoch khoảng 180s CPU; best epoch 2 theo validation loss. Validation precision 0,809, recall 0,990, F1 0,890, ROC-AUC 0,9997, AP 0,9816 tại threshold 0,5. Epoch 3 có F1/AP cao hơn nhưng loss cao hơn; checkpoint vẫn theo rule đã chốt. Reload best checkpoint đã cho cùng predictions. **Chưa đánh giá test hoặc chứng minh graph vượt tabular.**

Existing run bị từ chối; chọn root mới hoặc `--overwrite` giữ backup. Ví dụ tái tạo end-to-end:

```bash
python3 scripts/run_pipeline.py --stage all --output outputs/contextual-v4/seed-20261002/training/gatv2-v1-rerun
```

`all` xác minh audit/source SHA, split checksum/ratios, feature input/policy/code provenance trước khi reuse; stale artifacts bị từ chối. `--overwrite` trên `all` revalidate/re-export split/features có backup, không đổi raw source. Đây là offline dataset→checkpoint, chưa bao gồm generator, Kafka hay inference.


### Phase 5: baseline, ablation và evaluation

```bash
make evaluate CONFIG=configs/dataset.json
# hoặc trong environment đã cài requirements-training.txt:
python scripts/evaluate_phase5.py --config configs/dataset.json
```

`evaluation` trong config chứa output và rule feature columns. Runner mặc định chỉ đánh giá GATv2 Phase 4. Chọn threshold trên validation, ghi `selection.json` trước khi score test. Không ghi đè evaluation cũ: đặt output mới cho một run mới. `make all` vẫn dừng tại train; evaluate là bước riêng có mở test.

Run `outputs/contextual-v4/seed-20261002/evaluation/phase5-v1/` đã hoàn tất: feature MLP được validation chọn, test F1 0.9635/AP 0.9842; GAT đầy đủ F1 0.9466/AP 0.9751. Chưa thấy lợi ích message passing trong cấu hình hiện tại; feature giao dịch vẫn gồm thống kê lân cận graph. Một seed synthetic không chứng minh hiệu quả ngân hàng thật. Xem mục 18 trong plan để biết threshold, breakdown lỗi, benchmark và giới hạn.

Output gồm `report.md`, `report.json`, `selection.json`, checkpoint các baseline, test predictions NPZ, bộ hình báo cáo trong `figures/`. Chỉ số là đánh giá model trên dataset này; AP là average precision, không phải trapezoidal PR-AUC. Giá trị threshold rule có thể ngoài [0,1]; sigmoid score từ weighted BCE chưa calibration.


Xuất lại biểu đồ từ dự đoán test đã lưu (không train/score/chọn threshold lại):

```bash
python scripts/plot_evaluation.py --run outputs/contextual-v4/seed-20261002/evaluation/phase5-v1
# hoặc make plot-evaluation
```

Thư mục `evaluation/phase5-v1/figures/` có ba hình cần cho báo cáo: `roc_pr_curves` (PR + ROC của GATv2), `confusion_matrix` (GATv2 là model nghiên cứu chính, count và % trong cùng ô), `gatv2_metrics` (bảng chỉ số test GATv2). Mỗi hình xuất PNG/PDF; `evaluation_charts.pdf` gom 3 trang. Bộ chẩn đoán mở rộng cũ nằm trong `appendix/`, không xuất mặc định. Risk score từ weighted BCE chưa calibration. Exporter kiểm tra ROC/AP/confusion từ predictions khớp report và checksum selection không đổi.


Phạm vi đánh giá hiện tại: **chỉ GATv2**. Config mặc định không train thêm variant/rule baseline. Bộ báo cáo chính dùng `gatv2_report.md/json`; `figures/` chỉ có ROC/PR GATv2, confusion matrix GATv2 và `gatv2_metrics` (PNG/PDF), PDF tổng hợp 3 trang. Các kết quả comparison trước đây là lịch sử, không đưa vào báo cáo hiện tại.


Đối chiếu đầy đủ **GATv2 train / validation / test**:

```bash
python scripts/evaluate_gatv2_splits.py
```

Dùng cùng best checkpoint và threshold đã chọn trên validation cho cả ba tập, eval mode không dropout. Train cần thu thêm predictions; validation/test dùng predictions đã lưu. Output `gatv2_partitions/` lưu report và predictions từng tập. Bộ hình chính vẫn 3 trang: PR/ROC cả ba tập, ba confusion matrix phân biệt rõ split, và tỷ lệ/số lượng split cùng bảng chỉ số. `plot_evaluation.py` tự dùng bộ đầy đủ khi có `gatv2_partitions/report.json`. Tỷ lệ config 50/30/20 theo ngày liên tục, không phải lấy tỷ lệ trong mỗi ngày; tỷ lệ số request thực tế gần đúng vì lưu lượng ngày khác nhau.


### Dọn dữ liệu sinh lại được

```bash
make clean-data                    # xóa raw/splits/features, cache context và dữ liệu backup cũ
make generate-dataset              # sinh lại raw seed 20261002 bằng generator repo cùng cấp
# Trong environment training đã cấu hình, dựng lại split/features và train một run mới:
python scripts/run_pipeline.py --stage all --output outputs/contextual-v4/seed-20261002/training/gatv2-rebuilt
```

Preview trước khi dọn: `python scripts/clean_generated_data.py`. Cleaner chỉ dọn payload trong `outputs/`, giữ checkpoint, báo cáo, hình và saved evaluation predictions. Log đường dẫn đã dọn nằm trong `outputs/cleanup.json`. Sau cleanup, training/feature replay cần dựng lại input; hình báo cáo train/validation/test vẫn xuất lại được từ predictions đã giữ. Generator cần environment/dependencies của repo `../anomaly-data-pipeline`; `make generate-dataset` chỉ tạo raw, không tự train lại.


### Feature EDA trước khi rút gọn input GATv2

```bash
make feature-eda PYTHON=python
# Nếu còn tensor, có thể chạy trực tiếp:
python scripts/audit_features.py --config configs/dataset.json
```

`feature_eda` trong config đặt sample/correlation threshold, số node context, số validation context và permutation repeats. Khi tensor đã dọn, target Make dùng `--temporary-rebuild`: cần environment có `requirements-training.txt` cùng dependencies generator (mimesis, pydantic, PyYAML); raw/split/features được dựng trong thư mục tạm và tự dọn khi hoàn tất. Khi đã có feature artifact, runner sử dụng artifact đó. Thêm `--overwrite` nếu chủ đích tái tạo báo cáo audit đã tồn tại.

Output `outputs/contextual-v4/seed-20261002/eda/feature-audit-v1/`: `report.md/json`, CSV thống kê/correlation/history coverage, heatmap tương quan transaction/node, biểu đồ permutation importance và PDF `feature_audit.pdf`. Correlation và biến thiên đo trên train; grouped permutation chỉ dùng validation và checkpoint GATv2 cố định. Test chỉ xuất chỉ số lịch sử để đối chiếu, không dùng chọn feature. Permutation không phải retrained ablation; graph/node embeddings giữ nguyên và không đo importance của encoder. Correlation cao, low variance hoặc importance thấp không tự động loại feature.

Run audit hiện tại tái tạo khớp 21/21 payload file của input checkpoint, 10 invariant checks và 15 causal regression tests pass. Có hai cột transaction hằng (`type_unknown`, `channel_unknown`) và node unknown hằng trong sample known nodes. Đây là fallback schema, chưa nên xóa tùy tiện. Cặp transfer incoming count/amount có Spearman 0.9989 nhưng chỉ 0.5488 khi bỏ joint modal values (99.08% sample cùng giá trị phổ biến). Train cold-start 7.95%, validation/test 0%; median reference reliable 88.43% so với khoảng 96.2% ở holdout. Những khác biệt này cần tính đến khi đọc gap train/validation/test. Ablation retrain chưa chạy.


### Feature reviewer và window động

Đã thêm schema thử nghiệm `configs/features-v3.json`: mean completed theo account+type với window tùy chọn, request count 5m và optional auth/device/platform + new_device/missing flags. Giữ median cũ và legacy config riêng. `make configure-window WINDOW=1d` (hoặc `1w`, `1mo`) tạo profile tách output; có thể override mean/median/graph/node window độc lập. `1mo=30 ngày`. Xem [docs/FEATURE_WINDOWS.md](docs/FEATURE_WINDOWS.md) về mapping, nguồn dữ liệu còn thiếu và protocol đánh giá. Cần train GATv2 mới; checkpoint 35 chiều không đánh giá được schema 58 chiều.


### Device trong dữ liệu sinh

Generator repo cùng cấp đã bổ sung `transactions.device.device_id/platform` và request event `device_id/platform`, có RNG riêng và tỷ lệ đổi thiết bị cấu hình trong TOML. Validator đối chiếu snapshot/request. Dùng `configs/device-v1.json` với output riêng `outputs/contextual-v5-device/`; xem mục Device trong [docs/FEATURE_WINDOWS.md](docs/FEATURE_WINDOWS.md). Đã sinh 500k và train/evaluate đối chứng GATv2; chưa triển khai enrichment trong Go Backend.


### Train đối chứng tác động device trên GATv2

```bash
make generate-dataset DATASET_RAW=outputs/contextual-v5-device/seed-20261002/raw
python scripts/train_device_experiment.py --config configs/device-v1.json
```

Runner kiểm tra legacy 35-feature projection, labels/time/graph/node inputs khớp lần đầu bằng SHA256; train hai GATv2 cùng 58 chiều/số tham số/seed/budget: bật device và che 9 cột device trong suốt train/inference. Mean/5m/auth thiếu dữ liệu giống nhau giữa hai nhánh. Giữ checkpoint theo validation loss; freeze threshold cả hai nhánh bằng validation trước khi mở test. So original-35 chỉ mô tả thay đổi tổng thể; tác động device được đối chiếu giữa hai nhánh cùng 58 chiều. Artifact ở `outputs/contextual-v5-device/seed-20261002/experiments/device-v1/`; không ghi đè experiment cũ. Chỉ một seed, chưa là bằng chứng generalization hoặc significance.

### Áp dụng review v2 và chạy inference GATv2

[docs/REVIEW_V2_IMPLEMENTATION.md](docs/REVIEW_V2_IMPLEMENTATION.md) đối chiếu từng góp ý, hướng dẫn export bundle từ checkpoint/threshold validation đã freeze và replay một request hoặc chạy Kafka worker. Input/response giữ contract trong `specs/review.md`; output chuẩn hóa và cách BE/FE hiển thị nằm trong [docs/MODEL_RESPONSE_CONTRACT.md](docs/MODEL_RESPONSE_CONTRACT.md) + [specs/model_response.v1.json](specs/model_response.v1.json). Feature/scaler và causal graph dùng chung với offline. History hỗ trợ JSONL, native EventStoreDB banking và Mongo event projection. Chưa kiểm chứng hạ tầng thật hoặc tích hợp consumer kết quả trong Go/FE.

`configs/review-v2.json` là profile thử nghiệm graph/node history 7 ngày, mean 7 ngày, median 30 ngày, output riêng; cần export/train mới. Window 30 ngày có thể tạo bằng `configure_feature_window.py --base configs/device-v1.json --graph-window 1mo --node-window 1mo --output outputs/configs/review-graph-30d.json`. Không đổi cửa sổ của checkpoint đã train khi inference.

Đã train profile review-v2 (500k, seed 42, 3 epoch): chọn epoch 2 theo validation loss, threshold validation 0.94979429. Test F1 0.94619 (precision 0.94176, recall 0.95067), so full-device 24h 0.93218; gần ngang original 35 chiều 0.94661. Báo cáo `outputs/contextual-v5-device/seed-20261002/experiments/review-v2/report.md` và biểu đồ đầy đủ train/validation/test trong evaluation output của config. Một seed, cùng holdout lịch sử; chưa chứng minh generalization hoặc significance.

### Bộ tiêu chí v2 và pipeline EDA

[Đặc tả tiêu chí](specs/Bo_tieu_chi_phat_hien_giao_dich_bat_thuong.md) giữ nguyên 49 mã, tách dấu hiệu hành vi, dữ kiện xác minh và ngữ cảnh cần rà soát; quy định unit, thời điểm, settled/projected, baseline và ba trạng thái matched/not_matched/insufficient_data. Không dùng matched làm anomaly label mặc định hoặc cộng điểm luật vào sigmoid. Catalog [anomaly_criteria.v2.json](specs/anomaly_criteria.v2.json) là nguồn cho bảng tiêu chí; sửa catalog rồi chạy `python scripts/render_criteria_spec.py`, kiểm tra bằng `make criteria-spec-check`.

```bash
make dataset-eda EDA_CONFIG=configs/criteria-eda-v2.json PYTHON=python
```

Chạy trong environment có dependencies training. Pipeline gồm dataset plots Matplotlib, feature EDA/correlation/validation permutation trên đúng checkpoint GATv2 58 chiều/7d, và criteria source-readiness. Config EDA riêng giữ nguyên SHA config đã dùng train. Output `outputs/contextual-v5-device/seed-20261002/eda/criteria-v2/` chứa README, ba nhóm báo cáo JSON/Markdown/CSV và PDF/PNG. Nếu thư mục đã tồn tại, đổi `eda_pipeline.output` sang phiên bản mới; không ghi đè kết quả cũ mặc định. Không sinh lại dataset, nhãn hoặc train model trong lệnh này.

Kết quả 2026-10-19: 10 leakage screening checks và 15 causal tests pass; payload khớp checkpoint. Có 13 cột transaction hằng (gồm auth thiếu, cờ metadata luôn có và fallback categories); không tự động loại. Permutation validation: velocity/transfer-neighborhood/habits nhạy nhất; device AP drop khoảng 0.00095, auth bằng 0 vì không có nguồn. Đây là sensitivity của head với graph/node embeddings cố định, không phải retrained ablation hoặc đo toàn bộ GAT encoder.

Source readiness: 25/49 có đủ nhóm nguồn được liệt kê xuất hiện, 15 thiếu một phần, 9 thiếu nguồn; chưa xác nhận criterion matching, completeness/watermark hoặc support từng account. Có 23.731 device nhưng 0 shared-device; device/platform coverage 100%, auth coverage 0%. Train không có đủ 90/91 ngày lịch tham chiếu cho A01/A03; A04 với cấu hình đề xuất 210 ngày vượt toàn bộ 181 ngày dataset. Cần thêm prehistory/kịch bản/ngữ cảnh trước khi kỳ vọng model bao phủ các tiêu chí này.

### Account ID của server khi inference

Request có `sourceAccountId`/`destinationAccountId`; predictor hỗ trợ account mới
bằng node riêng theo policy `distinct-account-nodes-v1`. Không cần đổi ObjectID
thành UUID training, không gộp account mới vào một unknown node. Giữ nguyên bundle
và policy feature/history; xem [account identity](docs/INFERENCE_ACCOUNT_IDENTITY.md)
về cold-start, giới hạn và kiểm tra trước khi áp dụng cho dữ liệu thật.

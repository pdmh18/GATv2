# Kế hoạch pipeline train GATv2 phục vụ Anomaly

Cập nhật: 2026-10-09. Tài liệu bàn giao context và theo dõi các phase; các mục chưa đánh dấu vẫn là công việc dự kiến.

## 1. Mục tiêu và phạm vi

Xây pipeline train offline từ dữ liệu tổng hợp, đánh giá GATv2 cho bài toán phát hiện giao dịch bất thường, đóng gói model cùng cách xử lý feature, rồi tích hợp vào Anomaly.

Ba repo cùng cấp:

| Repo | Vai trò |
|---|---|
| `../anomaly-data-pipeline` | Sinh dataset có seed, event lịch sử và ground-truth labels |
| `GATv2` (repo này) | Loader, split, feature, graph, train, evaluation, artifact và inference |
| `../anomaly` | Ứng dụng ngân hàng; phát canonical event và sử dụng kết quả chấm điểm |

Luồng đích:

```text
Dataset JSONL + manifest + labels
  → validate → chronological split
  → causal features + graph
  → train → validation/threshold → test
  → model bundle → offline replay inference
  → tích hợp canonical event của Anomaly
```

Ưu tiên hoàn thành từng phase và kiểm chứng đầu ra trước khi chuyển tiếp. Hiện tập trung dữ liệu; chưa triển khai training hoặc realtime.

## 2. Các quyết định đã thống nhất

- Chấm điểm tại thời điểm yêu cầu giao dịch, trước kết quả xử lý và trước cập nhật số dư.
- Ground truth nằm riêng trong `labels.jsonl`, join bằng `transaction_id`; `event_id` của label trỏ vào event yêu cầu.
- Binary target: normal = 0, anomaly = 1. Không đồng nhất giao dịch thất bại/hủy với anomaly.
- Chia theo các khoảng thời gian liên tiếp: 50% đầu train, 30% tiếp validation, 20% cuối test. Tỷ lệ theo thời gian, không phải lấy 50/30/20 giao dịch trong từng ngày; số giao dịch thực tế có thể khác tỷ lệ này.
- Không dùng dữ liệu tương lai để tạo feature/graph. Encoder, imputer, scaler fit trên train; threshold và hyperparameter chọn trên validation. Test chỉ dùng cho đánh giá cuối.
- Node nền là account; đối tác ngoài hệ thống có loại node riêng. Transaction features được ghép với source/destination embeddings tại MLP, không đưa trực tiếp vào GATv2Conv trong kiến trúc hiện tại.
- Tham số nghiệp vụ đặt trong config, runner điều phối mỏng, xử lý chia thành các module có trách nhiệm rõ.
- Dữ liệu tổng hợp phục vụ nghiên cứu/thử nghiệm, chưa chứng minh hiệu quả phát hiện gian lận thực tế.

## 3. Trạng thái đã kiểm tra

### Dataset generator

Source snapshot: `anomaly-data-pipeline` commit `6116fee` (`transaction life cycle`). Manifest hiện có schema version 2, seed `20261002`, kỳ mô phỏng 181 ngày từ 2025-01-01 đến trước 2025-07-01.

| Output | Số dòng trong manifest hiện tại |
|---|---:|
| customers / accounts | 100 / 100 |
| transactions / labels | 2.500 / 2.500 |
| counterparties | 13 |
| transfer_events | 1.666 |
| events | 5.544 |

- Đã xử lý transaction theo thứ tự thời gian toàn cục; transfer thành công cập nhật số dư nguồn và đích.
- Người nhận là account đã tồn tại, ưu tiên nhóm người nhận quen. Merchant/cash counterparty dùng pool chung, ghi ở `counterparties.jsonl`.
- Đã loại `risk` sinh theo nhãn; `posted_at` null khi thất bại.
- Amount outlier theo mức giao dịch điển hình của khách hàng; có giao dịch lớn hợp lệ chồng lấn. Chưa có velocity/account takeover/structuring scenario.
- `events.jsonl`: envelope snake_case chung, schema 2. Payload transaction dùng field camelCase.
- `transfer_events.jsonl`: envelope đúng contract transfer của Anomaly, schema 1. Đây là một cách xuất của các transfer trong `events.jsonl`, không phải giao dịch bổ sung; không ingest cả hai rồi đếm hai lần.
- Kiểm tra ở phiên triển khai trước: hai bài unittest pass, gồm bộ 1.200 giao dịch kiểm tra ledger/contract/labels/reproducibility/report và trường hợp một account không đủ để sinh transfer. Đây chưa phải kiểm chứng replay vào Anomaly.

Lifecycle hiện có:

| Nghiệp vụ | Event yêu cầu, sequence 1 | Event kết quả, sequence 2 |
|---|---|---|
| TRANSFER | `TransferCreated`, status `awaiting_otp` | `TransferCompleted` / `TransferCancelled` |
| CASH_IN, CASH_OUT, PAYMENT, DEBIT | `TransactionRequested`, status `requested` | `TransactionPosted` / `TransactionRejected` |

Request payload có source/destination ID, amount, fee, currency, channel; ngoài transfer có thêm `transactionType`. Kết quả có balance snapshot. Request và kết quả hiện cùng timestamp, phân biệt bằng sequence.

Giới hạn cần giữ trong context:

- Chưa sinh `TransferOTPUpdated`, OTP expiry hay độ trễ xử lý.
- Loan events chưa cập nhật ledger số dư transaction. Không dùng loan disbursement để tự suy ra balance cho đến khi có quy tắc thống nhất.
- ID tổng hợp cần mapping khi nhập vào account thực của Anomaly.
- Hồ sơ customer/credit có thể chứa lịch sử; phải kiểm tra thời điểm từng thông tin trước khi dùng làm feature.

### GATv2

Source snapshot: commit `6f31446` (`create model gatv2`).

- `src/fraud_pipeline/models/gatv2.py` có encoder và edge classifier thực; hỗ trợ tách `edge_index` lịch sử khỏi `target_edge_index` cần dự đoán.
- `pipeline.py` mới in thông báo stage, chưa truyền dữ liệu/artifact. Training loop đang gán `reached = True`.
- Trainer gọi `model.fit()`, không khớp GATv2 `torch.nn.Module`. Inference hiện cũng chưa khớp input graph/tensor.
- Graph builder, tuning và registry save/load chưa hoàn thiện.
- Behavioral count/mean hiện tổng hợp toàn dataset; preprocessing fit scaler toàn dataset. Không dùng nguyên các hàm này cho đánh giá theo thời gian.
- Loader config tính root thành `src/`, cần sửa trước khi dùng CLI với config ở root.

### Anomaly

Canonical financial journal nằm trong EventStoreDB stream `banking`; MongoDB phục vụ projection/read model. Contract transfer ở `../anomaly/server/internal/domain/transaction/event.go`, journal envelope ở `bank.go`.

Event transfer hỗ trợ `TransferCreated`, `TransferOTPUpdated`, `TransferCompleted`, `TransferCancelled`. Đọc trường `Event` trong banking record cho downstream; không chuyển toàn bộ account/profile credentials vào detector. Scaffold Kafka → EventStore của GATv2 chưa phản ánh đường tích hợp này.

## 4. Các phase và tiêu chí hoàn thành

### Phase 1 — Kiểm chứng và chốt dataset (hoàn thành trong phạm vi feature ban đầu)

- [x] Chỉnh lifecycle, sequence, shared counterparties, chronological balances và loại risk sinh sẵn.
- [x] Sinh lại dataset/report, thêm kiểm tra hồi quy generator.
- [x] Tạo lệnh validate dataset hiện có, không chỉ kiểm tra dataset nhỏ được sinh trong test.
- [x] Kiểm tra schema theo event type, ID duy nhất, label join một-một, lifecycle hợp lệ, request trước terminal, timestamp/sequence và reference account/counterparty.
- [x] Replay ledger; kiểm tra số dư trước/sau, fee, failed/cancelled và bảo toàn giá trị của transfer theo fee.
- [x] Báo cáo phân bố theo ngày/tháng: count, amount, type, channel, status, normal/anomaly; thống kê graph và người nhận lặp lại.
- [x] Kiểm tra lịch sử credit/loan theo thời gian; ghi rõ phần chưa nhất quán và loại khỏi feature nếu cần.
- [x] Người dùng chốt tất cả nghiệp vụ: TRANSFER, CASH_IN, CASH_OUT, PAYMENT, DEBIT. Account/counterparty roles và cash-flow direction ghi trong feature policy; tensor graph chốt ở Phase 3.

Đầu ra: validation report, dataset manifest có thể truy vết, danh sách feature/event được phép dùng. Hoàn thành khi không còn lỗi contract/ledger/label và các giới hạn dữ liệu được ghi rõ.

### Phase 2 — Loader và chia thời gian

- [ ] Loader JSONL trong GATv2; đường dẫn dữ liệu cấu hình được, mặc định nguồn là repo cùng cấp.
- [ ] Trích event yêu cầu, parse timezone, giữ event/transaction ID để truy vết, join ground truth.
- [ ] Chốt hai mốc cắt theo kỳ mô phỏng 50/30/20 và quy tắc khoảng `[start, end)`. Không cắt cùng timestamp sang nhiều tập.
- [ ] Xuất split và metadata: mốc thời gian, count/label balance/phân bố từng tập, seed/schema/nguồn dataset.
- [ ] Không tự resample validation/test để làm đẹp phân bố; báo cáo nếu thiếu positive/negative và điều chỉnh quy mô generator có chủ đích.
- [ ] Sửa root config và nối CLI vào loader/split thực, giữ runner mỏng.

Đầu ra: train/validation/test có thể tái lập, không overlap transaction, sắp đúng thời gian. Hoàn thành khi mọi dòng có nguồn event và label rõ ràng, cả ba tập có thống kê và đủ dữ liệu để đánh giá.

### Phase 3 — Feature và graph theo thời gian

- [ ] Feature giao dịch: amount/log amount, fee, type, channel, calendar/time.
- [ ] Feature lịch sử: count 1h/24h, mean amount 7 ngày, thời gian từ giao dịch trước, người nhận mới; xác định rõ request/completed nào góp vào từng feature.
- [ ] Feature tại t chỉ dùng thông tin trước t; với event cùng timestamp, dùng thứ tự lifecycle được xác định. Không dùng kết quả giao dịch đang chấm.
- [ ] Fit preprocessing trên train; xử lý category/node chưa gặp và missing thống nhất.
- [ ] Mapping node/type; graph lịch sử, node features, target edges, edge features, labels và masks.
- [ ] Chốt chiều edge, CASH_IN/cash counterparty, self-loop, cửa sổ graph và chiến lược snapshot/batching.
- [ ] Cho phép lịch sử đã quan sát trong validation/test cập nhật context theo replay; không dùng label để cập nhật representation hoặc gradient.

Đầu ra: graph/tensor dataset cùng feature schema và preprocessing artifact. Hoàn thành khi shape/ID alignment đúng và kiểm tra thêm dữ liệu tương lai không làm đổi feature/prediction context của quá khứ.

### Phase 4 — Train cơ bản

- [ ] Vòng train PyTorch, binary loss trên logits, optimizer, device và seed.
- [ ] Xử lý imbalance chỉ dựa trên train; cấu hình hyperparameter và batch/snapshot strategy.
- [ ] Validation mỗi epoch, early stopping, checkpoint tốt nhất và log metrics.
- [ ] Nối stage thành một lệnh từ dataset đến checkpoint; không dùng vòng lặp đạt mục tiêu giả.

Đầu ra: run directory có config, dataset/split metadata, learning curves và checkpoint. Hoàn thành khi train thật chạy được, loss/gradient hợp lệ và checkpoint load lại được.

### Phase 5 — Đánh giá, baseline và tuning

- [ ] Baseline amount/rule và model tabular với cùng split, feature thời điểm và ngân sách đánh giá.
- [ ] Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix và phân tích theo scenario/type/thời gian.
- [ ] Chọn threshold trên validation; tuning xong mới chạy đánh giá test cuối.
- [ ] Báo cáo chi phí inference và kiểm tra lợi ích graph so với baseline.

Đầu ra: báo cáo so sánh, model config và threshold được chọn. Hoàn thành khi kết quả truy vết được và nêu rõ giới hạn nhãn tổng hợp; không coi đạt F1/AUC cố định là đủ để tích hợp.

### Phase 6 — Đóng gói và inference offline

- [ ] Bundle weights, model config, feature schema, preprocessing, threshold, phiên bản/metadata dataset và split.
- [ ] Load/predict dùng cùng code feature và thứ tự replay như training.
- [ ] Quản lý history/graph state, cold start, unknown account/category và checkpoint replay.
- [ ] Replay stream offline, đối chiếu kết quả với evaluation và kiểm tra load/restart.

Đầu ra: artifact có version và runner inference offline. Hoàn thành khi cùng input/context cho kết quả nhất quán và replay không đếm trùng lifecycle events.

### Phase 7 — Tích hợp Anomaly

- [ ] Adapter canonical banking record → input model, mapping account ID và event contract.
- [ ] Chốt thời điểm gọi detector, đồng bộ context với EventStore và cách lưu/hiển thị risk score.
- [ ] Thiết kế retry/idempotency, resume checkpoint, lỗi inference, latency và version model.
- [ ] Kiểm chứng event thực → feature/graph → inference → kết quả trong Anomaly.

Đầu ra: luồng tích hợp chạy được và evidence end-to-end. Test local hoặc xuất event tương thích chưa thay thế kiểm chứng EventStore/consumer/app thực.

## 5. Công việc tiếp theo và quy tắc bàn giao

**Task tiếp theo: Phase 2 — loader event yêu cầu, join label và xuất split theo thời gian.** Chưa làm feature graph/train/tuning/realtime trong task này. Đọc source generator, event contracts và manifest hiện tại trước khi triển khai; không chỉ dựa vào con số trong tài liệu.

Khi bàn giao phase, cập nhật checkbox, file/command đã thay đổi, lệnh kiểm tra và kết quả, giới hạn chưa kiểm chứng, cùng task tiếp theo. Chỉ đánh dấu hoàn thành khi đầu ra thực có thể chạy/kiểm tra; không coi scaffold hay thông báo stage là pipeline đã hoạt động.

Giữ nguyên contract Anomaly nếu chưa có yêu cầu đổi; adapter xử lý khác biệt schema. Không sửa các repo ngoài phạm vi phase hoặc thêm hạ tầng Kafka/realtime chỉ vì scaffold hiện có.


## 6. Bàn giao Phase 1 (2026-10-09)

Đã triển khai trong GATv2, không sửa generator/Anomaly ở phase này:

- `scripts/validate_dataset.py`: CLI đọc dataset hiện có, xuất report và exit code theo lỗi.
- `src/fraud_pipeline/validation/`: đọc JSONL/provenance, event contracts, cross-file checks, ledger replay, distributions và feature policy.
- `tests/test_dataset_validation.py`: 14 test với ledger viết tay độc lập, chủ động làm sai dữ liệu để kiểm tra khả năng phát hiện lỗi.
- `reports/dataset-validation/validation.json` và `.md`: artifact local, bị gitignore, tái tạo bằng CLI.

Lệnh kiểm chứng:

```bash
python3 -m unittest discover -s tests -p 'test_dataset_validation.py' -v
python3 scripts/validate_dataset.py
```

Kết quả dataset hiện có: 2.500 transactions, 40 anomaly; 5.544 events. Replay toàn bộ giao dịch không có lỗi contract/ledger/label. Validator trả `passed_with_warnings`, 0 errors, 105 warning occurrences (100 loan submissions chứa kết quả duyệt, 1 loan ledger exclusion, 1 final-state loan snapshot, 1 immediate-settlement limitation, 2 nghiệp vụ ít anomaly). Warnings được nhóm trong Markdown; JSON giữ từng vị trí cụ thể. Đây không phải backend replay.

Preview theo ngày nguyên của kỳ 181 ngày: train 90 ngày, validation 54 ngày, test 37 ngày. Mốc cắt `[start, end)` ở timezone +07:00:

| Tập | Khoảng thời gian | Giao dịch | Normal | Anomaly |
|---|---|---:|---:|---:|
| Train | 2025-01-01 → trước 2025-04-01 | 1.275 | 1.259 | 16 |
| Validation | 2025-04-01 → trước 2025-05-25 | 758 | 743 | 15 |
| Test | 2025-05-25 → trước 2025-07-01 | 467 | 458 | 9 |

Phase 2 dùng `floor(days × 0.5)` và `floor(days × 0.8)` làm hai mốc ngày, tại local midnight; không chia ngày/giao dịch ngẫu nhiên. Số lượng giao dịch khác tỷ lệ thời gian là bình thường. Preview chưa ghi dữ liệu splits.

Graph quan sát có 113 node, 1.372 cặp source/destination, 635 cặp lặp lại. Amount range normal/anomaly chồng lấn. DEBIT có 1 anomaly, CASH_IN có 4; các metric theo từng loại hoặc trên test nhỏ cần nêu giới hạn, không tự resample validation/test.

Quyết định feature ban đầu: dùng trường request và lịch sử causal, ID chỉ để join/mapping. Credit histories hiện không vượt thời điểm snapshot; vẫn loại credit cho đến khi có as-of lookup. Loan fields, identity và mọi trường kết quả xử lý bị loại. Các giới hạn loan/OTP/delay được chấp nhận bằng cách giới hạn feature và ghi rõ; không tuyên bố đã sửa toàn bộ mô phỏng loan.


## 7. Tăng quy mô dataset lên 500k (2026-10-10)

Theo yêu cầu người dùng, cấu hình mới là 1.000 khách hàng × 500 giao dịch = 500.000 giao dịch trong 181 ngày, giữ seed 20261002 và anomaly probability 2%. YAML, Python defaults và README của generator đã đồng bộ. Dataset cũ giữ ở `../anomaly-data-pipeline/data/generated-before-500k`; dataset mới ở `../anomaly-data-pipeline/data/generated`. Các số trong phần bàn giao Phase 1 phía trên là snapshot lịch sử 2.500 giao dịch.

Manifest mới có 1.005.469 domain events, 350.440 native transfer events (export của cùng transfer subset), 1.000 account và 13 counterparty. Validator chạy trên dataset mới: **0 errors**, 1.003 warning occurrences gồm 1.000 loan submission outcomes, loan ledger exclusion, loan final state và immediate settlement. Không còn cảnh báo ít anomaly theo từng nghiệp vụ. Tổng anomaly **9.975** (1,995%); đây là xác suất sinh nên không buộc đúng 10.000.

| Tập preview | Giao dịch | Normal | Anomaly |
|---|---:|---:|---:|
| Train | 251.610 | 246.635 | 4.975 |
| Validation | 148.363 | 145.408 | 2.955 |
| Test | 100.027 | 97.982 | 2.045 |

Mốc cắt theo ngày không đổi. Đây vẫn là preview; Phase 2 chưa tạo split artifacts. Graph có 1.013 node quan sát, 52.497 cặp source/destination và 447.503 giao dịch lặp cặp sau lần đầu. Validator đọc JSONL và hash từng dòng, tái sử dụng tên field để giảm RAM khi kiểm tra bộ lớn; các bảng parsed vẫn giữ trong RAM. Report/dashboard của generator và validation report đã được cập nhật. Các giới hạn credit/loan/OTP/delay và feature policy vẫn giữ nguyên.

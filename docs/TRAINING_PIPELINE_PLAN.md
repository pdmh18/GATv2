# Kế hoạch pipeline train GATv2 phục vụ Anomaly

Cập nhật: 2026-10-10. Tài liệu bàn giao context và theo dõi các phase; các mục chưa đánh dấu vẫn là công việc dự kiến.

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

Ưu tiên hoàn thành từng phase và kiểm chứng đầu ra trước khi chuyển tiếp. Phase 4 đã train thật trên seed chính, 3 epoch CPU, checkpoint load lại khớp dự đoán. Phase 5 đã hoàn thành so sánh, freeze threshold trên validation và đánh giá test; xem mục 18. Ưu tiên hiện tại: feature EDA và kiểm chứng rút gọn input GATv2 trước Phase 6.

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
- Đã kiểm tra lại ở Phase 2: `ROOT = Path(__file__).resolve().parents[2]` vốn trỏ đúng root GATv2; nhận định root thành `src/` trước đó không đúng. Giữ root, thêm JSON config và bỏ việc load training config khi import pipeline.

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

### Phase 2 — Loader và chia thời gian (hoàn thành)

- [x] Loader JSONL trong GATv2; đường dẫn dữ liệu cấu hình được, mặc định nguồn là repo cùng cấp.
- [x] Trích event yêu cầu, parse timezone, giữ event/transaction ID để truy vết, join ground truth.
- [x] Chốt hai mốc cắt theo kỳ mô phỏng 50/30/20 và quy tắc khoảng `[start, end)`. Không cắt cùng timestamp sang nhiều tập.
- [x] Xuất split và metadata: mốc thời gian, count/label balance/phân bố từng tập, seed/schema/nguồn dataset.
- [x] Không tự resample validation/test để làm đẹp phân bố; báo cáo nếu thiếu positive/negative và điều chỉnh quy mô generator có chủ đích.
- [x] Kiểm chứng root config đã đúng; hỗ trợ config JSON cho dataset và nối CLI vào loader/split thực, giữ runner mỏng.

Đầu ra: train/validation/test có thể tái lập, không overlap transaction, sắp đúng thời gian. Hoàn thành khi mọi dòng có nguồn event và label rõ ràng, cả ba tập có thống kê và đủ dữ liệu để đánh giá.

### Phase 3 — Feature và graph theo thời gian (revision hoàn tất ở seed chính)

- [x] Feature giao dịch: amount/log amount, fee, type, channel, calendar/time.
- [x] Feature lịch sử: count 1h/24h, median amount 7 ngày, thời gian từ giao dịch trước, người nhận mới; xác định rõ request/completed nào góp vào từng feature.
- [x] Feature tại t chỉ dùng thông tin trước t; với event cùng timestamp, dùng thứ tự lifecycle được xác định. Không dùng kết quả giao dịch đang chấm.
- [x] Fit preprocessing trên train; xử lý category/node chưa gặp và missing thống nhất.
- [x] Mapping node/type; graph lịch sử, node features, target edges, edge features, labels và masks.
- [x] Chốt chiều edge, CASH_IN/cash counterparty, self-loop, cửa sổ graph và chiến lược snapshot/batching.
- [x] Cho phép lịch sử đã quan sát trong validation/test cập nhật context theo replay; không dùng label để cập nhật representation hoặc gradient.

- [x] Đối chiếu độ phủ 1/7/30 ngày, mean/median và graph context với dataset thực của run (mục 15).
- [x] Chốt/triển khai median cùng account+type, amount-relative và support/cold-start flags từ kết quả audit.
- [x] Tách window graph message passing khỏi window short-term burst/fan-in/forwarding.
- [x] Bổ sung node features causal khác nhau giữa account; derive model dimensions từ schema.
- [x] Xuất lại tensors, kiểm tra no-future/label invariance và Torch/PyG shape compatibility sau sửa.

Đầu ra: graph/tensor dataset cùng feature schema và preprocessing artifact. Hoàn thành khi shape/ID alignment đúng và kiểm tra thêm dữ liệu tương lai không làm đổi feature/prediction context của quá khứ.

### Phase 4 — Train cơ bản

- [x] Vòng train PyTorch, binary loss trên logits, optimizer, device và seed.
- [x] Xử lý imbalance chỉ dựa trên train; cấu hình hyperparameter và batch/snapshot strategy.
- [x] Validation mỗi epoch, early stopping, checkpoint tốt nhất và log metrics.
- [x] Nối stage thành một lệnh từ dataset đến checkpoint; không dùng vòng lặp đạt mục tiêu giả.

Đầu ra: run directory có config, dataset/split metadata, learning curves và checkpoint. Hoàn thành khi train thật chạy được, loss/gradient hợp lệ và checkpoint load lại được.

### Phase 5 — Đánh giá, baseline và tuning

- [x] Baseline amount/rule và model tabular với cùng split, feature thời điểm và ngân sách đánh giá.
- [x] Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix và phân tích theo scenario/type/thời gian.
- [x] Chọn threshold trên validation; tuning xong mới chạy đánh giá test cuối.
- [x] Báo cáo chi phí inference và kiểm tra lợi ích graph so với baseline.

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

**Task hiện tại: feature EDA cho GATv2 trước khi rút gọn feature và sang Phase 6.** Phạm vi hiện tại chỉ GATv2; không train thêm model baseline. Correlation/biến thiên trên train; độ nhạy trên validation; không chọn feature bằng test.

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


## 8. Bàn giao Phase 2 (2026-10-10)

Đã xuất dataset request trên toàn bộ 500k giao dịch, cả 5 nghiệp vụ, không sửa generator/Anomaly:

- `configs/dataset.json`: nguồn, report validator, output và split ratios cấu hình được. JSON giúp chạy Phase 2 bằng standard library.
- `src/fraud_pipeline/ingestion/request_loader.py`: kiểm tra provenance SHA256, stream request events, join label một-một và giữ event/transaction ID + source line.
- `src/fraud_pipeline/split_export.py`: tính interval theo ngày nguyên, stream export, thống kê/checksum/manifest, kiểm tra lại nguồn trước publish và preserve previous output khi overwrite.
- `src/fraud_pipeline/config.py`, `pipeline.py`, `scripts/run_pipeline.py`: config theo repo root, import không kéo dependency train; stage split thực. Training config YAML và helper dataframe cũ vẫn riêng; `--stage all` chưa được phép chạy scaffold như pipeline end-to-end.
- `tests/test_request_splits.py`: kiểm tra boundaries, join, provenance cũ/sai dataset, nguồn thay đổi giữa export, output reproducibility, overwrite/rollback boundary và CLI chạy từ ngoài repo.

Lệnh:

```bash
python3 scripts/run_pipeline.py --stage split
python3 -m unittest discover -s tests -p 'test_request_splits.py' -v
python3 -m unittest discover -s tests -p 'test_dataset_validation.py' -v
```

Output local ở `data/splits/dataset-v2/`: `train.jsonl`, `validation.jsonl`, `test.jsonl`, `manifest.json`, `summary.md`. Bị gitignore; có thể tái tạo từ source dataset đã validate. Mỗi request xuất đúng một lần, event terminal không xuất thành training sample. Nhãn là target riêng trong sample; trace IDs và event type/line là metadata, không phải feature tự động.

| Tập xuất thật | Giao dịch | Normal | Anomaly |
|---|---:|---:|---:|
| Train | 251.610 | 246.635 | 4.975 |
| Validation | 148.363 | 145.408 | 2.955 |
| Test | 100.027 | 97.982 | 2.045 |

Giữ nguyên mốc ngày và phân bố nguồn, không resample. Các tập đều có hai class. Kiểm chứng gồm 12 test Phase 2 và 14 test validator, cùng scan trực tiếp output 500k để đối chiếu count, checksum, timestamp interval/order và ID không trùng; mẫu đầu/cuối mỗi tập được truy về event nguồn bằng source line. Chưa tạo feature/graph hoặc train.

Phase 3 cần đọc full chronological source events để xây context (cả request và terminal). Không chỉ ghép ba sample files để suy ra settled history; không dùng label hoặc kết quả của giao dịch hiện tại khi chấm request. Giữ policy credit/loan exclusion và cash-flow direction từ Phase 1.


## 9. Đánh giá tính hợp lý dataset trước Phase 3 (2026-10-10)

Theo yêu cầu người dùng, đã phân tích dataset 500k, đối chiếu SHA256 với report Phase 1 và split manifest. Evidence local: `reports/dataset-quality/assessment.json`. Chưa sửa generator, chưa train GATv2 và chưa kiểm chứng nhiều seed.

Phân bố trên cùng seed khá ổn định: anomaly rate train/validation/test là 1,977% / 1,992% / 2,044%. Total variation của type so với train dưới 0,23%, của channel dưới 0,18%; mean amount lệch dưới 0,5%. Nhãn theo tháng khoảng 1,959–2,054%. Đây là ổn định theo thời gian của một seed, không phải bằng chứng giống ngân hàng thật hoặc ổn định qua nhiều seed.

Các vấn đề cần chỉnh trước khi dùng dataset để kết luận hiệu quả GATv2:

1. **Nhãn mơ hồ:** anomaly và normal-large đều dùng multiplier [4,12] của cùng account typical amount; normal-large probability 5% trên normal, anomaly probability 2%. Chưa có ngữ cảnh khác để phân biệt hai nhóm. Đây là random mixture overlap, không phải hai hành vi có khác biệt quan sát được.
2. **Cạn tiền theo thời gian:** failure rate tăng 11,49% tháng 1 → 46,48% tháng 6, tổng failed 35,75%. Ledger hợp lệ nhưng CASH_IN 12% và các dòng tiền ra tạo mô phỏng thiếu nguồn tiền bù; loan cash flows chưa được nối. Nhãn/amount ổn định không có nghĩa trạng thái tài chính ổn định.
3. **Hành vi quá đồng nhất:** mỗi account đúng 500 giao dịch, giờ uniform (20.622–21.040 giao dịch/giờ), channel uniform. Chưa có profile tần suất, khung giờ và channel theo khách hàng.
4. **Graph chưa gắn với nhãn:** người nhận/merchant được chọn độc lập với anomaly, chưa có scenario quan hệ như chuỗi chuyển tiền, vòng giao dịch, receiver mới kết hợp amount/velocity. Chưa có cơ sở kỳ vọng graph vượt baseline dùng cùng lịch sử account.

Diagnostic baseline không train model: amount / median các request trước của source account, tối thiểu 20 request lịch sử; threshold 2,1 chọn bằng F1 validation. Khi xét validation/test, chỉ cập nhật lịch sử bằng request đã xảy ra, không dùng label để tạo score. Test: precision 29,75%, recall 100%, F1 0,459, ROC-AUC 0,975; TP 2.045 và FP 4.829. AUC cao không đồng nghĩa dataset cho phép precision/F1 cao. Với xác suất hiện tại, nhóm high-amount lý tưởng có precision kỳ vọng 0,02 / (0,02 + 0,98×0,05) ≈ 29%; đây là suy luận theo generator, không phải giới hạn chứng minh cho mọi mô hình trên một mẫu hữu hạn.

Đề xuất ưu tiên: gắn anomaly với ngữ cảnh quan sát được và có graph/behavior scenarios; nguồn tiền/tần suất theo profile khách hàng để tránh failure drift không chủ đích; kiểm tra nhiều seed và baseline sau khi chỉnh generator. Khi dataset đổi, chạy lại Phase 1 và Phase 2 trước Phase 3. Dataset hiện có vẫn dùng được để kiểm tra kỹ thuật và baseline amount-outlier, nhưng chưa nên dùng làm bằng chứng nghiên cứu tổng quát cho GATv2.


## 10. Contextual generator và kiểm chứng nhiều seed (2026-10-10)

Đã thực hiện yêu cầu chỉnh nguồn dữ liệu trước Phase 3. Các phần lịch sử phía trên mô tả bộ amount-only v2 cũ, không phải trạng thái dataset mặc định hiện tại.

Thay đổi trong `anomaly-data-pipeline`:

- `generation/behavior.py`: phân bổ tổng budget theo activity weights, typical amount theo thu nhập/tần suất, giờ/channel/người nhận quen và payday. Config nằm trong TOML.
- `generation/scenarios.py`: income CASH_IN định kỳ trong tổng budget và lịch scenario có warmup causal. `velocity_burst` positive sau hai request trước; `graph_fan_in` positive sau ba nguồn trước; `rapid_forwarding` sau fan-in. `contextual_amount` kết hợp amount, giờ/channel khác profile và ưu tiên đối tác chưa gặp.
- Stage transaction giữ ledger thật và contract request/result. Không cộng số dư ngầm. Public native transfer contract không đổi.
- `behavior_profiles.jsonl` là metadata audit, bị loại khỏi feature/baseline để tránh oracle preferences/typical amount. Model phải suy ra preferences từ lịch sử.
- Manifest vẫn dataset schema 2 (native transfer schema 1), thêm `scenario_profile=contextual-v3` và snapshot config. `transactions_per_customer` giờ là ngân sách trung bình: tổng vẫn customers × giá trị này, từng account khác nhau.
- Generator có test reproducibility, ledger/native export, profile heterogeneity và causal scenario witnesses.

Thay đổi trong GATv2:

- `validation/scenarios.py`: kiểm tra profile join/schema và prior witnesses theo scenario; Phase 1 được chạy lại trên từng seed.
- `evaluation/dataset_baseline.py`, `scripts/assess_dataset.py`: diagnostic rules amount-only, behavior và behavior+graph. Không đọc generation profiles hoặc labels/scenario names để tạo score; score trước current request, carry history qua các tập theo thời gian, threshold chọn bằng validation F1.
- Report có ROC-AUC/average precision, confusion matrix và test recall từng scenario; scenario names chỉ dùng phân tích kết quả.

Đã sinh, validate, split và baseline **3 seed × 500.000 giao dịch**, seeds 20261002, 20261003, 20261004. Tất cả có 0 lỗi contract/ledger/label/scenario-witness; warnings loan/instant settlement giữ nguyên. Tỷ lệ anomaly là budget mục tiêu 2%, không còn coin-flip xác suất trên mỗi giao dịch: mỗi run này có 10.000 positive, gồm contextual_amount 4.000, velocity_burst 3.000, graph_fan_in 2.400 và rapid_forwarding 600.

| Seed | Test F1 amount-only | Test F1 behavior | Test F1 behavior+graph | Failed toàn kỳ |
|---|---:|---:|---:|---:|
| 20261002 | 0,251 | 0,729 | 0,844 | 0,0836% |
| 20261003 | 0,263 | 0,733 | 0,851 | 0,0690% |
| 20261004 | 0,261 | 0,734 | 0,853 | 0,0702% |

Control amount-only cũ được chạy bằng cùng implementation: test F1 amount-only 0,455; behavior/behavior+graph khoảng 0,115. Tín hiệu mới có ích cho rule ngoài amount, không chứng minh lợi ích kiến trúc GATv2. Rules theo giả định scenario nên chưa phải benchmark độc lập hoặc bằng chứng real-bank generalization.

Artifacts tái tạo:

- Dataset các seed: `data/raw/contextual-v3/seed-<seed>`.
- Split các seed: `data/splits/contextual-v3/seed-<seed>`.
- Audit/baseline: `reports/contextual-v3/seed-<seed>/validation.json`, `assessment.json`.
- Tổng hợp: `reports/contextual-v3/multiseed.json`, `comparison.json`; control cũ ở `reports/contextual-v3/old-baseline/assessment.json`.
- Dataset mặc định `../anomaly-data-pipeline/data/generated` đã cập nhật bằng bản byte-identical của seed 20261002 sau kiểm tra SHA256. Audit root được retarget tới vị trí mới của cùng dữ liệu, không giả định dataset khác tương đương. Report/dashboard generator đã cập nhật.
- Bộ 500k amount-only giữ ở `../anomaly-data-pipeline/data/generated-amount-only-v2`; split cũ giữ ở `data/splits/dataset-v2.previous-<id>`. SHA/provenance của artifact cũ cần trỏ vào backup khi muốn replay; không dùng dataset mặc định mới cho artifact cũ.

Split mặc định mới đã xuất lại:

| Tập | Giao dịch | Normal | Anomaly |
|---|---:|---:|---:|
| Train | 251.815 | 246.705 | 5.110 |
| Validation | 148.467 | 145.551 | 2.916 |
| Test | 99.718 | 97.744 | 1.974 |

Lệnh baseline mặc định sau split:

```bash
python3 scripts/run_pipeline.py --stage split --overwrite
python3 scripts/assess_dataset.py
```

Kiểm tra hồi quy: 33 unittest GATv2 liên quan (14 validator, 12 loader/split, 3 causal-rule/metric, 4 contextual audit) và 3 unittest generator đều pass; `git diff --check` cả hai repo pass. Chưa train/inference GATv2 hoặc chạy consumer Anomaly thật.

Giới hạn thực tế của phiên bản mới: funding rất dư thanh khoản nên failure gần như chỉ ở đầu kỳ, chưa được hiệu chỉnh bằng số liệu ngân hàng; loan cash flows/OTP resend/expiry/processing delay vẫn ngoài phạm vi. Ở seed chính, 230/4.000 contextual_amount có dưới 20 prior requests: cold-start preferences chưa đủ để rule/profile estimator suy ra chắc chắn. Phase 3 cần history_count/cold-start handling; không đưa oracle profile vào input để đạt metrics cao.

**Task tiếp theo: Phase 3**, với request-history window 120s, transfer-neighborhood window 600s và các window dài đã dự kiến, đều cấu hình được. Chốt source/destination cash-flow roles và raw target-edge roles; xây feature/graph theo thời gian với policy label/profile/terminal exclusion đã có. Dataset mới đã kiểm chứng ở ba seed nhưng vẫn là mô phỏng theo scenario có kiểm soát.

Runner tái tạo multi-seed đã lưu tại `scripts/check_dataset_seeds.py`; đọc config generator và hỗ trợ seed/data/split/report roots qua CLI. Logic là orchestration generation/validation/export/assessment đã chạy trên ba seed; mặc định từ chối output đã tồn tại. README có lệnh dùng generator Python environment để tái tạo ở root mới.

## 11. EDA bằng Matplotlib (2026-10-10)

Đã chạy `scripts/plot_dataset.py` trên dataset mặc định contextual-v3 seed 20261002, kiểm tra hash nguồn khớp audit: 500.000 requests, 10.000 anomaly (2%). Xuất năm PNG và PDF năm trang ở `reports/eda/`, cùng `eda_summary.json`; đã xem trực tiếp tất cả hình. Lượt này không chỉnh generator.

- Amount hai class chồng lấn: median normal 314.957, anomaly 1.006.978,5; p99 normal 29 triệu cao hơn anomaly 13,41 triệu. Income CASH_IN tạo đuôi normal lớn; không đánh giá anomaly chỉ bằng amount tuyệt đối.
- Artifact giờ: không có anomaly trong các giờ 14, 15, 16 (+07:00), trong khi có tổng 120.763 normal. Đây là giao các giờ bình thường của profile generator; model có thể học shortcut theo giờ. Cần sinh cả anomaly trong giờ quen thuộc và normal ngoài giờ, kiểm tra lại bằng EDA trước khi diễn giải metrics.
- Failure tháng 1 khoảng 0,447%, tháng 2 khoảng 0,006%, từ tháng 3 đến tháng 6 bằng 0. Funding dư thanh khoản làm failure biến mất sau đầu kỳ; ổn định qua seed không đồng nghĩa hợp lý với dữ liệu ngân hàng thực.
- Activity khác nhau theo account (86–1.497 requests, median 412,5), nhưng 27 account cùng đạt trần 1.497: clipping profile vẫn để lại điểm tụ ở biên.
- Graph money-flow request có 45.812 cặp có hướng; CASH_IN đảo thành cash → account. Có cả failed requests, không phải settled-transfer graph và không so trực tiếp với số cặp raw source/destination trong audit.
- Baseline F1 amount khoảng 0,251–0,263; behavior 0,729–0,734; thêm graph 0,844–0,853 qua ba seed. Đây là rules theo scenario tổng hợp, chưa train GATv2.

Ưu tiên dữ liệu trước khi chốt Phase 3 benchmark: xử lý shortcut giờ và đánh giá lại funding/failure; giữ cold-start handling đã nêu ở mục trước. Tính hợp lý thực tế cần số liệu đối chiếu, không suy ra từ validator pass hay F1 ổn định.

## 12. Sửa shortcut giờ — contextual-v4 (2026-10-10)

Đã thay việc ép mọi anomaly ra ngoài preferred hours bằng cấu hình `anomaly_off_hours_probability=0.35`: 65% scenario starts trong giờ quen thuộc, 35% ngoài giờ. Normal `off_hours_probability=0.05` chọn đúng phần bù preferred hours, thay vì chọn lại toàn bộ 24 giờ. Burst/graph vẫn giữ prior witnesses và khoảng thời gian; graph start dựa trên hub nên preferred hours của donor có thể khác. Không thay event payload/schema hoặc dùng profile làm model input.

Manifest ghi `scenario_profile=contextual-v4`. Validator v4 kiểm tra contextual amount nằm trong interval theo typical amount và channel khác preference; không bắt buộc lệch giờ. Validator v3 vẫn giữ quy tắc cũ. Kiểm tra config thiếu/hỏng báo lỗi audit, không crash. Đây là kiểm tra cách sinh scenario, không chứng minh preferences đã quan sát đủ qua lịch sử.

Đã chạy ba seed, mỗi seed 500.000 requests và 10.000 anomaly; cả ba pass validator với 0 lỗi, xuất split thời gian và diagnostic baseline. Kết quả audit giờ:

| Seed | Anomaly giờ 14 | Giờ 15 | Giờ 16 | Giờ không có anomaly |
|---|---:|---:|---:|---|
| 20261002 | 552 | 585 | 479 | Không có |
| 20261003 | 585 | 534 | 492 | Không có |
| 20261004 | 520 | 484 | 553 | Không có |

Normal cũng có mặt đủ 24 giờ ở cả ba seed. Seed chính contextual_amount có 2.582 mẫu trong preferred hours, 1.418 ngoài giờ. Đã loại zero-support shortcut 14–16, nhưng tỷ lệ theo giờ vẫn khác giữa hai class; không kết luận loại hết shortcut hoặc đã hiệu chỉnh theo ngân hàng thực.

Diagnostic F1 test: amount 0,259 ở cả ba seed; behavior 0,695 / 0,695 / 0,705; behavior+graph 0,831 / 0,845 / 0,830. Đây là rule metrics, không phải chỉ số chất lượng dataset hay GATv2. Giảm behavior F1 sau thay đổi phù hợp việc mất một dấu hiệu giờ quá dễ; cần benchmark độc lập để đánh giá model.

Artifacts: `data/raw/contextual-v4/seed-<seed>`, `data/splits/contextual-v4/seed-<seed>`, `reports/contextual-v4/seed-<seed>`, tổng hợp `multiseed.json` và `hour-distribution.json`. Script `check_hour_distribution.py` xác minh source hash rồi đếm giờ/source-preference từng scenario. EDA mới ở `reports/eda-contextual-v4/`; EDA v3 trước đó vẫn ở `reports/eda/`.

Dataset mặc định sibling `data/generated` đã cập nhật từ seed 20261002 sau kiểm tra SHA256 trước/sau copy; v3 giữ ở `data/generated-contextual-v3`. Audit root và báo cáo/dashboard generator đã cập nhật. Split mặc định xuất lại: train 251.481 (5.002 anomaly), validation 148.653 (2.991), test 99.866 (2.007); split trước giữ bằng backup tự động. Provenance artifact v3 trỏ đường dẫn cũ phải retarget đúng backup và xác minh hash trước replay.

Kiểm tra: 36 unittest GATv2 liên quan và 3 unittest generator pass; diff whitespace checks pass. Không train GATv2. **Ưu tiên dữ liệu tiếp theo: funding/failure**, sau đó clipping activity. Hai giới hạn này chưa được sửa trong v4; không chốt benchmark thực tế chỉ từ kết quả overlap giờ.

## 13. Chuẩn hóa output (2026-10-10)

Đường dẫn hiện hành thay thế các đường dẫn output ghi trong lịch sử mục 1–12: `outputs/<version>/seed-<seed>/{raw,splits,validation,baseline,eda}`. Tổng hợp `multiseed.json`, `hour-distribution.json`, `comparison.json` ở `outputs/<version>/`. Phiên bản hiện hành contextual-v4, seed chính 20261002. `configs/dataset.json` và module `artifacts.py` là nguồn defaults chung cho validator, split, baseline, EDA và runner.

Đã di chuyển artifact v3/v4, tách baseline khỏi validation; backup/trial/duplicate default export ở `outputs/archive/`. Snapshot raw theo seed trong GATv2 là source của run; không phụ thuộc `data/generated` mutable của generator. Source JSONL và split JSONL không đổi; retarget audit dataset directory, split source directory và validation checksum sau di chuyển. Metadata trước sửa lưu tại `outputs/archive/migration-originals/`; source và split checksum đã được kiểm tra. Một split amount-only lịch sử thiếu audit gốc giữ checksum validation cũ và ghi rõ migration note; cần audit mới khi tái sử dụng.

Runner mới nhận `--output-root` thay ba roots rời nhau; ví dụ `outputs/contextual-v4-rerun`. CLI của validator/EDA/baseline vẫn có override source/report/output; đường dẫn defaults tính từ project root. README có cây output và lệnh mới. `outputs/` bị gitignore; không lưu dataset vào git.

Histogram request mỗi cặp được chỉnh trục Y sang log và bổ sung panel chỉ chứa cặp có ≥2 requests để thấy phần đuôi. Trục X vẫn log10(request count), trục Y là số cặp; cột singleton cao không phải model score thấp hoặc thiếu giao dịch. EDA hiện hành ở `outputs/contextual-v4/seed-20261002/eda/`.

## 14. Bàn giao Phase 3 (2026-10-10)

Đã triển khai `features/causal.py`, `features/export.py`, `features/dataset.py`, stage `features` và `scripts/check_feature_artifacts.py`. Không dùng behavioral dataframe `groupby` toàn kỳ; các scaffold cũ không tham gia luồng này. Window/policy/output nằm trong JSON config, không hardcode nghiệp vụ trong runner.

Availability: request tại t chỉ dùng timestamp < t. Mọi request cùng timestamp dùng pre-group history, không phụ thuộc terminal serialization order. Quan sát request và terminal sau chấm cả nhóm; lifecycle cùng timestamp có hiệu lực cho request ở timestamp sau. Đây là policy bảo thủ tránh current/future settlement. Hour/weekday theo timezone của simulation manifest.

34 cột: log amount/fee; sin/cos giờ và weekday, weekend; request counts 120s/1h/24h; tổng lịch sử/time-since-last và has_previous; completed amount mean/count 7 ngày và missing flag; receiver novelty; tỷ lệ channel/hour đã quan sát; transfer-attempt unique incoming sources của raw source/destination trong 600s; incoming settled transfer amount và outgoing transfer-attempt count; cold-start flag; type/channel one-hot contract vocabulary + unknown. Mean completed tính tất cả nghiệp vụ đã hoàn tất của source, không phải số dư hay feature label. Failed requests tính vào request history nhưng không completed mean; graph mode settled loại failed attempts.

Numeric columns có prefix log_ fit mean/std trên train, constant scale=1; categorical/sin/cos/flags giữ nguyên. Missing history dùng zero với explicit flags, không đọc generation profile để điền. Schema/thứ tự cột nằm trong manifest, preprocessing parameters ở `preprocessing.json`; source/input/code/output checksums lưu để truy vết.

Nodes: unknown bucket + account/merchant/cash, static type one-hot 4 chiều, created_at gate availability. Raw target edges giữ source/destination cho classifier; money-flow edges đảo CASH_IN. Graph timeline gồm request attempt và successful terminal cùng type/log amount/time; settled/attempts/both cấu hình được. Context `[t-600s,t)` chọn theo prefix, deduplicate directed pairs và mặc định thêm reverse edges. Không thêm self-loops trong artifact; GATv2Conv hiện có làm việc đó. Graph attributes lưu để audit/extension, kiến trúc GATv2 hiện không đưa chúng vào attention. Đặc trưng lịch sử nằm ở transaction head, node features ban đầu chỉ là type.

Batching: timeline + prefix tránh sao chép toàn bộ graph mỗi request. `TemporalDataset.context(i)` trả input của đúng cutoff, labels/masks riêng. Phase 4 phải giữ context của từng request hoặc dùng snapshot bảo thủ hơn; không dùng snapshot tương lai để batch. Đây là artifact NumPy độc lập Torch; chưa benchmark tốc độ training/sampling.

Đã chạy trên contextual-v4 seed 20261002, 500.000 requests: transaction features [500000,34], node features [1041,4], target edges [2,500000], graph timeline 999.513 events. Mask counts train 251.481, validation 148.653, test 99.866; scaler fit 251.481 train rows. Output `outputs/contextual-v4/seed-20261002/features/` khoảng 213 MiB; first export giữ bằng `.previous-*` sau khi bổ sung metadata.

Kiểm chứng: 11 test Phase 3 pass (future/label invariance, same timestamp/current terminal exclusion, failed request vs completion, window expiry, timezone equivalence, scaler train-only, held-out changes không đổi train contexts, cash direction/graph shape/mask, checksum corruption, refusal overwrite). 36 test validator/split/baseline pass. Artifact audit pass: checksum tất cả file, node bounds, finite values, masks, graph-prefix strict cutoff cho toàn bộ 500k; load 1.001 contexts, số graph edges 0–190 ở samples. Report `outputs/contextual-v4/seed-20261002/validation/features.json`.

Commands trong Python environment có NumPy:

```bash
python3 scripts/run_pipeline.py --stage features
python3 scripts/check_feature_artifacts.py
python3 -m unittest discover -s tests -p 'test_causal_features.py'
```

Output đã có: dùng root khác hoặc `--overwrite` có backup. Phase 3 chỉ chạy seed chính; không tuyên bố tensor exports cả ba seed. Funding/activity vẫn là giới hạn của dataset và node features chỉ mã hóa type. **Task tiếp theo: Phase 4 — trainer thật, Torch conversion, causal batching, loss/optimizer và checkpoint bằng train; validation chọn threshold, test giữ cho đánh giá cuối.** Không dùng `model.fit()` scaffold hoặc graph cuối kỳ.

## 15. Đối chiếu dataset và xét lại Phase 3 (2026-10-10)

Mục này thay kết luận chuyển Phase 4 ở mục 14: kỹ thuật causal export đúng nhưng input chưa phù hợp đủ với dữ liệu/model. User yêu cầu phân tích trước khi sửa; lượt này không đổi generator, feature engine hoặc GATv2. Đã replay 500.000 requests của seed chính, đúng source SHA256/feature artifact. Quyết định dựa trên độ phủ train và contract model; validation/test chỉ mô tả ổn định. Chưa train hay chọn window theo F1/test labels.

Artifacts ở `outputs/contextual-v4/seed-20261002/analysis/feature-suitability/`: `analysis.json`, `diagnostic_arrays.npz`, hai PNG và PDF `feature_suitability.pdf`. Script tái tạo `scripts/analyze_feature_suitability.py`, audit options trong `configs/dataset.json.feature_audit`. Candidate windows amount 1/7/30 ngày và settled graph 600s/1h/24h/7 ngày. Cùng timestamp chấm trước quan sát terminal; history gồm mọi completed transaction đã quan sát, không lọc bằng normal labels.

### Amount/history support

Fraction requests train có ít nhất 5 completed giao dịch cùng account+type:

| Nghiệp vụ | 7 ngày | 30 ngày |
|---|---:|---:|
| TRANSFER | 69,1% | 93,2% |
| CASH_IN | 51,3% | 89,5% |
| CASH_OUT | 25,1% | 78,5% |
| PAYMENT | 65,1% | 92,2% |
| DEBIT | 18,4% | 72,6% |
| Tất cả requests | 54,2% | 88,4% |

History cùng type median count là 1 / 5 / 18 ở window 1 / 7 / 30 ngày. 30 ngày đạt ≥10 mẫu cho 73,8% requests train; ngưỡng 5 chỉ minh họa support, không chứng minh estimate đã chính xác thống kê. Không thể dùng min-history toàn account=20 để thay thế support cùng nghiệp vụ. Các type ít hoạt động cần count/missing/reliability flags và fallback được xác định, không lấp bằng oracle profile hoặc số liệu toàn kỳ.

Trên cùng 136.328 train requests có ≥5 history cùng type trong 7 ngày, mean >2×median ở 23,6% khi gộp nghiệp vụ và 7,6% khi dùng đúng nghiệp vụ. Gộp CASH_IN với spending làm tham chiếu khó diễn giải; mean vẫn có đuôi lớn ngay trong CASH_IN do lịch nhận tiền định kỳ và top-up khác nhau. Median robust hơn với outliers, nhưng tỷ lệ cao không tự chứng minh anomaly: khoản thu lớn hợp lệ vẫn có thể vượt median. `all_history_has_cash_in_fraction` là có bất kỳ CASH_IN, không phải đã nhận salary riêng. Phân tích này không tách salary bằng oracle metadata.

Đề xuất input revision: giữ log amount tuyệt đối và scaler train-only; thay tham chiếu mean gộp type bằng median completed amount của account+type trong 30 ngày, thêm relative amount và support/has-history flags. Có thể giữ 7 ngày như nhánh ngắn nếu đủ support; không xem 7/30 ngày là ngưỡng tối ưu đã được benchmark. Calendar theo tháng là candidate để đánh giá riêng cho CASH_IN định kỳ, không đọc payday/profile của generator.

### Graph và độ phù hợp với GATv2

Settled graph, bidirectional như config hiện tại; tỷ lệ cô lập tại mỗi request train:

| Window | Source không có hàng xóm | Cả hai đầu không có hàng xóm |
|---|---:|---:|
| 600 giây | 93,4% | 65,8% |
| 1 giờ | 71,9% | 24,5% |
| 24 giờ | 4,1% | 0,15% |
| 7 ngày | 0,28% | 0,05% |

Validation/test có cùng hiện tượng: source cô lập khoảng 93,5–93,8% ở 600s, khoảng 3,8–4,0% ở 24h. Nối graph đủ dày không tự bảo đảm học tốt; 7d tăng phạm vi/shared counterparty hubs và chi phí. Candidate 24h cân bằng support hơn cho message passing, cần đo chi phí và ablation ở Phase 4. Giữ 120s request velocity và 600s transfer-neighborhood làm features MLP cho scenario ngắn; không đồng nhất window scalar với graph encoder window.

GATv2 hiện `encode(x, edge_index)` rồi ghép `[h_src,h_dst,transaction_features]` ở MLP; edge attributes bị bỏ trong encoder. Không kỳ vọng amount/type trên graph timeline được attention sử dụng khi code chưa truyền chúng vào GATv2Conv. Node input hiện chỉ one-hot type: mọi account bắt đầu bằng cùng vector. 95,7% source train có vùng 2-hop chỉ một loại node (bao gồm cô lập); riêng TRANSFER có kết nối, 48,1% requests có cả hai vùng 2-hop đồng loại.

Với đầu vào đồng nhất, shared transforms và normalized weighted aggregation trong attention không bảo đảm phân biệt account chỉ từ số hàng xóm; vùng đồng nhất có thể cho embedding giống nhau. Đây là nhận xét kiến trúc, chưa phải kết quả runtime hoặc model đã train. Cần bổ sung node-specific causal incoming/outgoing activity, amount/history support và degree vào x; cập nhật toàn bộ node features tại cutoff đúng, kể cả window decay của node không vừa phát event. Không dùng ID như biến liên tục, learned profile oracle hoặc labels để làm khác node.

Giữ thiết kế encoder+transaction classifier hiện có, derive `in_dim`/`edge_dim` từ artifact schema; input shapes `[V,F_node]`, `[2,E_history]`, `[2,N_target]`, `[N_target,F_tx]`. Re-export/test loader cần bỏ hardcode node dimension=4 khi schema mới thêm behavioral columns. Graph attributes hiện là audit/extension metadata, không yêu cầu sửa attention hoặc đổi kiến trúc trong revision này.

### Feature thưa/constant và việc còn lại

- `log_fee`, unknown type/channel không đổi trên train. Fee không có tín hiệu trong generator hiện tại; có thể loại khỏi active input nhưng giữ raw contract. Unknown buckets vẫn cần để xử lý contract/serving, không coi zero-on-train là lý do bỏ chúng.
- Request-count 120s bằng 0 ở 98,1% train, nhưng nonzero ở 1,26% normal và 31,75% anomaly train. Fan-in destination count nonzero 0,57% normal và 25,33% anomaly. Đây là descriptive diagnostics, không F1 hay bằng chứng real-bank; không loại short-term features chỉ vì sparse.
- Cold-start/window đủ support cần flags theo account+type, không chỉ tổng count. Tất cả history vẫn gồm dữ liệu prior chưa biết nhãn; median không được tính trên tập normal đã lọc.
- Hoàn thành revision: personal median/relative amount → dynamic node inputs → separate message-passing/scalar windows → re-export, no-future/label tests, tensor shape smoke test trên model thật. Chưa được đánh dấu lại Phase 3 hoàn tất chỉ vì tensor chạy được.

Hai test của audit History/Graph pass (duplicates/median, boundary expiry và repeated reverse-pair expiry), script compile và whitespace checks pass; các hình đã xem trực tiếp. Phạm vi số liệu là seed chính, chưa audit support trên các seed còn lại. Dataset funding/activity vẫn chưa được hiệu chỉnh.

## 16. Bàn giao revision Phase 3 — schema v2 (2026-10-10)

User đã đồng ý triển khai sau audit. Không đổi generator hay GATv2 architecture. Config thêm `amount_min_history=5`, `include_fee=false`, amount window 30d, message-passing graph 24h, node history 24h và scalar neighborhood window riêng 600s. Cold-start overall=20 vẫn là một flag khác với reference reliability cùng nghiệp vụ.

Transaction features [500000,35]: thay completed mean gộp type bằng median completed theo source+type; count và has-history cùng scope; thêm log(1+amount/median) và reliability flag. Relative amount=0 khi dưới 5 history, cùng reliability=0; vẫn giữ median/count đã quan sát để không giả vờ đây là reference đủ tin cậy. `log_amount` và scaler train-only giữ nguyên; bỏ fee khỏi active input do constant, không đổi raw/event contract. Unknown type/channel buckets giữ nguyên. Median history gồm prior completed cả normal/anomaly chưa biết nhãn, không dùng oracle profile.

Node inputs tại cutoff [1041,12]: type one-hot 4 + 8 dynamic columns: incoming/outgoing settled counts, unique incoming/outgoing partners, median incoming/outgoing amount và has-incoming/has-outgoing flags. Window 24h, CASH_IN theo money-flow orientation. Reconstruct trực tiếp từ graph timeline `[t-window,t)` nên cả node không vừa phát event cũng expire đúng; không stale snapshot, không current terminal. Static `node_features.npy` [1041,4] chỉ là nền, `TemporalDataset.context(i)` ghép dynamic columns và scale. Dynamic numeric scaler fit 256 evenly spaced train cutoffs × available known nodes; flags/type không scale, unknown bucket không tham gia fit. Params/indices ở `node_preprocessing.json`.

Sparse timeline không đổi 999.513 events; graph context chọn settled 24h, deduplicate và thêm reverse edges. Scalar velocity/fan-in/forwarding vẫn dùng windows ngắn. Labels/masks/input shape alignment không đổi. Schema v1 backups vẫn load được, schema v2 có dynamic context. Source/splits giữ nguyên SHA256; export có provenance/code/checksum và backup bản trước. Analytical report mục 15 giữ như evidence cho input cũ; không diễn giải nó thành kết quả input mới.

Đã export seed chính contextual-v4: 500.000 requests, 35 transaction dimensions, 12 node dimensions, 1.041 nodes. 15 tests causal/tensor pass, 36 validation/split/baseline tests và 2 audit tests pass (53 tổng). Test mới kiểm tra median không bị salary/other type trộn, outlier, low-support ratio flag, short neighborhood không đổi khi graph dài, node vectors khác nhau và inactive node expiry; held-out amount/labels không đổi train feature/node/context. Artifact audit kiểm tra toàn bộ prefix/masks/checksum/finite values, load 1.001 contexts: graph edges 0–7.638, source cô lập khoảng 4,6%; unique account input vectors 1–1.000 tùy context, first cold-start có thể giống nhau.

Đã chạy model thật CPU: Torch 2.14.1+cpu, PyG 2.8.0.post1; GATv2 derive `in_dim=12`, `edge_dim=35`, forward output [1] và finite gradients trên train rows 0 và 251480. Không optimizer update, không model training hoặc F1; report `validation/model-inputs.json`. Dependencies dùng temporary uv environment, chưa khóa training environment cho Phase 4. `check_model_inputs.py` tái tạo smoke check.

Feature/model illustration scripts cập nhật để mô tả median, node histories và dimensions mới. Task tiếp theo Phase 4 vẫn phải benchmark context reconstruction/caching và causal batching; graph 24h đủ support hơn nhưng lớn hơn nhiều. Không dùng shared later snapshot cho request sớm, không tuyên bố GATv2 tốt hơn tabular trước ablation. Funding/activity và rủi ro legitimate CASH_IN lớn vẫn cần đánh giá; median ratio riêng không là luật gắn nhãn anomaly.

## 17. Bàn giao Phase 4 (2026-10-10)

Đã thay `model.fit()` scaffold bằng trainer PyTorch thật; thêm `models/batching.py`, checkpoint registry, stage train và end-to-end `--stage all` cho validated raw → splits → features → checkpoint. Không sửa generator hoặc feature tensors. Pipeline reuse artifact theo source/split/policy/code checksum, từ chối stale hoặc run output đã có. Overwrite giữ backup; không fake loop đạt target metrics.

Config training ở dataset JSON: CPU, seed 42, một thread, 3 epochs, hidden 32/4 heads/2 layers/dropout 0,2; AdamW lr=0,001, wd=0,0001, gradient clip=1; batch max128, snapshot3600s. BCEWithLogitsLoss pos_weight=246479/5002=49,276 chỉ từ train. Early-stopping patience2/min_delta0,0001 theo weighted validation loss; run đủ3 epochs nên chưa trigger early stop. Dimensions derive schema: in_dim12, edge_dim35. Không đổi GATv2 encoder+MLP.

Snapshot/cache policy: context tại request đầu bucket thời gian 1h, dùng graph/node snapshot đó cho microbatches còn lại. Cutoff không muộn hơn request; current/next terminal không lọt vào graph. Transaction features exact tại mỗi request. New node sau cutoff map unknown tới refresh sau. Đây là snapshot bảo thủ, bỏ recent graph changes trong cùng giờ; không tuyên bố tương đương graph exact. Cache identity theo feature manifest/policy, mỗi npz có checksum và không chứa label. Train/validation groups tách nhau, test split bị batching từ chối.

Run `outputs/contextual-v4/seed-20261002/training/gatv2-v1/`: 251.481 train rows, 148.653 validation rows, 3.456 contexts cache, train3128 và val1857 microbatches/epoch. Khoảng179,9s CPU tổng gồm warm-cache và best-checkpoint reload validation; output khoảng403 MiB. Warm epoch1 train76,3s/val35,7s; cached epoch2 train25,7s/val5,8s. Timing trên máy hiện tại, không phải production latency hoặc phép đo exact per-request graph.

| Epoch | Train weighted loss | Validation weighted loss | Validation F1 @0,5 | Validation AP |
|---|---:|---:|---:|---:|
| 1 | 0,19349 | 0,04632 | 0,85640 | 0,97277 |
| 2 | 0,06081 | 0,03732 | 0,89029 | 0,98155 |
| 3 | 0,05373 | 0,03871 | 0,92749 | 0,98466 |

Best checkpoint epoch2 theo loss rule trước khi chạy, không chọn lại bằng F1/AP sau nhìn kết quả. Precision0,80863, recall0,99030, ROC-AUC0,99968; TP2962/FP701/FN29/TN144961. Threshold0,5 chưa tune; weighted-loss sigmoid score chưa calibration. Không so trực tiếp với baseline test F1 ở mục cũ và không suy ra lợi ích graph từ score validation cao.

Output có config/provenance, copy feature schema/preprocessing/mapping, `best.pt`/`last.pt` (model và optimizer state), learning curves, epoch history, best validation predictions và summary. Checkpoint load `weights_only=True`, strict state/schema identity; best checkpoint được load lại và dự đoán validation khớp bản lưu. Source/feature files đã xác minh trước/sau; test không được evaluate.

Kiểm tra: 4 training tests pass (first-cutoff/microbatch reuse, unknown new node, rejection test split, cache identity/corruption, real optimizer/checkpoint round-trip, fixed-seed repeat, end-to-end reuse/stale policy rejection). Cùng 53 test trước =57 test liên quan pass; compile/diff checks và Make dry-run pass. Big run dùng `--stage train`; end-to-end orchestration đã chạy trên hand-built fixture, không sinh lại source500k. Learning curve đã xem trực tiếp.

Direct CPU dependencies pin ở `requirements-training.txt`, versions đúng runtime đã kiểm tra; uv recipe có explicit CPU index. Chưa có full transitive lockfile. Make có `setup-training`, `train`, `all` và CONFIG/PYTHON overrides. Không đánh giá GPU/runtime Anomaly/inference; chạy offline CPU duy nhất seed chính.

**Task tiếp theo Phase 5:** baseline tabular/MLP cùng feature/split và budget; ablation node/graph và snapshot lag; chọn threshold trên validation, phân tích theo type/scenario/time, giữ test tới khi chốt cấu hình. Dataset synthetic shortcuts/funding/activity còn giới hạn; validation AP cao chưa là bằng chứng generalization thực tế.


## 18. Phase 5 — validation-first baseline, ablation và test (2026-10-10)

Đã chạy trên seed dataset `20261002`, 500.000 request, 35 transaction feature + 12 node feature. Dùng lại GAT Phase 4; train thêm feature MLP, endpoint MLP và GAT self-loop. Tất cả neural model dùng seed 42, cùng split, batch 128, optimizer/learning rate/pos_weight và ngân sách 3 epoch. Best epoch vẫn chọn theo weighted validation loss, không đổi sang chọn epoch theo AP sau khi xem kết quả. Kiến trúc/số tham số và compute khác nhau: cùng ngân sách epoch không có nghĩa cùng capacity. Rule không train weights; threshold được chọn trên validation.

Chọn threshold tối đa F1 trên validation (tie ưu tiên threshold cao hơn). Chọn ứng viên neural model theo validation F1, AP phá tie. `selection.json` chứa model/checkpoint SHA và mọi threshold, ghi trước khi chạy test; checksum được kiểm tra sau test. Test chỉ dùng để báo cáo, không cập nhật lựa chọn. Ứng viên theo quy tắc này là **feature MLP**, threshold `0.927707314491272`; model GAT trong config training vẫn được giữ để nghiên cứu.

| Model | Validation F1 (threshold đã chọn) | Test AP | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| GATv2 đầy đủ | 0.9567 | 0.9751 | 0.9480 | 0.9452 | 0.9466 |
| Feature MLP | 0.9726 | 0.9842 | 0.9613 | 0.9656 | 0.9635 |
| Endpoint MLP | 0.9420 | 0.9736 | 0.9299 | 0.9392 | 0.9346 |
| GAT self-loop | 0.9673 | 0.9807 | 0.9606 | 0.9586 | 0.9596 |
| Amount / trung vị cùng source + type | 0.2431 | 0.0773 | 0.1719 | 0.4145 | 0.2430 |
| History rule | 0.6780 | 0.6580 | 0.9805 | 0.5252 | 0.6840 |

`AP` là average precision dạng bậc thang theo score ties, không phải tích phân trapezoid PR curve. Hai rule đọc cùng causal tensor và train preprocessing: amount dùng `log_amount_to_type_median`; history rule lấy max các cột chuẩn hóa gồm amount ratio, request count 120s và incoming distinct transfer sources của source/destination. Đây là rule tham chiếu đơn giản, không phải rule cũ của EDA; threshold có thể ngoài [0,1] vì rule score không phải sigmoid.

Feature MLP test: TP 1.938, FP 78, FN 69, TN 97.781 trên 99.866 request (2.007 anomaly). False positive rate 0.0797% trên normal. Recall scenario: contextual_amount 92.52%, velocity_burst 99.46%, graph_fan_in 99.02%, rapid_forwarding 100% (127 case). 61/69 FN thuộc contextual_amount. Recall theo type: TRANSFER 98.70%, PAYMENT 98.37%, CASH_IN 91.42%, CASH_OUT 86.06%, DEBIT 97.76%. Cohort scenario chỉ có positive thì precision/F1 không phản ánh false alarm: nên đọc recall/FN; AUC/AP được để null khi chỉ có một class. Breakdown theo type/scenario/tháng nằm trong `report.json`.

**Diễn giải:** feature MLP và self-loop vượt GAT đầy đủ trong lần chạy này; chưa thấy lợi ích của message passing với dataset, snapshot 1h và budget hiện tại. Feature MLP vẫn có scalar thống kê neighborhood/funding nên không được kết luận mọi thông tin graph là vô ích. Điểm cao chủ yếu có thể giải thích bằng causal behavior features khớp giả định generator; đây chưa là bằng chứng generalization trên giao dịch ngân hàng thật.

Chi phí CPU 1 thread, forward amortized trong microbatch: GAT 38.70 us/request; feature MLP 0.75; endpoint MLP 0.98; self-loop 14.29. Chưa tính causal feature extraction và không phải latency online một request. GAT test wall 28.86s bao gồm xây cache mới; model sau dùng cache nóng nên không so trực tiếp wall time đó. Chưa benchmark online/restart.

Artifacts chuẩn: `outputs/contextual-v4/seed-20261002/evaluation/phase5-v1/`: config, checkpoint từng baseline, history/learning curves, immutable selection, test predictions, report JSON/Markdown, bộ hình báo cáo PNG/PDF trong `figures/`. Chạy bằng `make evaluate CONFIG=...` hoặc `python scripts/evaluate_phase5.py --config ...`; muốn chạy lại đặt `evaluation.output` mới, không ghi đè test report cũ.

**Giới hạn còn lại:** mới một seed dataset và seed train; chưa multi-seed robustness/CI; chưa thay snapshot lag hay quét hyperparameter. Funding/activity clipping của simulator và shortcut behavior vẫn còn. Threshold weighted-BCE score không là xác suất đã calibration. Tuning tương lai cần evaluation protocol mới, không tiếp tục tối ưu trên test vừa mở. Phase 6 đóng gói/replay offline, giữ schema/causal state giống train; chứng minh giá trị GAT cần tiếp tục thử nhiều seed và các scenario cần quan hệ graph ngoài scalar hiện có.


### Hình báo cáo Phase 5

Bộ hình chính được rút gọn tại `evaluation/phase5-v1/figures/`:

- `roc_pr_curves`: PR và ROC cùng một hình để so sánh các phương pháp; PR hữu ích khi anomaly chỉ khoảng 2%.
- `confusion_matrix`: duy nhất GATv2 là model nghiên cứu chính, hiển thị count và phần trăm theo true class trong cùng ô.
- `model_comparison`: bảng precision/recall/F1/AP/ROC-AUC/threshold; không lặp lại thành bar chart.

Mỗi hình có PNG/PDF; `evaluation_charts.pdf` gồm 3 trang. Bộ chẩn đoán mở rộng cũ đã chuyển vào `appendix/`; không sinh mặc định. Dùng lại saved predictions và threshold đã freeze, không train/inference/tuning lại. GATv2 là model nghiên cứu chính trong ma trận và được tô đậm trong bảng; MLP là baseline đối chứng. Kết quả chọn ứng viên theo validation vẫn ghi feature MLP trong selection.json, không thay đổi kết quả lựa chọn hoặc threshold khi chỉnh cách trình bày. Lệnh tái lập: `python scripts/plot_evaluation.py --run ...`. Đã kiểm tra score ties và AP theo step integration; 61 kiểm thử liên quan validation/split/features/training/evaluation đã pass trước lần chỉnh presentation này.


### Phạm vi đánh giá được chốt lại

Theo yêu cầu người dùng: chỉ đánh giá GATv2. `evaluation.train_variants=[]`, `include_rule_baselines=false`; runner mặc định chỉ dùng GATv2 checkpoint/threshold. Báo cáo trình bày hiện tại: `gatv2_report.md/json`, ROC/PR GATv2, một confusion matrix và bảng `gatv2_metrics`. Comparison/MLP từ lần chạy trước là dữ liệu lịch sử, không thuộc bộ báo cáo chính. Không train/score/chọn threshold lại khi đổi phạm vi trình bày.


### Đối chiếu GATv2 theo tập dữ liệu

Theo yêu cầu bổ sung: bộ hình chính cần đủ **train, validation, test**, cùng best checkpoint GATv2 và threshold validation đã freeze. Ba ma trận lần này tương ứng ba partition của một model, không phải các model khác. Cần phân biệt metric train in-sample với validation và test. Split đang áp dụng 50/30/20 theo các khoảng thời gian liên tục và nguyên ngày; đếm request thực tế 251.481/148.653/99.866 tương ứng 50.30%/29.73%/19.97%. Chưa đổi sang lấy tỷ lệ từng ngày: đã hỏi lại người dùng về ý nghĩa tỷ lệ để tránh thay đổi protocol và phải train lại ngoài ý định. Runner `evaluate_gatv2_splits.py` và `gatv2_partitions/` truy vết cùng checkpoint/threshold, dùng lại prediction validation/test.


### Cleanup dung lượng

Đã dọn payload có thể sinh lại trong `outputs/`: raw/split JSONL, feature tensors và backup, graph context cache, diagnostic arrays và archive data cũ. Giải phóng khoảng 8.73 GiB; outputs còn khoảng 43 MiB, giữ checkpoints/reports/figures và evaluation predictions. Trạng thái hiện tại không còn raw/split/features trên đĩa: cần `make generate-dataset` rồi dựng lại pipeline input trước khi train/replay. Các kết quả trong plan là kết quả lịch sử đã giữ báo cáo, không có nghĩa tensors vẫn tồn tại. `make clean-data` tái sử dụng cleanup; `outputs/cleanup.json` ghi các đường dẫn đã dọn.


## 19. Feature EDA — leakage, redundancy, sensitivity và overfit screening

Bổ sung theo review; ưu tiên đánh giá trước khi sửa schema. CLI `scripts/audit_features.py`, Make `feature-eda`, tùy chọn tại `configs/dataset.json.feature_eda`.

- [x] Dựng tạm seed 20261002, 500k request sau cleanup. 21/21 payload files khớp SHA với manifest input checkpoint; không dùng nhầm dataset/model. Dữ liệu tạm được dọn sau báo cáo.
- [x] 10 invariant checks: schema/policy, finite input, split disjoint/chronological, strictly prior graph prefixes, scaler train-only metadata và train moments, không có cột copy trực tiếp target/complement. 15 causal regression tests pass. Đây là kiểm tra có giới hạn, không chứng minh vắng mọi leakage/arrival-time mismatch.
- [x] Thống kê toàn bộ 251.481 train request; Pearson/Spearman trên fixed sample 50k train. Có 2 transaction constants `type_unknown`, `channel_unknown`; `has_previous` chiếm 99.60% một giá trị. Các cột fallback/missing flags không tự động vô nghĩa.
- [x] Node statistics/correlation trên 64 train cutoffs, 66.560 available known-node observations (lặp node theo thời gian, không coi là independent samples). `unknown` hằng vì sample known nodes; chưa đo effectiveness của unknown fallback realtime.
- [x] Có 2 transaction candidate pairs, gồm cặp unknown constants. Transfer incoming distinct-source count vs received amount: Spearman 0.9989, nhưng 99.08% joint modal values; sau khi bỏ phần này chỉ còn 0.5488 trên 460 sample. Không thể kết luận hai feature trùng thông tin chỉ từ full correlation. 4 node candidate pairs: count/unique-partner và median/history flag, cần kiểm tra theo nhóm trước khi rút gọn.
- [x] Grouped permutation trên 7.334 validation request / 64 hourly contexts (132 positive), 3 repeat. Giữ nguyên checkpoint epoch 2, threshold 0.9681, graph/node embeddings; baseline predictions khớp subset đã lưu. AP baseline subset 0.9859. Mean AP drop: request velocity 0.4915; transfer neighborhood 0.2929; behavior habits 0.2253; receiver novelty 0.1606; amount reference 0.1226; source history 0.0133; type 0.0011; calendar 0.0008; channel encoding 0.0002. Đây là sensitivity, không phải unique contribution; drops không cộng được, permutation có thể phá cross-group dependencies. Repeat std không phải confidence interval.
- [x] Đối chiếu metrics cùng checkpoint: train F1/AP 0.9101/0.9448, validation 0.9567/0.9816, test 0.9466/0.9751. Train cold-start 7.95% vs holdout 0%; reference reliable 88.43% vs validation 96.23%, test 96.17%. Chưa kết luận không overfit; khác biệt lịch sử/distribution cần xem cùng gap. Test chỉ mô tả, không dùng loại/chọn feature.
- [ ] Retrained ablation trên chính GATv2 với các nhóm feature đã chốt; chưa chạy. Không sửa schema hoặc train model khác trong bước EDA này.

Artifact: `outputs/contextual-v4/seed-20261002/eda/feature-audit-v1/report.md/json`, CSV thống kê/pairs/association/history coverage, correlation matrices NPZ, 3 PNG và PDF tổng hợp. File `causal_tests.txt` và `audit_code.json` ghi validation/code provenance. Báo cáo chỉ có summary/correlation, không giữ raw/tensor/cache lớn.

Kết luận bước EDA: chưa tìm thấy lỗi trong các leakage checks đã chạy; có candidate redundancy và nhóm feature ít sensitivity trong checkpoint hiện tại. Chưa có bằng chứng đủ để xóa feature ngay hoặc khẳng định không overfit. Bước kế tiếp là chốt ablation GATv2 bằng train/validation, không dùng lại test để tuning.


## 20. Feature reviewer và window động (chuẩn bị đánh giá)

- Thêm mean completed của cùng source+type, window mặc định 7d, count/has-history/reliability/ratio; giữ median 30d riêng. Thêm count request 300s bên cạnh 120s/1h/24h.
- Thêm optional request auth/device/platform: one-hot, boolean + availability flags, new_device theo prior requests của từng source. Không đưa raw device_id vào numeric tensor; posted_at hiện tại tiếp tục excluded. Generator/BE hiện chưa cung cấp context này; chưa đo effectiveness của feature thiếu nguồn.
- Cấu hình thử nghiệm `configs/features-v3.json` có 58 transaction dimensions, 12 node dimensions. Legacy 35-dim config/checkpoint không bị ghi đè; tensor schema v3 chỉ dùng model train mới.
- `scripts/configure_feature_window.py` và `make configure-window WINDOW=1d|1w|1mo` tạo config; window chung hoặc mean/median/graph/node độc lập. 1mo fixed 30d. Request frequency/neighborhood ngắn độc lập với graph lookback. Output phân biệt theo canonical duration để không trộn model/provenance.
- EDA được cập nhật nhóm mean/auth/device và coverage. Khi input không khớp checkpoint reference, bỏ qua model importance và đánh dấu metrics cũ là historical; không lấy số cũ để tuyên bố model mới tốt hơn.
- Đánh giá kế tiếp: bổ sung nguồn context nếu cần, dựng feature, so GATv2 trên validation giữa các window, freeze cuối rồi mới test. Chưa train lại hoặc chạy full-dataset experiment trong bước chuẩn bị này. Chi tiết và lệnh: `docs/FEATURE_WINDOWS.md`.

Kiểm chứng bước feature/window: 69 tests liên quan validation, split, causal tensors, training, EDA và reviewer additions pass. Bao gồm export tensor v3 trên fixture, mean cùng nghiệp vụ, missing-vs-false auth, device novelty same-timestamp, parser/profile windows và backward compatibility 35 dimensions. Đã tạo ba config 1d/1w/1mo ở `outputs/configs/`; chưa sinh lại 500k hoặc train checkpoint mới.


## 21. Device có nguồn trong generator

Đã cập nhật repo cùng cấp `anomaly-data-pipeline`: transaction snapshot có `device={device_id,platform}`; request event giữ cùng identity/platform, terminal không được dùng để bổ sung feature tương lai. `DeviceHistory` dùng RNG riêng, không đọc label/scenario; TOML điều khiển tỷ lệ device mới (mặc định 0.04), pool device quen (tối đa 3), phân bố Android/iOS. Web/desktop giữ platform tương ứng. Không emit cờ oracle `new_device`: detector tính từ prior request history. Bật device không đổi amount/type/labels.

Dataset có device dùng `configs/device-v1.json`, root `outputs/contextual-v5-device/`; không ghi đè kết quả v4. GAT validator hỗ trợ snapshot device và kiểm tra khớp request. Auth chưa được generator bổ sung; Go Backend enrichment chưa triển khai. Device IDs mô phỏng chưa phải bằng chứng xác thực hardware/trusted-device.

Kiểm chứng: 70 tests GATv2 pass; generator regression và integration 1.200 request kiểm tra data validation, snapshot/request consistency, novelty/reuse, normal-new/anomaly-known, reproducibility và nhãn/amount không đổi khi bật device. Chưa tạo lại bộ 500k, chưa đánh giá model/ablation với device mới. Bước tiếp theo: sinh dataset mới, EDA coverage/correlation/sensitivity, train GATv2 theo các window và chọn bằng validation.

## 22. Kết quả đối chứng device (cập nhật sau mục 21)

Đã sinh 500k device, train hai GATv2 cùng 58 chiều, seed/budget (3 epoch), chọn checkpoint theo validation loss và threshold validation trước test. Test F1: original 35 chiều 0.9466; 58 chiều che device 0.9378; bật device 0.9322. Bản mới chưa cải thiện trong lần thử này. Dataset device dùng lại các giao dịch/nhãn cũ, đã kiểm tra SHA của projection legacy; không dùng kết quả test để đổi lựa chọn frozen. Báo cáo: `outputs/contextual-v5-device/seed-20261002/experiments/device-v1/report.md`. Một seed, chưa chứng minh ý nghĩa thống kê hoặc chất lượng dữ liệu ngân hàng thật.

## 23. Áp dụng specs/review.md — replay/inference

- [x] Thay predictor tabular bằng `ReplayPredictor` dùng checkpoint GATv2, schema/scaler train-only và validation threshold trong bundle có checksum.
- [x] Chia sẻ graph/node context với offline; transaction feature dùng cùng causal State. Giữ first-request snapshot bucket của training; không đọc future/current terminal khi score.
- [x] Nhận raw transaction contract review, canonical schema 2 và native Go transfer schema 1. Trả risk_score, predicted_label, threshold, model_version, feature_version; chỉ GATv2.
- [x] History adapters JSONL/EventStoreDB/Mongo event projection; Kafka worker gửi result có acknowledgement trước khi commit. Replay có thể khôi phục từ history khi restart.
- [x] Profile graph/node history 7 ngày trong `configs/review-v2.json`, output riêng, window cấu hình động; đã train và đánh giá theo mục 24.
- [ ] Kiểm chứng hạ tầng Kafka/EventStoreDB/MongoDB thật; Go consume result/upsert và FE hiển thị; enrichment auth/device trong Go, projection completeness/watermark và node registration. Không đánh dấu hoàn tất end-to-end bằng unit tests.
- [ ] Tối ưu replay prefix thành state phục vụ throughput; bảo toàn policy lịch sử toàn cục, same-timestamp và rollback khi late arrival. GATv2 hiện không có TGN/LSTM memory.

Chi tiết mapping, contract, lệnh chạy và giới hạn tại `docs/REVIEW_V2_IMPLEMENTATION.md`.

Kiểm chứng local 2026-10-18: 82 tests và 14 subtests pass. Bundle thật khoảng 260 KiB; smoke checkpoint đã train với request index 20 của bộ 500k cho tensor và score replay khớp offline (max tensor difference 0). CLI replay từ working directory khác chạy thành công. Artifact parity: `outputs/inference/review-v2/parity.json`. Không chạy broker/database hoặc sửa Go trong bước này.

## 24. Train lại review-v2 — graph/node history 7 ngày

Ngày 2026-10-18, chạy `scripts/run_pipeline.py --stage all --config configs/review-v2.json`, sau đó `--stage evaluate` với cùng config. Raw dataset 500k giữ nguyên; validator mới chạy lại đầy đủ (0 errors, các warning giống bản trước), validation report và split lưu riêng. Split vẫn 251.481/148.653/99.866 request (5.002/2.991/2.007 anomaly). Đối chiếu 15 input checks với full-device 24h đều khớp: transaction tensors/scaler, labels/time/masks, target/graph timeline và node catalog không đổi. Graph/node history đổi từ 24h sang 7d; node scaler được fit lại bằng train. Mean 7d, median 30d, 58 transaction + 12 node dimensions giữ nguyên.

Train seed 42, 3 epoch, cùng optimizer/architecture/budget, không train model khác. Validation loss theo epoch 0.04207 / 0.03951 / 0.05091; AP tương ứng 0.97291 / 0.98198 / 0.98496. Chọn epoch 2 bằng **minimum weighted validation loss**, không đổi sang chọn AP sau khi xem kết quả. Reload checkpoint đã kiểm chứng khớp prediction. Training khoảng 484 giây gồm context/cache/validation; không so compute trực tiếp với run cũ sử dụng cache đã warm.

Threshold 0.94979429 chọn bằng validation F1 và freeze trước khi score test. Metric cùng checkpoint/threshold:

| Partition | Precision | Recall | F1 | AP |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.86825 | 0.96701 | 0.91497 | 0.95300 |
| Validation | 0.94356 | 0.96690 | 0.95509 | 0.98198 |
| Test | 0.94176 | 0.95067 | 0.94619 | 0.97293 |

Test confusion: TN 97.741, FP 118, FN 99, TP 1.908. So full-device 24h, F1 tăng 1.40 điểm phần trăm (0.93218 → 0.94619), FP giảm 32 và FN giảm 25. So original 35 chiều (F1 0.94661), kết quả gần ngang nhưng chưa vượt. Đây là một seed và holdout đã dùng trong các báo cáo trước, không phải test độc lập mới; không kết luận ý nghĩa thống kê hoặc tách riêng ảnh hưởng graph window với node-history window.

Artifact: `outputs/contextual-v5-device/seed-20261002/experiments/review-v2/{protocol,input_checks,report}.json`, `report.md`; training/evaluation ở đường dẫn trong config, có learning curves và ROC/PR/confusion đủ train/validation/test. Không tự đổi bundle inference mặc định hoặc mở kết nối hạ tầng.

## 25. Refactor bộ tiêu chí và EDA sau review

Ngày 2026-10-19, refactor `specs/Bo_tieu_chi_phat_hien_giao_dich_bat_thuong.md`, giữ 49 mã. Catalog `specs/anomaly_criteria.v2.json` lưu công thức, unit, evaluation mode, nguồn, lịch tham chiếu, tham số thử nghiệm, trường hợp bình thường dễ nhầm và phạm vi hỗ trợ model. Bảng Markdown sinh từ catalog qua `render_criteria_spec.py`; có kiểm tra chống lệch. Phân biệt behavior_signal/external_fact/contextual_review và matched/not_matched/insufficient_data; không gắn matched thành nhãn anomaly. Làm rõ tiền settled/projected, baseline trước cửa sổ xét, timestamps/availability, episode không trùng và điều kiện dùng watchlist. Bỏ bảng điểm cộng tối đa 135 chưa được hiệu chỉnh; giữ risk_score/criterion_results/decision riêng. Các tham số mới là đề xuất nghiên cứu, giá trị null ngăn matching đầy đủ đến khi có cấu hình nghiệp vụ.

Pipeline `make dataset-eda` dùng `configs/criteria-eda-v2.json`, output `outputs/contextual-v5-device/seed-20261002/eda/criteria-v2/`. Dùng lại raw/feature/checkpoint 500k của review-v2, không đổi generator, nhãn, scaler, model hay threshold; config train đã preregister giữ nguyên SHA. Config EDA trỏ đúng checkpoint 7d thay vì reference 35 chiều cũ. Sửa lỗi grouping của `type_mean_reference_reliable`: phải thuộc amount_reference, không phải one-hot transaction_type.

- [x] 87 tests + 14 subtests pass; causal checks trong EDA pass 15 tests; catalog đủ 49 ID, nội dung Markdown khớp JSON; test missing-vs-false, shared device count, calendar boundary và tri-state source readiness.
- [x] Dataset EDA Matplotlib: 500k, 10k anomaly, đủ năm nghiệp vụ; 4 trang tổng quan/phân bố amount/hành vi-funding/graph requests. Không thêm model hoặc diagnostic rule F1.
- [x] Feature EDA: 10 screening invariants pass, toàn bộ payload khớp checkpoint; 58 transaction/12 node dimensions. 13 transaction constants và 52 correlation candidates, gồm nhiều cặp hằng/giá trị phổ biến. Không tự động xóa feature.
- [x] Grouped permutation trên 7.334 validation request (132 positive), 64 hourly contexts, 3 repeat; baseline AP subset 0.98394, tái hiện prediction đã freeze. Mean AP drops: velocity 0.54675; transfer neighborhood 0.27068; habits 0.19231; receiver novelty 0.15044; amount references 0.13735; device context 0.000947; auth 0. Graph/node embeddings cố định, không phải unique/causal importance hoặc retrained ablation.
- [x] Criteria source readiness: 25 sources_present, 15 partial_sources, 9 missing_sources. Đây là nhóm nguồn có xuất hiện, không phải 25 tiêu chí đã thực thi/khớp hoặc đủ per-account support. EDA không tạo criterion labels/matches hay score lại test.
- [x] Device/platform 100% request, auth 0%; 23.731 device, 0 device dùng chung account, max 1 account/device. Generator chưa mô phỏng shared-device/login/security events.
- [x] Train calendar coverage cho A01 90d và A03 91d bằng 0; validation/test mới có thể đủ ngày. A04 cấu hình đề xuất 210d vượt dataset 181d. Đây là upper bound calendar, không thay support/watermark. Mean7d reliable trên train khoảng 54.21% so median30d 88.43%; cold start train 7.95%, holdout 0%.
- [ ] Bổ sung prehistory/nguồn quan sát, trường hợp bình thường dễ nhầm và kịch bản cần thiết trước khi mở rộng feature/nhãn. Feature mới phải có causal tests và protocol validation/multi-seed; dùng holdout mới để đánh giá cuối, không tiếp tục tuning trên test đã xem.

Báo cáo: `dataset/dataset_eda.pdf`, `features/report.md` + `feature_audit.pdf`, `criteria/report.md` + `criteria_readiness.pdf`, và CSV coverage/correlation trong output root. Các metric GATv2 cũ chỉ được đọc để mô tả, không có train hoặc thay threshold ở bước này.

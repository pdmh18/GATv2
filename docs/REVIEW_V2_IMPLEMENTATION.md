# Áp dụng specs/review.md

## Phạm vi

Giữ GATv2: `x + edge_index → embeddings`, sau đó ghép source/destination embeddings với transaction features để tạo logit, sigmoid và risk score. MLP ở đây là prediction head của GATv2, không phải model baseline riêng. Không thêm TGN/LSTM hoặc SMOTE.

Replay theo thời gian bảo đảm nhân quả; temporal features tóm tắt thứ tự và khoảng cách giữa các giao dịch. GATv2 hiện tại không có recurrent memory và không tự học chuỗi như TGN/LSTM. Không mô tả thứ tự event như bằng chứng rằng đã có mô hình chuỗi.

## Đối chiếu feedback

| Góp ý | Áp dụng / giới hạn |
| --- | --- |
| `avg_amount_7d` | `log_src_type_completed_mean_amount`, cửa sổ mặc định 604800 giây, cùng source + nghiệp vụ, chỉ giao dịch đã hoàn tất trước request. Giữ median 30 ngày thành feature riêng; không gọi median là average. Có count/missing/reliability và ratio so với mean. |
| `tx_count_5m/1h/24h` | Các cột `log_src_request_count_300s/3600s/86400s`. Count request, gồm cả giao dịch sau này thất bại; không dùng kết quả tương lai để lọc. Cột 120s cũ vẫn có trong schema 58 chiều. |
| `time_since_last_tx`, `new_receiver` | Có từ prior requests; giao dịch đồng thời dùng cùng lịch sử trước timestamp. |
| Auth / device / platform | Adapter nhận đúng trường review, boolean phân biệt thiếu/false, categorical one-hot; device ID chỉ dùng tra novelty, không đưa mã ID vào số thực. Auth chưa có nguồn trong generator/Go hiện tại; không tự giả định verified=true. |
| Graph 7/30 ngày | Profile mới `configs/review-v2.json`: graph và node history 7 ngày, mean 7 ngày, median 30 ngày. Window tiếp tục cấu hình độc lập. Profile này cần export/train mới; không sửa policy của checkpoint 24h đã train. 30 ngày là lựa chọn thử nghiệm, chưa được xác nhận tốt hơn. |
| Leakage / redundancy / importance | Báo cáo EDA trước đó đã có; đối chứng device có hai GATv2 cùng 58 chiều. Chưa có full retrained feature ablation hay multi-seed. Không xóa feature chỉ vì correlation hoặc permutation importance thấp. |
| Replay / inference | `ReplayPredictor` dựng đúng input GATv2 bằng cùng `State`, scaler train-only và hàm dựng graph/node dùng chung với offline. Không gọi API tabular `predict_proba(DataFrame)` nữa. |
| EventStoreDB / MongoDB | Có reader lịch sử và khả năng dựng lại khi restart. EventStore đọc stream canonical, adapter nhận native Go schema 1 hoặc dataset schema 2. Mongo dùng projection event được mô tả dưới đây. Chưa xác nhận kết nối server thật. |
| Kafka → ML → BE | Worker đọc request, lấy lịch sử, inference, gửi result topic rồi mới commit input. Go phải consume result và cập nhật backend/FE; repo này chưa sửa Go hay frontend. |

## Bundle dùng cho inference

```bash
python scripts/export_inference_bundle.py \
  --training outputs/contextual-v5-device/seed-20261002/experiments/device-v1/without_device/training \
  --selection outputs/contextual-v5-device/seed-20261002/experiments/device-v1/without_device/evaluation/selection.json \
  --model-version gatv2-device-control-v1 \
  --output outputs/registry/gatv2-device-control-v1
```

Bundle gồm checkpoint, feature manifest/options, transaction/node scaler, node catalog, frozen selection và checksum. Threshold lấy từ validation; không có threshold mặc định 0.5 ở serving và không chọn lại bằng test. Bản `without_device` ở ví dụ là đối chứng đã train; mask device nằm trong checkpoint và được giữ khi inference. Đây không phải khẳng định triển khai bản tốt nhất: checkpoint 35 chiều cũ vẫn có test F1 cao hơn trong thí nghiệm lịch sử. Có thể export checkpoint GATv2 khác với selection tương ứng và model version riêng.

Bundle hoạt động độc lập với raw/split/tensor đã export, nhưng **inference vẫn cần nguồn lịch sử**. Node chưa có trong catalog dùng unknown bucket giống policy offline; chưa có online registration và phân loại loại node mới. Cần refresh catalog/bundle hoặc thiết kế registry node riêng trước khi dùng dài hạn với tài khoản mới.

## Request và response

`scripts/run_inference.py --config configs/inference.json --request request.json` nhận raw JSON đúng bảng review, hoặc request event canonical/native. Raw contract dùng VND và fee=0; khi cần fee thực phải dùng event envelope. ID như `TX001` được ánh xạ UUID nội bộ ổn định để replay; response giữ nguyên ID bên ngoài. `posted_at`, số dư kết quả và label/risk không thuộc request đầu vào.

```json
{
  "transaction_id": "TX001",
  "source_account": "ACC001",
  "destination_account": "ACC002",
  "amount": 50000000,
  "transaction_type": "TRANSFER",
  "channel": "MOBILE",
  "authentication_method": "OTP",
  "authentication_verified": true,
  "device_id": "DEVICE01",
  "platform": "ANDROID",
  "created_at": "2026-10-17T00:50:00+07:00"
}
```

Response: `transaction_id`, `risk_score`, `predicted_label` (0/1), `threshold`, `model_version`, `feature_version`. Risk score là sigmoid logit từ weighted BCE, chưa được hiệu chỉnh thành xác suất gian lận thực tế.

## Nguồn lịch sử và causal cutoff

- JSONL: `{"kind":"jsonl","path":".../events.jsonl"}`. Dùng kiểm tra offline, không phải latency benchmark realtime.
- EventStoreDB: `{"kind":"eventstore","uri_env":"EVENTSTORE_URI","stream":"banking"}`. Biến môi trường chứa URI; không ghi credentials vào config. Adapter tách `Record.event` từ journal Go (`id/type/at/event/transfer`), kiểm tra metadata khớp và bỏ snapshot `transfer`, profile, funding/loan records. Cũng nhận native event trực tiếp. Native Go hiện chỉ có TRANSFER; để phục vụ mọi nghiệp vụ, upstream phải phát đủ request/terminal các nghiệp vụ còn lại. `TransferOTPUpdated` và `TransferCancelled` không thay đổi feature/graph; cancellation của Go không có balance snapshots, không tự điền số dư giả. Request của giao dịch hủy trước đó vẫn được tính velocity.
- MongoDB: `{"kind":"mongo","uri_env":"MONGO_URI","database":"fraud_ml","collection":"event_history"}`. Mỗi document có `timestamp_us` (UTC epoch microseconds) và `event` (nguyên event canonical/native); index `timestamp_us`. Đây là projection riêng, không đọc terminal snapshot trong collection transactions làm request. Upstream phải xây projection đầy đủ; reader không tự tạo dữ liệu.

Reader lấy toàn bộ committed prefix `< created_at/occurred_at`, không chỉ lịch sử source account hay 7 ngày. `new_receiver`, device novelty và các tỷ lệ thói quen hiện dùng toàn lịch sử. Thu ngắn prefix tùy tiện sẽ lệch feature so với train. Source phải chứa đầy đủ request lẫn terminal; kiểm thử không chứng minh tính đầy đủ của dữ liệu thực.

Transaction features dùng lịch sử `< t`. Graph/node dùng cutoff tại request đầu tiên trong bucket `snapshot_seconds`, giống `TemporalBatches`; không chuyển sang graph mới nhất giữa bucket. Events cùng timestamp với target, terminal hiện tại và events tương lai bị loại trước khi dựng input. Replay không train lại model và không cập nhật scaler.

BE cần commit lịch sử trước khi phát Kafka request; Mongo projection phải bắt kịp watermark tương ứng. Worker chưa có giao thức chờ projection/watermark, vì vậy chỉ chọn nguồn đã bảo đảm đầy đủ. Đọc lại toàn bộ prefix cho từng request hiện ưu tiên tính đúng/replay; chưa tối ưu throughput hay state incrementally.

## Worker Kafka

```bash
make inference-worker INFERENCE_CONFIG=configs/inference.json
# `make infer` gọi cùng worker.
```

Worker cần `requirements-inference.txt` (ML + kafka-python, esdbclient, pymongo); replay JSONL chỉ cần dependencies ML. Dependency availability không đồng nghĩa broker/database đã được cấu hình.

Config đặt bootstrap servers, input/result topic, consumer group, poll interval và thời gian chờ delivery. Topic input/output phải khác nhau. Worker chỉ dự đoán request; terminal có trong history và không phát prediction riêng. Input lỗi, lỗi đọc history hoặc lỗi gửi output làm worker dừng và không commit message đó.

Kết quả có Kafka key `transaction_id:model_version:feature_version`. Output có thể trùng nếu crash sau publish, trước commit; BE cần upsert theo ba trường này. Chưa có exactly-once transaction, DLQ, API HTTP hay consumer kết quả trong Go. Không tự phát message vào broker trong bước kiểm chứng local.

## Kiểm chứng

`tests/test_inference.py` đối chiếu tensor/score replay với offline cho đủ năm nghiệp vụ; kiểm tra graph snapshot dùng lúc train, auth/device, same-timestamp/future exclusion, duplicate delivery, restart, checksum/policy, raw/native contracts và commit-after-ack bằng adapter giả. Full suite kiểm tra regression train/evaluation. Kết quả test local không thay cho kiểm chứng Kafka, EventStoreDB, MongoDB và Go/FE đang chạy thật.

Kết quả ngày 2026-10-18: **82 tests pass, 14 subtests pass**. Đã export bundle thật `outputs/registry/gatv2-device-control-v1/` (~260 KiB). Smoke trên request index 20 của bộ 500k và checkpoint đã train: tensor `x`, `edge_index`, `target_edge_index`, `edge_features` đều chênh lệch tối đa 0 so với offline snapshot; risk score đều `0.9945715069770813`, threshold validation `0.9517361521720886`. CLI replay chạy thành công từ ngoài project root với prefix JSONL tương ứng. Báo cáo [parity.json](../outputs/inference/review-v2/parity.json) chỉ xác nhận parity trên request đó, không phải đánh giá lại model/test set hay chứng minh tích hợp server thật.

## Kết quả train lại profile 7 ngày

Đã train `configs/review-v2.json` trên cùng 500k, seed 42, 3 epoch. Checkpoint epoch 2 chọn bằng validation loss; threshold validation 0.94979429 freeze trước test. Test precision/recall/F1/AP lần lượt 0.94176 / 0.95067 / 0.94619 / 0.97293. F1 tăng 1.40 điểm phần trăm so full-device 24h (0.93218), gần ngang original 35 chiều (0.94661). Graph và node history đổi cùng lúc, một seed và holdout được tái sử dụng; không khẳng định ý nghĩa thống kê hoặc quy toàn bộ cải thiện cho graph window. Báo cáo [review-v2/report.md](../outputs/contextual-v5-device/seed-20261002/experiments/review-v2/report.md), chi tiết tại mục 24 trong training plan. Bundle control ở ví dụ inference phía trên vẫn là artifact riêng; chưa tự thay bundle mặc định.

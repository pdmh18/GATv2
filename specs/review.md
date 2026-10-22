# Review kế hoạch GATv2 — Version 2

> Trạng thái áp dụng: xem [đối chiếu implementation](../docs/REVIEW_V2_IMPLEMENTATION.md). Các nhận xét phía dưới là bản review tại thời điểm viết; mean 7d, count 5m, optional auth/device và window động đã được bổ sung. Inference hiện có GATv2 replay, bundle và history/Kafka adapters; tích hợp server Go/FE và kiểm chứng hạ tầng thật còn chưa hoàn tất. Replay theo thời gian không đồng nghĩa GATv2 có recurrent temporal memory.

## Đúng kế hoạch ban đầu

- Chia Train / Validation / Test theo thời gian
- không dùng dữ liệu tương lai khi tính feature
- Feature Engineering từ lịch sử
- Transaction + Behavioral + Temporal features
- Dựng graph Account → Account bằng Transaction
- Graph theo time window
- Đưa dữ liệu đã xử lý vào gatv2.py
- src_emb + dst_emb + transaction_features → MLP
- Logit → Sigmoid → Risk Score
- Threshold để ra 0/1

## Làm thêm

- Validate / kiểm tra dữ liệu trước khi train => ok
- Train bằng BCEWithLogitsLoss => ok
- pos_weight để xử lý class imbalance => ok (cách này cx đc, ban đầu tính dùng SMOTE, nhưng cần xem xét nếu weight quá lớn, model dễ báo fraud quá nhiều)
- Early stopping, gradient clipping, checkpoint best.pt => ok
- 35 transaction features + 12 node features có nghĩa là input nhiều quá so với dự kiến (Nhiều feature không đồng nghĩa model tốt hơn. Phải kiểm tra leakage, redundancy, feature vô nghĩa, overfit. Nên làm ablation / feature importance / correlation.)

## Nên rút gọn phạm vi lại như kế hoạch

### Đầu vào train (chuyển từ ảnh)

| English | Tiếng Việt | Nguồn |
| --- | --- | --- |
| source_account | Tài khoản nguồn | transactions.source.account |
| destination_account | Tài khoản đích | transactions.destination.account |
| amount | Số tiền giao dịch | transactions.amount |
| transaction_type | Loại giao dịch | transactions.type |
| channel | Kênh giao dịch | transactions.channel |
| authentication_method | Phương thức xác thực | transactions.authentication.method |
| authentication_verified | Xác thực thành công hay không | transactions.authentication.verified |
| device_id | Thiết bị thực hiện giao dịch | transactions.device.device_id |
| platform | Nền tảng thiết bị | transactions.device.platform |
| created_at | Thời điểm tạo giao dịch | transactions.created_at |
| posted_at | Thời điểm giao dịch được ghi nhận | transactions.posted_at |

### Cần thêm (chuyển từ ảnh)

| English | Tiếng Việt |
| --- | --- |
| avg_amount_7d | Số tiền giao dịch trung bình trong 7 ngày |
| tx_count_1h | Số giao dịch trong 1 giờ gần nhất |
| tx_count_24h | Số giao dịch trong 24 giờ gần nhất |
| time_since_last_tx | Thời gian kể từ giao dịch gần nhất |
| new_receiver | Có phải người nhận mới hay không |
| new_device | Có phải thiết bị mới hay không |

Tên collection có thể là: `transaction_features`.

- avg_amount_7d: giao dịch hiện tại có lệch mức chi tiêu bình thường không
- tx_count_1h / tx_count_24h: có giao dịch dồn dập bất thường không
- time_since_last_tx: hai giao dịch có xảy ra quá sát nhau không
- new_receiver: có chuyển cho người nhận chưa từng giao dịch không
- new_device: có dùng thiết bị chưa từng dùng không
- Rule-based baseline / model variants / ablation => ok

## Chưa đúng theo kế hoạch

- avg_amount_7d  → một số chỗ dùng rolling median/ratio 30 => cần bàn lại
- Graph window → hiện đnag dùng 24h, thấy hơi ít có thể đổi sàng 7ngayf hoặc 30 ngày

## Những cái cần nắm (version 2)


| Tiêu chí | Ban đầu | Update v2: đã làm hiện tại |
| --- | --- | --- |
| Đối tượng chính | Dựng Transaction Graph: Node = Account, Edge = Transaction | Node nền = account<br>Đối tác ngoài hệ thống = node riêng (thực thể tham gia giao dịch)<br>Edge = Transaction<br>Các loại node (account, merchant, cash, unknown) |
| Học quan hệ tài khoản | Học quan hệ Account ↔ Account thông qua các node hàng xóm trong graph | Như cũ |
| Học thứ tự giao dịch theo thời gian | Bổ sung temporal features từ lịch sử: tx_count_5m, tx_count_1h, tx_count_24h, avg_amount_7d, time_since_last_tx, amount_vs_avg_7d... | ~~không tự học thứ tự thời gian như TGN/LSTM, nên mình bù bằng temporal features từ lịch sử~~ Dùng cả 2, do event tuân theo thời gian, cũng là điểm nổi bật của đề tài<br><br>Hiện tại: 5m → 120s, avg 7d → median 30d, amount_vs_avg_7d → amount_vs_median_30d. |
| Xử lý timestamp | Python chuyển created_at/posted_at thành hour_of_day, day_of_week, time_since_last_tx, các feature theo cửa sổ thời gian. Có thể kết hợp time-window graph<br> | event['occurred_at']<br>hour<br>hour_sin + hour_cos<br>weekday<br>weekday_sin + weekday_cos+ weekend<br>lịch sử transaction<br>log_seconds_since_previous,<br>request history, <br>count theo các time window, <br>transaction history, graph theo time window |
| Dữ liệu đầu vào | Python chuẩn bị x = node features, edge_index = quan hệ account, edge_attr = đặc trưng transaction. Categorical phải encode, numerical xử lý/scaling nếu cần | Tất cả như cũ trừ edge_attr = transaction features đưa vào GATv2Conv thành <br>x + edge_index<br>      ↓<br>    GATv2<br>      ↓<br>Account embeddings<br>      ↓<br><br>src_emb + dst_emb + transaction_features<br>      ↓<br>     MLP<br>Tức là transaction features không đi trực tiếp vào GATv2Conv nữa, mà được ghép ở MLP cuối. lấy src_embedding + dst_embedding + transaction_features → MLP → sigmoid |
| Dynamic Graph | Khi transaction mới đến: thêm/cập nhật node-edge vào graph hiện tại. Khi service restart có thể đọc MongoDB/EventStoreDB để dựng lại graph | hiện tại chủ yếu là offline training pipeline, tiếp tục triển khai như kế hoạch. làm ở bước inference/integration. |
| Event Sourcing | EventStoreDB đóng vai trò lưu lịch sử. Replay event → lấy lịch sử → tính behavioral/temporal features → dựng graph cho GATv2 | Đúng kế hoạch về mặt kiến trúc Event Sourcing và cách xử lý lịch sử. Nhưng hiện tại mới làm phần replay logic trên dataset/event history offline, chưa nối thật với EventStoreDB. Tiếp tục triển khai |
| Event Replay | Đọc event theo thứ tự thời gian đến thời điểm t → tạo lại node/edge → tính lại feature → dùng model đã train để inference. Không phải train lại model | hiện tại đã làm được replay lịch sử để dựng feature/graph theo thời điểm, nhưng chưa hoàn chỉnh thành chức năng replay một thời điểm bất kỳ từ EventStoreDB rồi inference như hệ thống thật. tiếp tục triển khai |
| Real-time inference | Transaction mới → Kafka → Python ML Service → lấy history → Feature Engineering → Update Graph → GATv2 → MLP → Risk Score → Alert | Đúng kế hoạch về luồng Real-time Inference. Các thành phần Feature Engineering → Graph → GATv2 → MLP → Risk Score đã có logic.<br>Kafka → ML Service → lấy history realtime → cập nhật graph → trả Alert/Backend chưa được tích hợp hoàn chỉnh. Tieeos tục triển khai |
| State / Memory | Không cần gắn thêm memory cho phạm vi hiện tại. EventStoreDB/MongoDB giữ lịch sử; Feature Engineering đọc và tóm tắt lịch sử; graph hiện tại giữ trạng thái quan hệ | Hợp lí rồi, cần bổ sung như kế hoacjk:lấy state/history trực tiếp từ EventStoreDB/MongoDB khi chạy realtime |
| Feature Engineering | Nhóm phải tự code module Python. Train và inference phải dùng cùng một logic tính feature | Như cũ. |
| Behavioral features | Ví dụ: avg_amount_7d, avg_amount_30d, new_device, new_receiver, unique_receivers_7d, usual_channel... → mô tả thói quen/hành vi của tài khoản | Đã note ở trên |
| Temporal features | Ví dụ: tx_count_5m, tx_count_1h, tx_count_24h, time_since_last_tx, hour_of_day, day_of_week... → mô tả hành vi theo thời gian | Tất cả như cũ trừ tx_count_5m đang là 2m |
| Transaction features | Ví dụ: amount, type, channel, auth_method, auth_verified, device/platform... | Đã sử dụng amount, type, channel<br>Có thể bổ sung thêm , auth_method, auth_verified, device/platform |
| Graph Construction | Từ transaction tạo src_account → dst_account; có thể giới hạn graph theo 7/30 ngày gần nhất thay vì toàn bộ lịch sử | Đã note ở trên |
| Output của GATv2 | Lấy src_embedding + dst_embedding + transaction_features → đưa qua MLP → sigmoid → Risk Score | Như cũ |
| Hyperparameter | Tập trung trước vào hidden_dim, num_layers, heads, dropout, learning_rate, weight_decay; không cần tune quá nhiều với thời gian hiện tại | Như cũ |
| Framework | Dùng Python + PyTorch + PyTorch Geometric; không cần tự code công thức attention từ đầu | Như cũ |
| Implementation | Nên tách module: data/preprocessing → feature_engineering → graph_builder → model → training → inference | Ok chi tiết hơn rất nhiều <br>ingestion/ → đọc/load dữ liệu<br>validation/ → kiểm tra dữ liệu<br>features/ → Feature Engineering → history → node features → graph/context<br>models/ → gatv2.py → batching.py → trainer.py → variants.py<br>evaluation/ → đánh giá model → threshold → baseline → biểu đồ<br>registry/ →save/load model<br>inference/ → phần inference |


## Thiết kế BE gọi ML

### Luồng suy luận thời gian thực (Real-time Inference) — chuyển từ ảnh

| Bước | Thành phần / Xử lý |
| --- | --- |
| 1 | Frontend |
| 2 | Go Backend |
| 3 | BE tạo giao dịch |
| 4 | Kafka |
| 5 | Python ML Service nhận transaction/event |
| 6 | Lấy lịch sử cần thiết (EventStoreDB / MongoDB) |
| 7 | Feature Engineering |
| 8 | Dựng / cập nhật graph |
| 9 | GATv2 |
| 10 | MLP |
| 11 | logit |
| 12 | Sigmoid |
| 13 | risk_score |
| 14 | threshold |
| 15 | predicted_label 0/1 |

### Chỗ cần quan tâm — đoạn luồng BE → ML (chuyển từ ảnh)

| Bước | Thành phần |
| --- | --- |
| 1 | Go Backend |
| 2 | BE tạo giao dịch |
| 3 | Kafka |
| 4 | Python ML Service nhận transaction/event |

### Dữ liệu BE gửi tới ML Service

Tức là ở phía BE ko cần gửi tính năng gì xuống cho model, chỉ cần gửi dữ liệu đầu vào(input đã gửi) của transaction/event: ví dụ

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

Phía ML Service biến transaction/event thành node features, edge_index, transaction features, target_edge_index rồi đưa vào model đã train để inference

Sau khi model đã được train và load vào ML Service, mỗi khi có transaction mới, ML Service sẽ inference và trả kết quả cho BE: ví dụ

```json
{
"transaction_id": "TX001",
"risk_score": 0.91,
"predicted_label": 1,
"threshold": 0.70,
"model_version": "gatv2-v1",
"feature_version": "v1"
}
```

Rồi BE sẽ trả cho FE như design

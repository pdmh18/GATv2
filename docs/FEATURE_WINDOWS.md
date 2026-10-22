# Feature reviewer và window linh động

Cấu hình mới: `configs/features-v3.json`. Cấu hình `configs/dataset.json` và kết quả 35 chiều cũ được giữ riêng. V3 hiện có 58 transaction input dimensions (gồm one-hot/availability flags), 12 node dimensions; không phải 58 raw fields. Tensor v3 cần train checkpoint mới trước khi đánh giá model.

## Mapping các trường reviewer

| Trường đề cập | Cách xử lý |
|---|---|
| source/destination account | Map node index cho target edge; không dùng ID như số liên tục |
| amount | Giữ `log_amount`; bổ sung mean/ratio theo window bên cạnh median/ratio cũ |
| transaction_type, channel | Giữ one-hot và unknown bucket |
| avg_amount_7d | `log_src_type_completed_mean_amount`: mean completed trước request của cùng source + nghiệp vụ, mặc định 7 ngày; kèm count/has-history/reliability và ratio. Window nằm trong manifest, không cố định trong tên cột |
| tx_count_5m / tx_count_1h / tx_count_24h | Request counts 300/3600/86400 giây; giữ thêm 120 giây để đối chiếu. Bao gồm failed attempts, chưa dùng kết quả terminal hiện tại |
| time_since_last_tx, new_receiver | Đã có: thời gian từ request trước và novelty recipient theo toàn bộ prior requests |
| authentication_method | One-hot vocabulary cấu hình được; `has_authentication_method` phân biệt thiếu dữ liệu |
| authentication_verified | Giá trị bool + `has_authentication_verified`; missing không đồng nghĩa false |
| device_id | Chỉ dùng để tính `new_device`; không đưa raw ID/hashing tùy tiện vào tensor |
| new_device | Chưa từng thấy device này trong các request trước của cùng source; kèm `has_device`, `has_device_history`. Request cùng timestamp đều chấm trước khi cập nhật history |
| platform | One-hot vocabulary + `has_platform` |
| created_at | Dùng canonical request `occurred_at` để tính calendar/time-since; không lấy snapshot cuối |
| posted_at | Không dùng posted_at của giao dịch đang chấm; completed history chỉ gồm terminal events đã xảy ra trước đó |

Các field auth/device là **optional enrichment tại request**: `authentication_method`, `authentication_verified`, `device_id`, `platform` trong payload. Đây là phần mở rộng detector input; chưa phải thay đổi đã triển khai trong Go Backend hay generator repo cùng cấp. Loader/split/validator hỗ trợ đúng các key này. Category được lowercase khi encode (`OTP` → `otp`, `ANDROID` → `android`); device ID giữ nguyên identity. Khi upstream chỉ có snapshot nested authentication/device, cần adapter tạo request enrichment trước khi gửi ML, không join kết quả xác thực tương lai.

Dataset v4 cũ chưa có auth/device. Generator hiện đã bổ sung device/platform tại request; auth vẫn chưa được sinh. Khi thiếu trường, cờ availability bằng 0 và dùng unknown bucket. EDA ghi coverage/cột hằng; dữ liệu device mô phỏng cần đánh giá riêng, chưa chứng minh hiệu quả trên dữ liệu ngân hàng thật.

## Window truyền qua cấu hình

Đơn vị: `s`, `m`, `h`, `d`, `w`, `mo`. `1mo` luôn là **30 ngày**, không phải một tháng lịch. Window lịch sử là `[t-window, t)` theo causal replay; current terminal/same-timestamp events không tham gia.

Tạo cấu hình một window chung cho mean, median, graph và node history:

```bash
make configure-window WINDOW=1d
make configure-window WINDOW=1w
make configure-window WINDOW=1mo
```

Output tương ứng `outputs/configs/window-1d.json`, `window-1w.json`, `window-1mo.json`. Các duration cùng nghĩa như `1w`/`7d` có cùng canonical profile ID. Mỗi profile tách feature/training/evaluation/EDA output để không trộn checkpoint. Lệnh từ chối ghi đè config đã có.

Có thể đặt riêng từng window:

```bash
python scripts/configure_feature_window.py \
  --amount-window 1w --median-window 1mo --graph-window 1d --node-window 1d \
  --output outputs/configs/custom-window.json
```

`--amount-window` chỉnh **mean**, `--median-window` chỉnh median. `--window` đặt chung trước; các tùy chọn riêng ghi đè sau. Request-count windows 120s/5m/1h/24h và transfer-neighborhood 600s độc lập; thay graph 1 tháng không tự biến count 1h thành count 1 tháng. Có thể thay danh sách request windows trong JSON.

## Quy trình đánh giá tiếp theo

1. Bổ sung dữ liệu request auth/device ở upstream nếu muốn đo giá trị thực của nhóm đó.
2. Dựng lại raw/split (sau cleanup), export feature và kiểm tra EDA cho từng profile.
3. Train GATv2 checkpoint mới cho mỗi profile với cùng seed/budget/split; so sánh **validation**, không chọn window bằng test.
4. Chốt window + checkpoint + threshold trên validation rồi mới chạy test của cấu hình cuối.

```bash
make generate-dataset
make all CONFIG=outputs/configs/window-1w.json
# Chạy evaluate riêng sau khi đã chốt window bằng validation:
make evaluate CONFIG=outputs/configs/window-1w.json
```

`feature_eda.reference_training_output` / `reference_evaluation_output` trong cấu hình v3 hiện trỏ báo cáo cũ để tham khảo. Khi feature payload khác checkpoint, EDA bỏ qua permutation importance và đánh dấu metrics cũ là historical reference, không nhận chúng làm kết quả của schema mới. Sau khi đã train/evaluate profile mới, đổi hai reference này sang đúng output của profile nếu cần EDA model sensitivity tương ứng.

Chưa train lại 500k hoặc chọn window tối ưu trong bước cập nhật feature/configuration này.


## Dữ liệu device từ generator

Cấu hình dataset có device: `configs/device-v1.json`, output tách ở `outputs/contextual-v5-device/` để không trộn với báo cáo v4. Generator lưu `transactions.device.device_id/platform`; request event mang cùng `device_id/platform` ngay tại thời điểm request. Validator kiểm tra snapshot và request khớp nhau. Device không được lấy từ terminal event.

Generator dùng RNG riêng, có tỷ lệ tạo thiết bị mới cấu hình được, lưu pool thiết bị đã dùng theo account/platform; không đọc anomaly label/scenario để quyết định device. Không gán `new_device` từ generator: detector tính từ prior request history. Device mới có thể hợp lệ; anomaly vẫn có thể dùng thiết bị quen. Đây là ID mô phỏng client, chưa phải fingerprint phần cứng hay tính năng chặn giao dịch của backend.

```bash
make generate-dataset DATASET_RAW=outputs/contextual-v5-device/seed-20261002/raw
make all CONFIG=configs/device-v1.json
# Window động cho dataset device:
python scripts/configure_feature_window.py --base configs/device-v1.json --window 1w --output outputs/configs/device-1w.json
```

Trường auth vẫn chưa được generator sinh; chỉ device/platform có nguồn mới. Cần chạy EDA rồi train GATv2 v3 và đánh giá validation trước khi kết luận device giúp cải thiện model.

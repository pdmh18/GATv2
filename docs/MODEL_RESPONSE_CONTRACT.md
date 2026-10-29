# Hợp đồng output model → Backend → Giao diện (v1)

Tài liệu này chốt cách ML service trả kết quả cho Backend (BE) và cách BE chuyển kết quả đó cho Frontend (FE). Phạm vi là **hợp đồng dữ liệu và quy tắc hiển thị**; repo này chưa chứa mã Go Backend/Frontend và chưa có HTTP API.

Nguồn chuẩn máy đọc được: [`specs/model_response.v1.json`](../specs/model_response.v1.json). Code kiểm tra hợp đồng: [`src/fraud_pipeline/inference/response.py`](../src/fraud_pipeline/inference/response.py).

## 1. Luồng hiện tại

```text
Frontend → Go BE tạo giao dịch → Kafka (transaction)
        → Python ML service: history + feature + GATv2
        → Kafka (transaction-risk) → Go BE lưu kết quả → REST/SSE/WebSocket → Frontend
```

- ML service chỉ chấm điểm **request**; event terminal (`TransferCompleted`, `TransferCancelled`...) chỉ dùng làm lịch sử.
- Output hiện được publish lên topic `transaction-risk` (cấu hình tại `configs/inference.json`).
- ML service **không** quyết định allow/review/block và **không** tạo lý do giải thích. Các phần đó thuộc BE policy.

## 2. Hợp đồng ML service → Backend

### 2.1. Kafka envelope

| Thuộc tính | Giá trị |
| --- | --- |
| Topic | `transaction-risk` (theo config; input/output phải khác nhau) |
| Key | `transaction_id:model_version:feature_version` |
| Value | JSON UTF-8 đúng schema `model-response.v1` |
| Content-Type | Ngầm định `application/json` (worker hiện chưa gắn header) |
| Delivery | At-least-once; có thể trùng nếu worker crash sau publish, trước commit |
| Thứ tự | Chỉ đảm bảo theo key; không có thứ tự toàn cục giữa các giao dịch |

**Không parse key bằng cách split `:`.** Key là chuỗi opaque dùng cho partition/dedupe; các trường thật nằm trong value. `transaction_id` có thể chứa ký tự đặc biệt.

### 2.2. Payload

```json
{
  "transaction_id": "TX001",
  "risk_score": 0.9821,
  "predicted_label": 1,
  "threshold": 0.9517361521720886,
  "model_version": "gatv2-device-control-v1",
  "feature_version": "schema-3-0b9efd44d793"
}
```

| Trường | Kiểu | Ý nghĩa | BE dùng để | FE dùng để |
| --- | --- | --- | --- | --- |
| `transaction_id` | string, không rỗng | ID giao dịch phía ngoài. Với request raw, giữ nguyên ID BE gửi. Với event canonical/native, là `aggregate_id` | Join với giao dịch | Hiển thị/đối chiếu |
| `risk_score` | number `[0,1]` | Sigmoid của weighted-BCE logit | Lưu, hiển thị thô, audit | Thanh điểm/điểm rủi ro |
| `predicted_label` | integer `0` hoặc `1` | `1` nếu `risk_score >= threshold` | Áp policy riêng của BE | Badge trạng thái do BE quyết định |
| `threshold` | number `[0,1]` | Ngưỡng đã freeze trên validation cùng `model_version` | Lưu cùng kết quả | Vạch ngưỡng/tooltip |
| `model_version` | string, không rỗng | Định danh artifact model | Upsert + audit | Chi tiết/audit |
| `feature_version` | string, không rỗng | Định danh schema/policy feature | Upsert + audit | Chi tiết/audit |

Quy tắc bắt buộc:

1. `risk_score` **không phải xác suất gian lận đã hiệu chỉnh**. Không gọi là "xác suất", "99% chắc chắn gian lận", hoặc dùng làm căn cứ duy nhất để kết luận.
2. `predicted_label` chỉ là kết quả sau ngưỡng, **không phải quyết định cuối**. BE có thể thêm `decision`/`policy_version` riêng, nhưng không được ghi đè hoặc làm sai lệch output model.
3. `threshold` phải đi cùng `risk_score` và `predicted_label`. FE không tự tính lại label và không tự đổi ngưỡng.
4. `risk_score`, `threshold`, `predicted_label` không bao giờ là `null`, chuỗi, hay boolean. Không có trường nào được phép thiếu.
5. Output không chứa PII, số tài khoản, số tiền, device, auth, hay lý do. Nếu BE cần hiển thị các thông tin đó, lấy từ giao dịch gốc, không yêu cầu ML trả thêm.

### 2.3. Vì sao có `model_version` và `feature_version`

- `model_version` đổi khi checkpoint/threshold thay đổi.
- `feature_version` đổi khi schema hoặc policy feature thay đổi.
- Bộ ba `(transaction_id, model_version, feature_version)` là khóa upsert. Cùng một bộ ba phải cho cùng một kết quả; nếu không, BE cần log bất thường thay vì âm thầm ghi đè.
- Không tái sử dụng `model_version` với threshold khác. Threshold hiện được freeze từ validation trước khi mở test.

## 3. Backend nhận và lưu kết quả

### 3.1. Lưu trữ tối thiểu

```text
model_predictions
  transaction_id
  model_version
  feature_version
  risk_score
  predicted_label
  threshold
  received_at          # BE tự stamp, không có trong output model
  source               # kafka | replay | admin
  UNIQUE (transaction_id, model_version, feature_version)
```

- Upsert theo khóa trên; message trùng là chuyện bình thường, không tạo bản ghi mới và không coi là "lần chấm thứ hai".
- Giữ lịch sử nếu cần audit; bảng hiển thị chính chỉ đọc bản ghi của model version đang active.
- `received_at` do BE ghi. Output model hiện không có timestamp xử lý.

### 3.2. Chọn phiên bản đang hiển thị

BE cấu hình một `active_model_version` (và `active_feature_version` nếu cần). Khi đổi model:

- Kết quả cũ vẫn giữ để audit, nhưng không trộn `threshold` của version cũ với score của version mới.
- Nếu một giao dịch có kết quả từ nhiều version, UI chỉ hiển thị version active; các version khác nằm trong phần chi tiết/audit.

### 3.3. Khi chưa có hoặc không có kết quả

ML worker hiện dừng và không publish khi input sai, đọc history lỗi, hoặc gửi Kafka thất bại. Vì vậy BE phải có timeout và trạng thái thay vì chờ vô hạn:

| Tình huống | Trạng thái BE | Ghi chú |
| --- | --- | --- |
| Chưa nhận result | `pending` | Không hiển thị score 0 |
| Quá thời gian cấu hình | `failed` + `error_code` do BE định nghĩa | Ví dụ `ML_RESULT_TIMEOUT` |
| Result sai schema | `failed` + log | Không publish lên FE dữ liệu sai |
| Worker/input/history lỗi | `failed` | ML hiện không có error topic/DLQ; BE suy ra từ timeout/giám sát |
| Đã có result hợp lệ | `ready` | Hiển thị bình thường |

`error_code` là của BE, không phải trường trong output model. Không đưa stack trace, tên bảng, URI database lên FE.

## 4. Hợp đồng Backend → Frontend (đề xuất)

### 4.1. REST lấy trạng thái chấm điểm

```http
GET /api/v1/transactions/{transaction_id}/risk
```

`200` khi `ready`:

```json
{
  "transaction_id": "TX001",
  "status": "ready",
  "risk": {
    "score": 0.9821,
    "label": 1,
    "threshold": 0.9517361521720886,
    "model_version": "gatv2-device-control-v1",
    "feature_version": "schema-3-0b9efd44d793",
    "evaluated_at": "2026-10-23T10:15:30+07:00"
  }
}
```

`200` khi `pending`:

```json
{
  "transaction_id": "TX001",
  "status": "pending",
  "risk": null
}
```

`200` khi `failed`:

```json
{
  "transaction_id": "TX001",
  "status": "failed",
  "risk": null,
  "error_code": "ML_RESULT_TIMEOUT"
}
```

Quy ước:

- `status`: `pending | ready | failed` (BE có thể thêm `not_requested` nếu giao dịch chưa từng được gửi chấm).
- `risk` chỉ khác `null` khi `status=ready`.
- `evaluated_at` là thời điểm BE nhận/ghi kết quả, không phải timestamp trong output model.
- `error_code` ổn định, không chứa thông tin nội bộ.
- Nếu BE thêm `decision` (`allow | review | block`), đó là contract riêng của BE, phải có `policy_version`; không gộp vào `label` của model.

### 4.2. Push realtime (nếu FE cần)

Sau khi lưu result, BE phát một event qua SSE/WebSocket:

```json
{
  "type": "risk.updated",
  "data": {
    "transaction_id": "TX001",
    "status": "ready",
    "risk": {
      "score": 0.9821,
      "label": 1,
      "threshold": 0.9517361521720886,
      "model_version": "gatv2-device-control-v1",
      "feature_version": "schema-3-0b9efd44d793",
      "evaluated_at": "2026-10-23T10:15:30+07:00"
    }
  }
}
```

- FE subscribe theo `transaction_id` sau khi tạo giao dịch; event đến muộn vẫn cập nhật đúng giao dịch.
- FE dedupe theo `transaction_id + model_version + feature_version`; bỏ qua event cũ hơn `evaluated_at` nếu cùng khóa.
- Nếu không có push, FE polling endpoint trên với backoff; BE vẫn là nguồn sự thật.

## 5. Hiển thị trên giao diện

### 5.1. Bảng ánh xạ

| Dữ liệu | Cách hiển thị | Không được hiển thị |
| --- | --- | --- |
| `status=pending` | "Đang chấm điểm…" + skeleton/spinner | Score `0`, badge "An toàn" |
| `status=failed` | "Không thể chấm điểm" + mã lỗi rút gọn + nút thử lại (nếu BE hỗ trợ) | Score `0`, "Gian lận", "An toàn" |
| `status=ready`, `label=1` | Badge "Cần xem xét" / "Vượt ngưỡng" theo màu cảnh báo | "Gian lận" như kết luận cuối |
| `status=ready`, `label=0` | Badge "Không vượt ngưỡng" / "Bình thường" | "An toàn tuyệt đối" |
| `score` | "Điểm rủi ro: 0.982" hoặc thanh 0–100 ghi rõ "điểm", kèm vạch ngưỡng | "Xác suất gian lận 98.2%" |
| `threshold` | Vạch ngưỡng + tooltip "Ngưỡng mô hình: 0.952" | Ẩn ngưỡng khi đang hiển thị label |
| `model_version`, `feature_version` | Tooltip/trang chi tiết/audit | Đưa lên badge chính |
| `evaluated_at` | "Chấm lúc …" ở phần chi tiết | Dùng làm thời gian giao dịch |

### 5.2. View model gợi ý cho FE

BE có thể trả trực tiếp contract trên, hoặc chuẩn hóa thêm cho FE:

```json
{
  "transactionId": "TX001",
  "state": "ready",
  "scoreText": "0.982",
  "scorePercent": 98.2,
  "thresholdPercent": 95.17,
  "isAboveThreshold": true,
  "labelText": "Cần xem xét",
  "tone": "danger",
  "modelVersion": "gatv2-device-control-v1",
  "featureVersion": "schema-3-0b9efd44d793",
  "evaluatedAt": "2026-10-23T10:15:30+07:00"
}
```

- `labelText`/`tone` do BE quyết định theo policy, không do ML.
- `scorePercent` chỉ nên dùng khi UI ghi rõ là thang điểm; không đổi nhãn thành xác suất.
- FE không tự suy ra `isAboveThreshold` từ `score` nếu BE đã trả `label`/`decision`, để tránh lệch ngưỡng khi model đổi.

### 5.3. Trạng thái UI tối thiểu

```text
pending  → đang chấm điểm, có thể timeout
ready    → hiển thị score + label + threshold
failed   → lỗi chấm điểm, không có score
```

Không dùng chung một màu "thành công" cho `label=0` và `failed`; hai trạng thái này khác nhau về nghiệp vụ và vận hành.

## 6. Tương thích và phiên bản hợp đồng

- Hợp đồng hiện tại: `model-response.v1`.
- **Thay đổi tương thích:** thêm trường mới. Producer cập nhật schema; consumer phải bỏ qua trường lạ và chỉ validate các trường bắt buộc. Schema JSON đặt `additionalProperties: false` cho output của producer, không dùng nó để từ chối message ở consumer.
- **Thay đổi không tương thích:** xóa/đổi tên trường, đổi kiểu, đổi ý nghĩa `risk_score`/`label`/`threshold` → tạo `v2`, topic mới hoặc header/schema version mới; không sửa lặng lẽ trên `v1`.
- `predicted_label` phải giữ kiểu số nguyên `0/1`, không đổi thành boolean/chuỗi.
- Không đổi `threshold` trong cùng `model_version`.
- Consumer không được giả định chỉ có 6 trường mãi mãi; producer không được bỏ trường bắt buộc trong `v1`.

## 7. Bảo mật và dữ liệu

- Output chỉ chứa ID giao dịch và metadata model; không chứa số tài khoản, số tiền, device, auth, PII.
- Kafka topic là nội bộ; TLS/SASL, ACL, mã hóa at-rest do hạ tầng/BE cấu hình, không nằm trong payload.
- Log BE không ghi toàn bộ request gốc nếu có PII; chỉ log `transaction_id`, version, score, trạng thái.
- FE chỉ hiển thị các trường đã được BE cho phép; không nhận trực tiếp output Kafka.

## 8. Checklist kiểm thử tích hợp

- [ ] Output đúng `specs/model_response.v1.json`; không thiếu/thừa trường; `label` đúng `score >= threshold`.
- [ ] Gửi trùng cùng key: BE chỉ có một bản ghi, không nhân đôi hiển thị.
- [ ] Worker restart/replay: score ổn định với cùng lịch sử; message trùng vẫn idempotent.
- [ ] Không có result: FE hiển thị `pending`, sau timeout chuyển `failed`, không hiện score 0.
- [ ] Đổi `model_version`: UI chỉ hiển thị version active; threshold cũ không bị trộn.
- [ ] `label=1` không hiển thị thành "Gian lận" như kết luận cuối; `label=0` không hiển thị "An toàn tuyệt đối".
- [ ] Score không được gọi là xác suất; ngưỡng luôn hiển thị/audit cùng score.
- [ ] Message lạ/trường lạ không làm consumer crash.
- [ ] Test hợp đồng: `pytest -q tests/test_response_contract.py`.

## 9. Giới hạn hiện tại và bước tiếp theo

Đã có:

- Output 6 trường, schema JSON, validator runtime, test hợp đồng.
- Kafka key chuẩn hóa qua `response_key()`.
- Tài liệu mapping BE/FE.

Chưa có trong repo này:

- HTTP API (FastAPI/Uvicorn) cho BE gọi đồng bộ.
- Consumer Kafka phía Go, lưu DB, REST/SSE/WebSocket cho FE.
- Error topic/DLQ, exactly-once, timestamp xử lý trong payload.
- Hiệu chỉnh xác suất, lý do/giải thích, `criterion_results`.
- Đo latency online và kiểm chứng trên hạ tầng thật (Kafka/EventStoreDB/MongoDB/Go/FE).

Nếu triển khai FastAPI sau này, giữ nguyên schema response ở trên và bổ sung:

```http
POST /v1/predict
Content-Type: application/json

<raw transaction hoặc canonical/native request event>
```

- `200`: đúng `model-response.v1`.
- `400`: input sai contract (không có nhãn/risk fields).
- `503`: history/model chưa sẵn sàng.
- `500`: lỗi không mong đợi.

Endpoint HTTP chỉ là adapter; không thay đổi ý nghĩa các trường output.

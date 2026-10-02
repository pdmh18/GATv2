# Fraud Detection Pipeline

Khung project theo sơ đồ 10 khối: Event & Pipeline -> Tiền xử lý -> Feature Engineering -> Chia tập -> Train -> Tuning -> Đánh giá -> So sánh -> Lưu model -> Inference.

## Trạng thái hiện tại
- Đây mới là **khung project**, làm dần từng khối.
- `scripts/run_pipeline.py --stage ...` hiện **chỉ in log**, chưa gọi các module thật. Cần nối trong `src/fraud_pipeline/pipeline.py` khi làm từng khối (ví dụ `stage_ingest()` chưa gọi `ingestion/`).
- **Chưa chốt thuật toán**: `models/`, `tuning/`, `registry/` mới là khung trống; `configs/config.yaml` để `model.name` và `tuning.method` là `null`.
- `features/graph.py` và `ingestion/simulator.py` (nhãn fraud đang đặt cứng = 0) còn `TODO`.
- `evaluation/compare.py`: chưa có model baseline (phương pháp truyền thống) để so sánh. Trong `run_training_loop()` đang đặt cứng `reached = True`.

| Khối | Trạng thái |
|---|---|
| 1. Event & Pipeline | Chưa nối, có code khung |
| 2. Khảo sát & tiền xử lý | Chưa nối, có code khung |
| 3. Feature Engineering | Chưa nối, có code khung |
| 4. Chia tập | Chưa nối, có code khung |
| 5. Train | Khung trống |
| 6. Tuning | Khung trống |
| 7-8. Đánh giá, so sánh | Chưa nối, có code khung |
| 9. Lưu model | Khung trống |
| 10. Inference | Chưa nối, có code khung |

## Cấu trúc project
```
fraud-detection-pipeline/
├── configs/            # config.yaml: tham số chung
├── data/               # raw -> interim -> processed -> splits
├── models/             # nơi lưu file model đã train
├── notebooks/          # notebook (EDA...), đang trống
├── reports/            # biểu đồ, bảng kết quả
├── scripts/            # run_pipeline.py
├── tests/              # kiểm tra tự động (pytest)
├── src/fraud_pipeline/ # CODE của cả 10 khối
├── .env.example  .gitignore  docker-compose.yml  Makefile
└── pyproject.toml  README.md  requirements.txt
```

## Vị trí code của từng khối (đều nằm trong `src/fraud_pipeline/`)
| Khối | Nội dung | Vị trí | Dữ liệu vào -> ra |
|---|---|---|---|
| 1 | Event & Pipeline | `ingestion/` | simulator -> Kafka -> EventStoreDB -> Pandas -> MongoDB |
| 2 | Khảo sát & tiền xử lý | `preprocessing/` (`eda.py`, `cleaning.py`) | `data/raw` -> `data/interim` |
| 3 | Feature Engineering | `features/` (`transaction`, `behavioral`, `temporal`, `graph`, `build_features`) | `data/interim` -> `data/processed` |
| 4 | Chia Train/Val/Test | `data_split.py` (`time_split()`) | `data/processed` -> `data/splits` |
| 5 | Xây dựng & Train | `models/` (`base.py`, `trainer.py`) | `data/splits` -> `models/` |
| 6 | Hyperparameter Tuning | `tuning/tuner.py` | nằm trong vòng lặp 5-8 |
| 7 | Đánh giá | `evaluation/metrics.py`, `plots.py` | kết quả -> `reports/` |
| 8 | So sánh kết quả | `evaluation/compare.py` | kết quả -> `reports/` |
| 9 | Lưu model | `registry/save_load.py` | -> `models/` |
| 10 | Inference | `inference/` (`predictor.py` batch, `stream_inference.py` real-time) | dữ liệu mới -> risk score |

Hai file không thuộc khối nào: `config.py` (đọc cấu hình) và `pipeline.py` (điều phối các khối, chứa vòng lặp "chưa đạt -> chỉnh model/tham số" ở `run_training_loop`). Các khối 5-8 nằm trong vòng lặp này; khối 9 chỉ chạy khi kết quả đã đạt.

## Các thư mục và file ngoài `src/` dùng để làm gì
| Tên | Vai trò |
|---|---|
| `configs/` | `config.yaml`: tỉ lệ chia tập, topic Kafka, ngưỡng đạt (`target_metrics`)... Đổi thông số ở đây, không sửa code. |
| `data/` | Dữ liệu qua từng bước: `raw` (thô), `interim` (sau tiền xử lý), `processed` (sau feature), `splits` (train/val/test). |
| `models/` | Lưu file model đã train (khối 9). Khác với `src/fraud_pipeline/models/` là nơi chứa code. |
| `notebooks/` | Notebook, ví dụ vẽ biểu đồ EDA ở khối 2. Không cần có thể xóa. |
| `reports/` | Biểu đồ ROC, bảng so sánh (khối 7, 8). |
| `scripts/` | `run_pipeline.py`: chạy pipeline bằng lệnh. |
| `tests/` | Test tự động, hiện có test kiểm tra chia tập không bị leakage. Chạy bằng `pytest`. |
| `.env.example` | Mẫu địa chỉ kết nối Kafka, EventStoreDB, MongoDB. Copy thành `.env` rồi điền thông tin thật. |
| `.gitignore` | Báo Git bỏ qua `.env`, dữ liệu, model nặng. |
| `docker-compose.yml` | Dựng Kafka, EventStoreDB, MongoDB bằng `docker compose up -d` (cho khối 1). |
| `Makefile` | Lệnh tắt (`make train`...). Windows không cài `make` thì bỏ qua được. |
| `pyproject.toml` | Khai báo project, để Python nhận ra thư mục `src/`. |
| `requirements.txt` | Danh sách thư viện: `pip install -r requirements.txt`. |

## Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d             # Kafka, EventStoreDB, MongoDB
```

## Chạy từng khối
```bash
python scripts/run_pipeline.py --stage ingest      # Khối 1
python scripts/run_pipeline.py --stage preprocess  # Khối 2
python scripts/run_pipeline.py --stage features    # Khối 3
python scripts/run_pipeline.py --stage split       # Khối 4
python scripts/run_pipeline.py --stage train       # Khối 5
python scripts/run_pipeline.py --stage tune        # Khối 6
python scripts/run_pipeline.py --stage evaluate    # Khối 7
python scripts/run_pipeline.py --stage compare     # Khối 8
python scripts/run_pipeline.py --stage save        # Khối 9
python scripts/run_pipeline.py --stage infer       # Khối 10
python scripts/run_pipeline.py --stage all
```
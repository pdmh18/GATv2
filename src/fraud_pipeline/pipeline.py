"""Điều phối 10 bước + vòng lặp "chưa đạt -> điều chỉnh"."""
from .config import load_config

cfg = load_config()

def stage_ingest():     print("[1] Ingest: simulator -> Kafka -> EventStoreDB -> Pandas -> MongoDB")
def stage_preprocess(): print("[2] Khảo sát & tiền xử lý")
def stage_features():   print("[3] Feature engineering")
def stage_split():      print("[4] Chia Train/Val/Test")
def stage_train():      print("[5] Train model")
def stage_tune():       print("[6] Hyperparameter tuning")
def stage_evaluate():   print("[7] Đánh giá")
def stage_compare():    print("[8] So sánh kết quả")
def stage_save():       print("[9] Lưu model")
def stage_infer():      print("[10] Inference")

STAGES = {
    "ingest": stage_ingest, "preprocess": stage_preprocess, "features": stage_features,
    "split": stage_split, "train": stage_train, "tune": stage_tune,
    "evaluate": stage_evaluate, "compare": stage_compare,
    "save": stage_save, "infer": stage_infer,
}

def run_training_loop():
    """Bước 5-8, lặp lại nếu kết quả chưa đạt yêu cầu."""
    for i in range(cfg["target_metrics"]["max_iterations"]):
        stage_train(); stage_tune(); stage_evaluate(); stage_compare()
        reached = True  # TODO: gọi compare.meets_target(...)
        if reached:
            break
        print(f"Chưa đạt, lặp lần {i + 2}")

"""Điều phối 10 bước + vòng lặp "chưa đạt -> điều chỉnh"."""
from .config import load_config, project_path
from .split_export import export_splits

def stage_ingest():     print("[1] Ingest: simulator -> Kafka -> EventStoreDB -> Pandas -> MongoDB")
def stage_preprocess(): print("[2] Khảo sát & tiền xử lý")
def stage_features(config_path='configs/dataset.json', *, data=None, output=None, validation_report=None, overwrite=False):
    from .features.export import export_features
    cfg = load_config(config_path)
    dataset = cfg['dataset']
    result = export_features(
        project_path(dataset['source']) if data is None else data,
        project_path(dataset['validation_report']) if validation_report is None else validation_report,
        project_path(dataset['output']),
        project_path(dataset['features_output']) if output is None else output,
        cfg['features'], overwrite=overwrite)
    print(f"Features: {result['request_count']} requests, {result['node_count']} nodes, "
          f"{len(result['feature_names'])} transaction features, {result['graph_event_count']} graph events")
    return result
def stage_split(config_path='configs/dataset.json', *, data=None, output=None, validation_report=None, overwrite=False):
    cfg = load_config(config_path)
    dataset = cfg['dataset']
    result = export_splits(
        project_path(dataset['source']) if data is None else data,
        project_path(dataset['validation_report']) if validation_report is None else validation_report,
        project_path(dataset['output']) if output is None else output,
        cfg['split'], overwrite=overwrite)
    for name, summary in result['splits'].items():
        print(f"{name}: {summary['count']} samples, {summary['anomaly']} anomalies")
    return result
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
    cfg = load_config()
    for i in range(cfg["target_metrics"]["max_iterations"]):
        stage_train(); stage_tune(); stage_evaluate(); stage_compare()
        reached = True  # TODO: gọi compare.meets_target(...)
        if reached:
            break
        print(f"Chưa đạt, lặp lần {i + 2}")

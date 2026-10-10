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
def stage_train(config_path='configs/dataset.json', *, output=None, overwrite=False):
    from .models.trainer import train_run
    cfg=load_config(config_path)
    return train_run(project_path(cfg['dataset']['features_output']),
                     project_path(cfg['training']['output']) if output is None else output,
                     cfg['training'],overwrite=overwrite)

def run_dataset_training(config_path='configs/dataset.json', *, data=None, validation_report=None, output=None, overwrite=False):
    """Validated raw dataset → splits → causal tensors → checkpoint, reusing exact artifacts."""
    import json
    from .ingestion.request_loader import verify_validation, file_hash
    from .validation import validate_dataset
    from .validation.report import markdown
    from .features.dataset import TemporalDataset
    from .features.export import export_features
    from .models.trainer import train_run
    cfg=load_config(config_path)
    source=project_path(cfg['dataset']['source']) if data is None else data.resolve()
    report=project_path(cfg['dataset']['validation_report']) if validation_report is None else validation_report.resolve()
    splits=project_path(cfg['dataset']['output'])
    features=project_path(cfg['dataset']['features_output'])
    if source==report.parent or source in report.parents:
        raise ValueError('Validation report must be outside raw inputs')
    if not report.exists() or overwrite:
        audit=validate_dataset(source)
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(audit,indent=2)+'\n')
        report.with_suffix('.md').write_text(markdown(audit))
    _,hashes,_=verify_validation(source,report)
    if not splits.exists() or overwrite:
        export_splits(source,report,splits,cfg['split'],overwrite=overwrite)
    else:
        split_manifest=json.loads((splits/'manifest.json').read_text())
        if (split_manifest['source']['directory']!=str(source.resolve()) or split_manifest['source']['sha256']!=hashes
            or split_manifest['split_rule']['ratios']!=cfg['split']
            or split_manifest['validation']['report_sha256']!=file_hash(report)):
            raise ValueError('Existing splits are stale; use a new run or --overwrite')
        for value in split_manifest['splits'].values():
            if file_hash(splits/value['file'])!=value['sha256']:
                raise ValueError('Split checksum mismatch')
    if not features.exists() or overwrite:
        export_features(source,report,splits,features,cfg['features'],overwrite=overwrite)
    else:
        artifact=TemporalDataset(features).manifest
        if artifact['source']!=str(source.resolve()) or artifact['source_sha256']!=hashes or artifact['options']!=cfg['features']:
            raise ValueError('Existing features use a different dataset or policy')
        for path,digest in artifact['input_sha256'].items():
            if file_hash(project_path(path))!=digest:
                raise ValueError('Existing feature provenance is stale')
        for name,digest in artifact['code_sha256'].items():
            if file_hash(project_path('src/fraud_pipeline/features')/name)!=digest:
                raise ValueError('Feature context code differs from artifact; re-export features')
    print('Validated dataset, split and feature artifacts; starting training',flush=True)
    return train_run(features,project_path(cfg['training']['output']) if output is None else output,
                     cfg['training'],overwrite=overwrite)
def stage_tune():       print("[6] Hyperparameter tuning")
def stage_evaluate(config_path='configs/dataset.json'):
    from .evaluation.phase5 import run
    return run(load_config(config_path), project_path('.'))
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
    """Run configured training; tuning and comparison are separate Phase 5 work."""
    return stage_train()

"""Resolve artifact defaults from the dataset configuration."""
from pathlib import Path
from .config import load_config, project_path


def defaults():
    config = load_config('configs/dataset.json')
    dataset = config['dataset']
    run_root = project_path(dataset['run_root'])
    seed_root = run_root / f"seed-{dataset['seed']}"
    return {'run_root': run_root, 'source': project_path(dataset['source']),
            'validation': project_path(dataset['validation_report']),
            'splits': project_path(dataset['output']),
            'baseline': seed_root / 'baseline', 'eda': seed_root / 'eda',
            'multiseed': run_root / 'multiseed.json'}


def seed_paths(run_root, seed):
    root = Path(run_root) / f'seed-{seed}'
    return {name: root / name for name in ('raw', 'splits', 'validation', 'baseline', 'eda')}

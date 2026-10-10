#!/usr/bin/env python3
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.evaluation.splits import evaluate_splits
from fraud_pipeline.evaluation.split_plots import export_split_plots
config=json.loads((ROOT/'configs/dataset.json').read_text())
run=ROOT/config['evaluation']['output']
if not (run/'gatv2_partitions/report.json').exists():
    evaluate_splits(config,ROOT)
print(export_split_plots(run))

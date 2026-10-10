#!/usr/bin/env python3
"""Run validation-first baselines and final frozen test evaluation."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.evaluation.phase5 import run
parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,default=ROOT/'configs/dataset.json')
args=parser.parse_args()
result=run(json.loads(args.config.read_text()),ROOT)
print(json.dumps({'selected_model':result['selected_model'],'status':result['status']},indent=2))

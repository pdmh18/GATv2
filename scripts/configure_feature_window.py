#!/usr/bin/env python3
"""Write a versioned config before exporting/training a new feature window."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.config import load_config
from fraud_pipeline.features.windows import window_profile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',default='configs/features-v3.json')
parser.add_argument('--output',type=Path,required=True)
for key in ('window','amount-window','median-window','graph-window','node-window'):
    parser.add_argument('--'+key)
args=parser.parse_args()
try:
    config=window_profile(load_config(args.base),**{k:getattr(args,k) for k in ('window','amount_window','median_window','graph_window','node_window')})
    if args.output.exists():parser.error('Config exists; choose a new output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,indent=2)+'\n')
    print(args.output)
except ValueError as exc:parser.error(str(exc))

#!/usr/bin/env python3
"""Export ROC, PR, confusion and diagnostic charts from saved predictions."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_pipeline.evaluation.plots import export_plots
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=ROOT/'outputs/contextual-v4/seed-20261002/evaluation/phase5-v1')
args=parser.parse_args()
if (args.run/'gatv2_partitions/report.json').exists():
    from fraud_pipeline.evaluation.split_plots import export_split_plots
    print(export_split_plots(args.run))
else:
    print(export_plots(args.run))

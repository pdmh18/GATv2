#!/usr/bin/env python3
"""Export a GATv2 checkpoint and validation threshold, without training or test scoring."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.registry.bundle import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-version', required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.training, args.selection, args.output,
                                 model_version=args.model_version), indent=2))


if __name__ == '__main__':
    main()

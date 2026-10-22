#!/usr/bin/env python3
"""Replay one request or start the configured Kafka GATv2 worker."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.inference.history import from_config
from fraud_pipeline.inference.stream_inference import handle_message, load_inference_config, run
from fraud_pipeline.inference.predictor import ReplayPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/inference.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request', type=Path, help='JSON raw transaction or canonical/native request event')
    mode.add_argument('--kafka', action='store_true')
    args = parser.parse_args()
    config = load_inference_config(args.config)
    if args.kafka:
        run(config)
    else:
        predictor = ReplayPredictor(config['bundle'], device=config.get('device', 'cpu'))
        result = handle_message(predictor, from_config(config['history']), json.loads(args.request.read_text()))
        if result is None:
            parser.error('The requested event is not a transaction request')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

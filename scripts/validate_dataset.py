"""Audit the sibling generator dataset without training dependencies."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fraud_pipeline.validation import validate_dataset
from fraud_pipeline.validation.report import markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'anomaly-data-pipeline/data/generated')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/dataset-validation')
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('Output must differ from input directory')
    report = validate_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'validation.md').write_text(markdown(report), encoding='utf-8')
    print(f"{report['status']}: {report['error_count']} errors, {report['warning_count']} warnings")
    print(f"Report: {args.output / 'validation.md'}")
    return 1 if report['error_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

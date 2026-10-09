import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud_pipeline.pipeline import STAGES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=[*STAGES, 'all'])
    parser.add_argument('--config', default='configs/dataset.json', help='Phase 2 dataset config (paths relative to repo root)')
    parser.add_argument('--data', type=Path, help='Override dataset source directory')
    parser.add_argument('--output', type=Path, help='Override split output directory')
    parser.add_argument('--validation-report', type=Path, help='Successful Phase 1 report for this dataset')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing split dataset, preserving previous directory')
    args = parser.parse_args()
    try:
        if args.stage == 'split':
            STAGES['split'](args.config, data=args.data, output=args.output,
                            validation_report=args.validation_report, overwrite=args.overwrite)
        elif args.stage == 'all':
            parser.error('--stage all is not implemented end-to-end; use --stage split for Phase 2')
        else:
            if any([args.data, args.output, args.validation_report, args.overwrite]):
                parser.error('Dataset/output options currently apply only to split or all')
            STAGES[args.stage]()
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Pipeline failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

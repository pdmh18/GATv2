import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud_pipeline.pipeline import STAGES, run_dataset_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=[*STAGES, 'all'])
    parser.add_argument('--config', default='configs/dataset.json', help='Dataset config (paths relative to repo root)')
    parser.add_argument('--data', type=Path, help='Override dataset source directory')
    parser.add_argument('--output', type=Path, help='Override output directory of selected stage')
    parser.add_argument('--validation-report', type=Path, help='Successful Phase 1 report for this dataset')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing split dataset, preserving previous directory')
    args = parser.parse_args()
    try:
        if args.stage in {'split', 'features'}:
            STAGES[args.stage](args.config, data=args.data, output=args.output,
                            validation_report=args.validation_report, overwrite=args.overwrite)
        elif args.stage == 'train':
            if args.data or args.validation_report:
                parser.error('Train reads feature artifact from config; raw dataset overrides apply to split/features')
            STAGES['train'](args.config,output=args.output,overwrite=args.overwrite)
        elif args.stage == 'evaluate':
            if any([args.data, args.output, args.validation_report, args.overwrite]):
                parser.error('Evaluation paths are configured in dataset/evaluation; use a new evaluation.output for a new run')
            STAGES['evaluate'](args.config)
        elif args.stage == 'all':
            run_dataset_training(args.config,data=args.data,validation_report=args.validation_report,
                                 output=args.output,overwrite=args.overwrite)
        else:
            if any([args.data, args.output, args.validation_report, args.overwrite]):
                parser.error('Dataset/output options currently apply only to split or features')
            STAGES[args.stage]()
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Pipeline failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

import argparse,gc,json,sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.artifacts import defaults, seed_paths
parser=argparse.ArgumentParser(description='Generate, validate, split and assess contextual datasets across seeds')
parser.add_argument('--seeds', type=int, nargs='+', default=[20261002,20261003,20261004])
parser.add_argument('--generator-root', type=Path, default=ROOT.parent/'anomaly-data-pipeline')
parser.add_argument('--output-root', type=Path, default=defaults()['run_root'], help='Version/run directory containing seed directories')
args=parser.parse_args()
GEN=args.generator_root.resolve()
sys.path[:0]=[str(ROOT/'src'),str(GEN/'src'),str(ROOT/'scripts')]
from anomaly_data_pipeline.config import load_config,load_pipeline_config
from anomaly_data_pipeline.generation.pipeline import generate
from fraud_pipeline.validation import validate_dataset
from fraud_pipeline.validation.report import markdown
from fraud_pipeline.split_export import export_splits
from assess_dataset import assess
ratios=json.loads((ROOT/'configs/dataset.json').read_text())['split']
base=load_config(GEN/'configs/base.yaml');pipeline=load_pipeline_config(GEN/'configs/pipeline.toml')
results=[]
for seed in args.seeds:
 paths=seed_paths(args.output_root, seed)
 source, reports, splits=paths['raw'], paths['validation'], paths['splits']
 if (args.output_root/f'seed-{seed}').exists():raise SystemExit('Refusing existing run: '+str(seed))
 print(f'[{seed}] generating {base.customers * base.transactions_per_customer} requests',flush=True)
 counts=generate(base.model_copy(update={'seed':seed}),source,pipeline)
 print(f'[{seed}] generated {counts}',flush=True);gc.collect()
 print(f'[{seed}] validating contracts, ledger and causal scenario witnesses',flush=True)
 report=validate_dataset(source);reports.mkdir(parents=True,exist_ok=True)
 (reports/'validation.json').write_text(json.dumps(report,indent=2)+'\n');(reports/'validation.md').write_text(markdown(report))
 print(f'[{seed}] audit {report["status"]}: {report["error_count"]} errors',flush=True)
 if report['error_count']:raise SystemExit('Audit failed: '+str(seed))
 del report;gc.collect()
 print(f'[{seed}] exporting chronological splits',flush=True)
 exported=export_splits(source,reports/'validation.json',splits,ratios)
 print(f'[{seed}] split counts '+str({k:v['count'] for k,v in exported['splits'].items()}),flush=True)
 print(f'[{seed}] assessing causal diagnostic rules',flush=True)
 result=assess(splits,reports/'validation.json',paths['baseline'])
 results.append(result);gc.collect()
 print(f'[{seed}] complete',flush=True)
summary=args.output_root/'multiseed.json'
summary.write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(args.seeds)} datasets audited/split/assessed. '+str(summary),flush=True)

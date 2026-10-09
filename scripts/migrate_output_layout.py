"""One-time migration of legacy local outputs; defaults to a dry run."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fraud_pipeline.ingestion.request_loader import file_hash, verify_validation
from fraud_pipeline.split_export import split_summary
from fraud_pipeline.validation.report import markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--resume-metadata', action='store_true', help='Finish provenance updates after files were moved')
    args = parser.parse_args()
    mappings = []
    for version in ('contextual-v3', 'contextual-v4'):
        for seed in (20261002, 20261003, 20261004):
            new = ROOT / 'outputs' / version / f'seed-{seed}'
            mappings.extend([(ROOT / 'data/raw' / version / f'seed-{seed}', new / 'raw'),
                             (ROOT / 'data/splits' / version / f'seed-{seed}', new / 'splits')])
            old = ROOT / 'reports' / version / f'seed-{seed}'
            for name in ('validation.json', 'validation.md'):
                mappings.append((old / name, new / 'validation' / name))
            mappings.append((old / 'assessment.json', new / 'baseline/assessment.json'))
        for name in ('multiseed.json', 'comparison.json', 'hour-distribution.json'):
            mappings.append((ROOT / 'reports' / version / name, ROOT / 'outputs' / version / name))
    mappings.extend([(ROOT / 'reports/eda', ROOT / 'outputs/contextual-v3/seed-20261002/eda'),
                     (ROOT / 'reports/eda-contextual-v4', ROOT / 'outputs/contextual-v4/seed-20261002/eda')])
    # Retain experiments, duplicate exports and old defaults in a separate archive.
    archive = ROOT / 'outputs/archive'
    smoke = Path('/tmp/anomaly-contextual-generator/smoke')
    if smoke.exists():
        mappings.append((smoke, archive / 'data/raw/contextual-smoke'))
    for parent in ('data/raw', 'data/splits', 'reports'):
        directory = ROOT / parent
        for child in directory.iterdir():
            if child.name.startswith('.') or child.name in {'contextual-v3', 'contextual-v4', 'eda', 'eda-contextual-v4'}:
                continue
            mappings.append((child, archive / parent / child.name))
    mappings.extend([(ROOT / 'reports/contextual-v3/old-baseline', archive / 'contextual-v3/old-baseline'),
                     (ROOT / 'reports/contextual-v4/previous-default-validation.json', archive / 'contextual-v4/previous-default-validation.json'),
                     (ROOT / 'reports/contextual-v4/promote-validation.json', archive / 'contextual-v4/promote-validation.json')])
    mappings = [(a, b) for a, b in mappings if a.exists()]
    if args.resume_metadata:
        mappings = []
    for old, new in mappings:
        if new.exists():
            raise SystemExit(f'Refusing existing destination: {new}')
        print(old, '→', new.relative_to(ROOT))
    if not args.apply:
        return
    journal = archive / 'migration-originals'
    if journal.exists() and not args.resume_metadata:
        raise SystemExit('Migration already applied')
    for old, new in mappings:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    # Identify relocated sources by immutable dataset manifest checksum.
    sources = {}
    for path in sorted((ROOT / 'outputs').glob('contextual-v*/seed-*/raw')):
        sources[file_hash(path / 'manifest.json')] = path
    for path in [archive / 'data/raw/contextual-smoke', *sorted((ROOT.parent / 'anomaly-data-pipeline/data').glob('generated*'))]:
        if (path / 'manifest.json').exists():
            sources.setdefault(file_hash(path / 'manifest.json'), path)
    audits = {}
    changes = []
    def rewrite(path, content):
        relative = path.relative_to(ROOT / 'outputs')
        original = journal / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists():
            shutil.copy2(path, original)
        before = file_hash(path)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
        changes.append({'file': str(relative), 'before': before, 'after': file_hash(path)})
    for path in sorted((ROOT / 'outputs').rglob('*.json')):
        if journal in path.parents:
            continue
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or 'dataset' not in report or 'sha256' not in report:
            continue
        source = sources.get(report['sha256'].get('manifest.json'))
        if source is None:
            raise RuntimeError(f'Cannot find source for {path}')
        report['dataset'] = str(source.resolve())
        rewrite(path, report)
        verify_validation(source, path)
        audits.setdefault(report['sha256']['manifest.json'], path)
        if archive not in path.parents:
            audits[report['sha256']['manifest.json']] = path
        companion = path.with_suffix('.md')
        if companion.exists():
            companion.write_text(markdown(report))
    for path in sorted((ROOT / 'outputs').rglob('manifest.json')):
        if journal in path.parents:
            continue
        report = json.loads(path.read_text())
        if 'source' not in report or not isinstance(report['source'], dict):
            continue
        digest = report['source']['sha256']['manifest.json']
        source, audit = sources.get(digest), audits.get(digest)
        if source is None:
            raise RuntimeError(f'Cannot resolve split provenance: {path}')
        for name, expected in report['source']['sha256'].items():
            if file_hash(source / name) != expected:
                raise RuntimeError(f'Source checksum mismatch: {path}: {name}')
        for stats in report['splits'].values():
            if file_hash(path.parent / stats['file']) != stats['sha256']:
                raise RuntimeError(f'Split checksum mismatch: {path}')
        report['source']['directory'] = str(source.resolve())
        if audit is not None:
            report['validation']['report_sha256'] = file_hash(audit)
        else:
            report['validation']['migration_note'] = 'Original validation checksum retained; historical audit file unavailable. Source and split checksums verified.'
        rewrite(path, report)
        (path.parent / 'summary.md').write_text(split_summary(report))
    (archive / 'migration.json').write_text(json.dumps({'moves': [[str(a), str(b)] for a, b in mappings],
                                                       'metadata_changes': changes}, indent=2) + '\n')
    # Remove only directories that are now empty.
    for parent in ('data/raw', 'data/splits', 'reports'):
        for path in sorted((ROOT / parent).rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    print('Migration complete; source and split checksums verified.')


if __name__ == '__main__':
    main()

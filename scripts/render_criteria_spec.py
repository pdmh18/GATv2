#!/usr/bin/env python3
"""Render/check the catalog section in the criteria specification from its JSON source."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_pipeline.evaluation.criteria_audit import load_catalog, render_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT/'specs/Bo_tieu_chi_phat_hien_giao_dich_bat_thuong.md'
    start, end = '<!-- CRITERIA_CATALOG_START -->', '<!-- CRITERIA_CATALOG_END -->'
    before, remaining = path.read_text().split(start)
    _, after = remaining.split(end)
    rendered = before+start+'\n\n'+render_catalog(load_catalog(ROOT/'specs/anomaly_criteria.v2.json'))+'\n'+end+after
    if args.check:
        if path.read_text() != rendered:
            parser.error('Criteria markdown differs from catalog; rerun without --check')
    else:
        path.write_text(rendered)


if __name__ == '__main__':
    main()

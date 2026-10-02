import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_pipeline.pipeline import STAGES, run_training_loop

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
a = ap.parse_args()

if a.stage == "all":
    for n in ("ingest", "preprocess", "features", "split"):
        STAGES[n]()
    run_training_loop()
    STAGES["save"](); STAGES["infer"]()
else:
    STAGES[a.stage]()

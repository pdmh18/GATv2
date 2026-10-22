STAGES = ingest preprocess features split train tune evaluate compare save infer
PYTHON ?= python3
CONFIG ?= configs/dataset.json
.PHONY: $(STAGES) all test setup-training
$(filter-out infer,$(STAGES)):
	$(PYTHON) scripts/run_pipeline.py --stage $@ --config "$(CONFIG)"
all:
	$(PYTHON) scripts/run_pipeline.py --stage all --config "$(CONFIG)"
setup-training:
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-training.txt
test:
	$(PYTHON) -m pytest -q

.PHONY: plot-evaluation
plot-evaluation:
	$(PYTHON) scripts/plot_evaluation.py

GENERATOR_ROOT ?= ../anomaly-data-pipeline
DATASET_RAW ?= outputs/contextual-v4/seed-20261002/raw
SEED ?= 20261002
.PHONY: generate-dataset clean-data
# Generate raw inputs in this repo; then run split/features/train with the configured environment.
generate-dataset:
	$(MAKE) -C "$(GENERATOR_ROOT)" generate CONFIG=configs/base.yaml PIPELINE=configs/pipeline.toml OUTPUT="$(abspath $(DATASET_RAW))" SEED="$(SEED)"
# Preview: python scripts/clean_generated_data.py
clean-data:
	$(PYTHON) scripts/clean_generated_data.py --apply

.PHONY: feature-eda
feature-eda:
	$(PYTHON) scripts/audit_features.py --config "$(CONFIG)" --temporary-rebuild

EDA_CONFIG ?= configs/criteria-eda-v2.json
.PHONY: dataset-eda criteria-spec-check
dataset-eda:
	$(PYTHON) scripts/run_eda.py --config "$(EDA_CONFIG)"
criteria-spec-check:
	$(PYTHON) scripts/render_criteria_spec.py --check

WINDOW ?= 1w
WINDOW_CONFIG ?= outputs/configs/window-$(WINDOW).json
.PHONY: configure-window
configure-window:
	$(PYTHON) scripts/configure_feature_window.py --window "$(WINDOW)" --output "$(WINDOW_CONFIG)"

.PHONY: train-device-experiment
train-device-experiment:
	$(PYTHON) scripts/train_device_experiment.py --config configs/device-v1.json

INFERENCE_CONFIG ?= configs/inference.json
.PHONY: inference-worker
inference-worker:
	$(PYTHON) scripts/run_inference.py --config "$(INFERENCE_CONFIG)" --kafka
infer: inference-worker

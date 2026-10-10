STAGES = ingest preprocess features split train tune evaluate compare save infer
PYTHON ?= python3
CONFIG ?= configs/dataset.json
.PHONY: $(STAGES) all test setup-training
$(STAGES):
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

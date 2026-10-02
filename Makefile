STAGES = ingest preprocess features split train tune evaluate compare save infer
.PHONY: $(STAGES) all test
$(STAGES):
	python scripts/run_pipeline.py --stage $@
all:
	python scripts/run_pipeline.py --stage all
test:
	pytest -q

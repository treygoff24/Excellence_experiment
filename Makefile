VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
venv:
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
data:
	$(PY) -m scripts.prepare_data --config config/eval_config.yaml
build:
	$(PY) -m scripts.build_batches --config config/eval_config.yaml
upload:
	$(PY) -m fireworks.upload_dataset --account $(ACCOUNT) --name "t0_control" --file data/batch_inputs/t0_control.jsonl
	$(PY) -m fireworks.upload_dataset --account $(ACCOUNT) --name "t0_treatment" --file data/batch_inputs/t0_treatment.jsonl
	$(PY) -m fireworks.upload_dataset --account $(ACCOUNT) --name "t07_control" --file data/batch_inputs/t07_control.jsonl
	$(PY) -m fireworks.upload_dataset --account $(ACCOUNT) --name "t07_treatment" --file data/batch_inputs/t07_treatment.jsonl
start-jobs:
	@echo "Use scripts/run_all.py to create jobs for all datasets and temperatures"
poll:
	@echo "Use scripts/run_all.py to poll and download all jobs"
parse:
	$(PY) -m fireworks.parse_results --results_jsonl results/results_combined.jsonl --out_csv results/predictions.csv
score:
	$(PY) -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared --out_dir results
stats:
	$(PY) -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/significance.json
report:
	$(PY) -m scripts.summarize_costs --pred_csv results/predictions.csv --config config/eval_config.yaml --out_path results/costs.json
eval:
	$(PY) -m scripts.run_all
smoke:
	$(PY) -m scripts.smoke_test --config config/eval_config.yaml --mode flow --n 2 --out_dir results/smoke

# Audit prompt token lengths and approximate input cost deltas
audit:
	$(PY) -m scripts.audit_prompts --config config/eval_config.yaml --out_json results/prompt_audit.json

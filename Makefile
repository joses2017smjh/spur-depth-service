.PHONY: test lint demo export-fuse export-both cpp aggregate stack

PYTHON ?= python

test:
	$(PYTHON) -m pytest -m "not gpu and not legacy_pyc and not ckpt" -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

demo:
	$(PYTHON) scripts/demo.py

export-fuse:
	$(PYTHON) -m spur_depth.export.to_onnx --graph fuse_decode --out engines/

export-both:
	$(PYTHON) -m spur_depth.export.to_onnx --graph both --ckpt "$$SPUR_CKPT" --out engines/

cpp:
	$(MAKE) -C cpp/scale_shift test

aggregate:
	$(PYTHON) -m spur_depth.bench.aggregate_runs

stack:
	$(PYTHON) -m spur_depth.pipeline.run --out docs/readme

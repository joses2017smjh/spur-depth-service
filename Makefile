.PHONY: test lint demo export-fuse

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

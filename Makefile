.PHONY: lint test smoke

lint:
	python -m compileall hist

test:
	pytest

smoke:
	pytest tests/test_cli_smoke.py

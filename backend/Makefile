.PHONY: setup test serve

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

test:
	.venv/bin/python -m pytest -q

serve:
	.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload

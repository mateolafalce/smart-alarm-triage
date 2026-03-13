.PHONY: install test train train-sample docker-build docker-train clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

train:
	python scripts/train.py

train-sample:
	python scripts/train.py --sample 50000 --models lightgbm

docker-build:
	docker compose build

docker-train:
	docker compose run --rm train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf models/*.pkl reports/figures/*.png reports/evaluation_results.json

.PHONY: install train evaluate run-api run-ui test test-cov docker-up docker-down mlflow-ui lint download-data pre-commit clean

install:
	pip install -r requirements.txt

train:
	python -m src.models.trainer

evaluate:
	python -m src.models.evaluate --train-if-missing

run-api:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ui/app.py --server.port 8501

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

download-data:
	python scripts/download_data.py

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf mlruns models/*.joblib models/*.json reports/figures/*.png data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: install test lint clean run dashboard docker-build docker-run docker-compose-up docker-compose-down docker-gpu

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker-build:
	docker build -t satellite-analysis .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs satellite-analysis

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-gpu:
	docker-compose --profile gpu up -d app-gpu

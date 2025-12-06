# Makefile for Real Estate Investment Advisor

.PHONY: help install setup data train run docker-build docker-run clean test

help:
	@echo "Real Estate Investment Advisor - Available Commands"
	@echo "=================================================="
	@echo "install      - Install all dependencies"
	@echo "setup        - Setup project (create dirs, env file)"
	@echo "data         - Generate sample dataset"
	@echo "train        - Train all ML models"
	@echo "run          - Run Streamlit application"
	@echo "docker-build - Build Docker image"
	@echo "docker-run   - Run Docker container"
	@echo "clean        - Clean generated files"
	@echo "test         - Run tests"

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed models/saved_models models/explainability logs reports exports
	cp .env.example .env
	@echo "Setup complete! Edit .env with your API keys"

data:
	python src/data_preprocessing.py

train:
	python src/model_training.py --report

run:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t real-estate-advisor .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf logs/*.log

test:
	pytest tests/ -v

all: install setup data train run
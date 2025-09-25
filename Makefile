# QuantAI Trading Platform - Essential Commands

.PHONY: help install test lint format clean run-api run-dashboard

help: ## Show this help message
	@echo "QuantAI Trading Platform - Essential Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[dev]
	pre-commit install

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-coverage: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-fail-under=80

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

security: ## Run security checks
	bandit -r src/
	safety check

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

run-api: ## Run API server
	python main.py api

run-dashboard: ## Run dashboard server
	python main.py dashboard

run-test: ## Run test suite
	python main.py test

run-data-pipeline: ## Run data pipeline
	python main.py data-pipeline

run-train: ## Run model training
	python main.py train

run-backtest: ## Run backtesting
	python main.py backtest

dev-setup: ## Complete development setup
	make install-dev
	make format
	make test

dev-check: ## Run all development checks
	make format
	make lint
	make security
	make test-coverage
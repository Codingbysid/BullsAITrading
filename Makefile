# QuantAI Trading Platform - Development Makefile
# Comprehensive development workflow with quality gates

.PHONY: help install test lint format security clean docs build deploy

# Default target
help: ## Show this help message
	@echo "QuantAI Trading Platform - Development Commands"
	@echo "=============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install all dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest tests/ -v --cov=src --cov=apps --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

test-performance: ## Run performance tests only
	pytest tests/performance/ -v -m performance

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov=apps --cov-report=html --cov-report=xml

# Code Quality
lint: ## Run all linting tools
	flake8 src/ apps/ tests/
	mypy src/ apps/
	bandit -r src/ apps/
	safety check

format: ## Format code with Black and isort
	black src/ apps/ tests/
	isort src/ apps/ tests/

format-check: ## Check code formatting without making changes
	black --check src/ apps/ tests/
	isort --check-only src/ apps/ tests/

# Security
security: ## Run security checks
	bandit -r src/ apps/
	safety check
	pip-audit

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

# Database
db-init: ## Initialize database
	python scripts/init_db.py

db-migrate: ## Run database migrations
	alembic upgrade head

db-reset: ## Reset database
	alembic downgrade base
	alembic upgrade head

# Backtesting
backtest: ## Run all backtests
	python run_quantai.py backtest

backtest-simple: ## Run simple backtest
	python apps/backtesting/backtesters/simple_backtest.py

backtest-standalone: ## Run standalone backtest
	python apps/backtesting/backtesters/standalone_backtest.py

backtest-advanced: ## Run advanced backtest
	python apps/backtesting/backtesters/advanced_quantitative_backtester.py

backtest-focused: ## Run focused 5-ticker backtest
	python apps/backtesting/backtesters/focused_5_ticker_backtester.py

# Portfolio Management
portfolio: ## Run portfolio manager
	python apps/portfolio/portfolio_manager_main.py

portfolio-cli: ## Run portfolio manager CLI
	python apps/portfolio/portfolio_manager_main.py --mode cli

portfolio-api: ## Run portfolio manager API
	python apps/portfolio/portfolio_manager_main.py --mode api

# Trading
trading: ## Run trading system
	python apps/trading/focused_quantai_main.py

# Data
data-fetch: ## Fetch market data
	python scripts/setup_market_data.py

data-test: ## Test data sources
	python -c "from src.data.real_market_data_integration import PRDMarketDataIntegration; print('Data sources working')"

# Quality Gates
quality-gate: ## Run all quality checks
	@echo "Running quality gates..."
	@make format-check
	@make lint
	@make test
	@make security
	@echo "All quality gates passed! ✅"

# CI/CD
ci: ## Run CI pipeline
	@echo "Running CI pipeline..."
	@make install-dev
	@make quality-gate
	@make docs
	@echo "CI pipeline completed! ✅"

# Development
dev-setup: ## Setup development environment
	@echo "Setting up development environment..."
	@make install-dev
	@make pre-commit-install
	@make db-init
	@echo "Development environment ready! ✅"

dev-test: ## Run development tests
	@echo "Running development tests..."
	@make test-unit
	@make test-integration
	@echo "Development tests completed! ✅"

# Production
prod-build: ## Build for production
	@echo "Building for production..."
	@make quality-gate
	@make docs
	@echo "Production build ready! ✅"

prod-deploy: ## Deploy to production
	@echo "Deploying to production..."
	@make prod-build
	@echo "Production deployment completed! ✅"

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".bandit" -exec rm -rf {} +

clean-data: ## Clean up data files
	rm -rf data/*.csv
	rm -rf data/*.json
	rm -rf cache/*
	rm -rf output/*.json
	rm -rf results/*.json

clean-all: clean clean-data ## Clean everything
	rm -rf .venv/
	rm -rf venv/
	rm -rf env/

# Monitoring
monitor: ## Start monitoring dashboard
	streamlit run src/dashboard/performance_monitor.py

monitor-api: ## Start monitoring API
	python src/dashboard/dashboard_api.py

# Utilities
check-deps: ## Check for outdated dependencies
	pip list --outdated

update-deps: ## Update dependencies
	pip install --upgrade -r requirements.txt

check-imports: ## Check for unused imports
	unimport --check src/ apps/

fix-imports: ## Fix unused imports
	unimport --remove-all src/ apps/

# Help
help-install: ## Show installation help
	@echo "Installation Commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make dev-setup    - Complete development setup"

help-test: ## Show testing help
	@echo "Testing Commands:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-performance - Run performance tests"
	@echo "  make test-coverage - Run tests with coverage"

help-quality: ## Show quality help
	@echo "Quality Commands:"
	@echo "  make lint         - Run all linting tools"
	@echo "  make format       - Format code"
	@echo "  make security     - Run security checks"
	@echo "  make quality-gate - Run all quality checks"

# Default target
.DEFAULT_GOAL := help
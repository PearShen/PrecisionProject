# PrecisionProject Makefile

.PHONY: install test test-unit test-integration clean demo format lint help

# Default target
all: install

# Install dependencies
install:
	pip install -r requirements.txt

# Run all tests
test:
	pytest tests/ -v

# Run only unit tests
test-unit:
	pytest tests/test_*.py -v -m "not integration"

# Run only integration tests
test-integration:
	pytest tests/test_integration.py -v -m "integration"

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf data/*/golden data/*/test data/*/results

# Run demo
demo:
	python main.py --mode demo

# Format code
format:
	black src/ tests/ main.py --line-length 88

# Lint code
lint:
	flake8 src/ tests/ main.py --max-line-length=88 --ignore=E203,W503

# Show help
help:
	@echo "PrecisionProject - Model Precision Testing Framework"
	@echo ""
	@echo "Available targets:"
	@echo "  install         - Install dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  clean           - Clean up generated files"
	@echo "  demo            - Run demo"
	@echo "  format          - Format code with black"
	@echo "  lint            - Lint code with flake8"
	@echo "  help            - Show this help message"
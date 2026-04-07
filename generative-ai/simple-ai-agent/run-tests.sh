#!/bin/bash
# Run pytest test suite

cd "$(dirname "$0")"

echo "Running pytest test suite..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run tests with coverage
python -m pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

echo ""
echo "Coverage report saved to htmlcov/index.html"

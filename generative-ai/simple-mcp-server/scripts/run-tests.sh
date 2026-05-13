#!/bin/bash
# Run pytest test suite with uv

echo "Running pytest test suite..."
echo ""

# Run tests with coverage using uv
uv run pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

echo ""
echo "Coverage report saved to htmlcov/index.html"
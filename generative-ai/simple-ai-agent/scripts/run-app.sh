#!/bin/bash
# Start the FastAPI server

echo "Starting AI Agent API..."
echo "API will be available at http://localhost:8000"
echo "Interactive docs at http://localhost:8000/docs"
echo ""

uv run python src/app.py

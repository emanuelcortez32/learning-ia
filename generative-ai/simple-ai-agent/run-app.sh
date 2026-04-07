#!/bin/bash
# Start the FastAPI server

cd "$(dirname "$0")"

echo "Starting AI Agent API..."
echo "API will be available at http://localhost:8000"
echo "Interactive docs at http://localhost:8000/docs"
echo ""

python src/app.py

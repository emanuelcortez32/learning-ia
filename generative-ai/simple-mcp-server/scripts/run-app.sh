#!/bin/bash
# Start the FastAPI server

echo "Starting MCP Server..."
echo "API will be available at http://localhost:8088"
echo ""

uv run python src/server.py

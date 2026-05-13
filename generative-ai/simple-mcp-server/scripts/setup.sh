#!/bin/bash
# Setup script for the Simple MCP project

set -e

echo "🚀 Setting up Simple MCP Agent..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or via pip:"
    echo "  pip install uv"
    echo ""
    exit 1
fi

echo "✅ Found uv version: $(uv --version)"
echo ""

# Sync dependencies
echo "📦 Installing dependencies..."
uv sync

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  make dev"
echo ""
echo "Or manually:"
echo "  uv run python src/app.py"
echo ""

#!/bin/bash
# Using python3.12 to avoid Pydantic V1 incompatibility with 3.14
PYTHON_BIN=/usr/local/bin/python3.12

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python 3.12 not found at $PYTHON_BIN. Trying 'python3'..."
    PYTHON_BIN=python3
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $($PYTHON_BIN --version)..."
    $PYTHON_BIN -m venv venv
fi

# Activate and install
source venv/bin/activate
pip install langchain-core langchain-community langchain-text-splitters faiss-cpu fastapi uvicorn langchain-ollama rich
# Note: Ensure you have Ollama installed and have run: ollama pull gemma3:12b
echo "Installation complete. To run the demo, use: ./venv/bin/python main.py <mode>"

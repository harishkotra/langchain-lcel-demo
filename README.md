# LCEL Advanced Capabilities Demo

A comprehensive Python demo showcasing advanced patterns in **LangChain Expression Language (LCEL)** using local LLMs (**Ollama** + **Gemma**).

This project demonstrates how to build robust AI pipelines without imperative "glue code", relying instead on composable Runnables.

## Features

1.  **Branching & Routing** (`RunnableBranch`):
    - Dynamically classifies user intent (Code vs. Data vs. General).
    - Routes the query to a specialized sub-chain based on the intent.
    - *Example*: "Write a binary search function" -> Routes to the **Senior Engineer** chain.

2.  **Parallel Fan-out RAG** (`RunnableParallel`):
    - Queries multiple retrievers in parallel (LangChain docs, Ollama docs, General tips).
    - Merges results from all sources into a single context for the LLM.
    - Demonstrates efficient concurrent execution in LCEL.

3.  **Streaming Middleware** (`RunnableLambda` generator):
    - Intercepts the LLM's token stream in real-time.
    - Applies transformations (Redaction of sensitive emails, Throttling) on the fly.
    - Shows how to build "middleware" for streaming responses.

## Prerequisites

- **Python 3.12+** (Required to avoid Pydantic V1 incompatibility issues).
- **Ollama**: Installed and running.
- **Models**:
    ```bash
    ollama pull gemma3:12b
    ollama pull nomic-embed-text
    ```

## Installation

1.  Clone this repository.
2.  Run the installation script to set up the virtual environment:
    ```bash
    sh install.sh
    ```

    *Note: The script creates a `venv` and installs all dependencies, including `langchain`, `fastapi`, and `rich`.*

## Usage

Activate the virtual environment (optional if using the direct path below):
```bash
source venv/bin/activate
```

Run the demo in one of the three modes:

### 1. Routing Demo
```bash
python main.py routing --query "Show me python code for binary search"
```

### 2. Parallel RAG Demo
```bash
python main.py parallel_rag --query "What is LCEL?"
```

### 3. Streaming Middleware Demo
```bash
python main.py stream_middleware --query "My email is test@example.com"
```

## Project Structure

- `main.py`: Single-file implementation of all valid LCEL patterns.
- `install.sh`: Setup script for dependencies.

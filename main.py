import asyncio
import argparse
import re
import sys
import time
from typing import AsyncIterator, Dict, List, Any

# Rich Imports
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live

# LangChain Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- UX Utilities ---
console = Console()

# --- Configuration ---
MODEL_NAME = "gemma3:12b"  # Fallback to gemma2:latest if needed
EMBEDDING_MODEL = "nomic-embed-text" # Standard decent local embedding model, usually avail with Ollama users

def get_llm():
    """Configures the local Ollama LLM."""
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,
        num_ctx=8192,
    )

def get_embeddings():
    """Configures local embeddings."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# --- Part 1: Branching & Routing ---

def run_routing_demo(query: str):
    console.print(Panel(f"[bold blue]Running Routing Demo[/bold blue]\nQuery: [italic]{query}[/italic]"))
    llm = get_llm()

    # 1. Define distinct chains
    
    # Code Chain
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Python engineer. Answer in code blocks."),
        ("human", "{query}")
    ])
    code_chain = code_prompt | llm | StrOutputParser()

    # Data Chain
    data_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data scientist. Focus on analysis and metrics."),
        ("human", "{query}")
    ])
    data_chain = data_prompt | llm | StrOutputParser()

    # General Chain
    general_prompt = ChatPromptTemplate.from_template("You are a helpful assistant.\nQuestion: {query}")
    general_chain = general_prompt | llm | StrOutputParser()

    # 2. Intent Classifier
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user intent into exactly one of these labels: 'code', 'data', 'general'. Do not output anything else."),
        ("human", "{query}")
    ])
    
    # Simple parser to clean up response just in case
    def parse_intent(ai_msg):
        text = ai_msg.content.strip().lower()
        if "code" in text: return "code"
        if "data" in text: return "data"
        return "general"

    classifier_chain = classifier_prompt | llm | RunnableLambda(parse_intent)

    # 3. Routing Logic
    
    chain_with_intent = RunnableParallel({
        "intent": classifier_chain,
        "query": RunnablePassthrough()
    })

    routing_chain = chain_with_intent | RunnableBranch(
        (lambda x: x["intent"] == "code", RunnableLambda(lambda x: console.print(f"[bold green][Router] Detected 'code'[/bold green]") or x) | RunnableLambda(lambda x: x["query"]) | code_chain),
        (lambda x: x["intent"] == "data", RunnableLambda(lambda x: console.print(f"[bold cyan][Router] Detected 'data'[/bold cyan]") or x) | RunnableLambda(lambda x: x["query"]) | data_chain),
        RunnableLambda(lambda x: console.print(f"[bold yellow][Router] Defaulting to 'general' (intent={x['intent']})[/bold yellow]") or x) | RunnableLambda(lambda x: x["query"]) | general_chain
    )

    # Invoke
    with console.status("[bold green]Routing and Generating Response...[/bold green]", spinner="dots"):
        result = routing_chain.invoke(query)
    
    console.print("\n[bold]Final Output:[/bold]")
    console.print(Markdown(result))
    return result


# --- Part 2: Parallel Fan-out (Multi-Retriever RAG) ---

def create_retriever(name: str, texts: List[str]):
    """Helper to create an in-memory FAISS retriever from text."""
    console.print(f"Building index for [bold]{name}[/bold]...")
    embeddings = get_embeddings()
    docs = [Document(page_content=t, metadata={"source": name}) for t in texts]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

def run_parallel_rag_demo(query: str):
    console.print(Panel(f"[bold blue]Running Parallel RAG Demo[/bold blue]\nQuery: [italic]{query}[/italic]"))
    
    # 1. Setup Data
    langchain_texts = [
        "LangChain is a framework for developing applications powered by language models.",
        "LCEL (LangChain Expression Language) provides a declarative way to compose chains.",
        "Runnables are the building blocks of LCEL."
    ]
    ollama_texts = [
        "Ollama allows you to run open-source large language models globally.",
        "Gemma is a family of lightweight, state-of-the-art open models built by Google.",
        "You can use 'ollama pull gemma3:12b' to get the model."
    ]
    misc_texts = [
        "Python 3.10 introduced structural pattern matching.",
        "Asyncio is a library to write concurrent code using the async/await syntax.",
        "Decorators in Python are a very powerful tool."
    ]
    
    try:
        with console.status("[bold blue]indexing Documents...[/bold blue]"):
            retriever_lc = create_retriever("langchain", langchain_texts)
            retriever_ollama = create_retriever("ollama", ollama_texts)
            retriever_misc = create_retriever("misc", misc_texts)
    except Exception as e:
        console.print(f"[bold red]Error creating retrievers:[/bold red] {e}")
        return

    # 2. Parallel Retrieval
    parallel_retrievers = RunnableParallel({
        "lc_docs": retriever_lc,
        "ollama_docs": retriever_ollama,
        "misc_docs": retriever_misc,    
    })

    # 3. Merge Step
    def merge_docs(inputs: dict) -> List[Document]:
        merged = []
        console.print("[bold purple][Merger][/bold purple] Retrieving from sources:")
        for key, docs in inputs.items():
            console.print(f"  - {key}: {len(docs)} docs found")
            merged.extend(docs)
        return merged

    # 4. RAG Chain
    llm = get_llm()
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful technical assistant. Answer strictly based on the context provided."),
        ("human", "Question: {query}\n\nContext:\n{context}")
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])

    rag_chain = (
        {
            "query": RunnablePassthrough(),
            "context": parallel_retrievers | RunnableLambda(merge_docs) | RunnableLambda(format_docs)
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Invoke
    with console.status("[bold green]Parallel Retrieval & Synthesis in progress...[/bold green]", spinner="dots"):
        result = rag_chain.invoke(query)
        
    console.print("\n[bold]Final Output:[/bold]")
    console.print(Markdown(result))
    return result


# --- Part 3: Streaming Middleware ---

SENSITIVE_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")

async def middleware_stream(iterable: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Async generator that acts as middleware with buffering to handle split tokens."""
    buffer = ""
    
    async for chunk in iterable:
        # 1. Extract text content
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        buffer += text
        
        # 2. Check for separators to flush safe parts
        # If the buffer ends with part of a potential email, we keep it.
        # Simple heuristic: Split by space or newline. Process all but the last potentially incomplete word.
        
        if " " in buffer or "\n" in buffer:
            # Find the last separator
            last_space = max(buffer.rfind(" "), buffer.rfind("\n"))
            
            # Extract safe chunk, keep remainder
            to_process = buffer[:last_space+1] # Include the separator
            buffer = buffer[last_space+1:]
            
            # Redact in the safe chunk
            safe_chunk = SENSITIVE_PATTERN.sub("[REDACTED_EMAIL]", to_process)
            
            # 3. Throttle (simulated per 'flush')
            await asyncio.sleep(0.02)
            
            yield safe_chunk

    # Flush remaining buffer at the end
    if buffer:
        safe_chunk = SENSITIVE_PATTERN.sub("[REDACTED_EMAIL]", buffer)
        yield safe_chunk

async def run_stream_middleware_demo(query: str):
    console.print(Panel(f"[bold blue]Running Streaming Middleware Demo[/bold blue]\nQuery: [italic]{query}[/italic]"))
    
    llm = get_llm()
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a verbose assistant. If I ask about personal info, invent some fake emails."),
        ("human", "{query}")
    ])
    
    # Base chain
    chain = base_prompt | llm
    
    console.print("\n[bold]Streaming Output (with Middleware):[/bold]\n")
    
    with console.status("[bold green]Initializing Stream...[/bold green]"):
        raw_iterator = chain.astream(query)
        # Fake init delay to show spinner
        time.sleep(0.5) 
    
    # For streaming, we construct the text incrementally
    # Rich doesn't support streaming Markdown updates easily in Live without flickering or full redraws.
    # So we'll print raw text for the stream to appear 'streaming', then render final Markdown if we wanted.
    # Or just stream plain text. The user asked for better markdown handling generally, for streaming plain text is usually clearer.
    # However, we can use Live display to update a Markdown object if we buffer it. Let's try that for "wow" factor.
    
    full_response = ""
    with Live(Markdown(""), refresh_per_second=10, console=console) as live:
        async for chunk in middleware_stream(raw_iterator):
            full_response += chunk
            live.update(Markdown(full_response))
            
    console.print("\n\n[bold green][Stream Complete][/bold green]")

def main():
    parser = argparse.ArgumentParser(description="LCEL Advanced Capabilities Demo")
    parser.add_argument("mode", choices=["routing", "parallel_rag", "stream_middleware"], help="Demo mode to run")
    parser.add_argument("--query", type=str, help="Input query", default="")

    args = parser.parse_args()

    # Defaults if no query provided
    defaults = {
        "routing": "Write a Python function to compute fibonacci.",
        "parallel_rag": "What is LCEL and how do I download gemma?",
        "stream_middleware": "My email is test@example.com, please repeat it back to me."
    }
    
    query = args.query if args.query else defaults[args.mode]

    if args.mode == "routing":
        run_routing_demo(query)
    elif args.mode == "parallel_rag":
        run_parallel_rag_demo(query)
    elif args.mode == "stream_middleware":
        asyncio.run(run_stream_middleware_demo(query))

if __name__ == "__main__":
    main()

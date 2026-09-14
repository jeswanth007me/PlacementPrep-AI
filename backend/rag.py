"""
Agentic RAG & Preparation Knowledge Retrieval for PlacementPrep AI.

Milestone 3:
- Curated local interview-preparation corpus for technical interviews
- Vector similarity search using gemini-embedding-001 and cosine similarity
- Confidence-gated retrieval with explicit threshold validation (default threshold: 0.70)
- Fallback guidance on low-confidence or empty results
- Callable LangChain tool for the LangGraph interview agent
"""

import json
import logging
import math
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.tools import tool

# Curated knowledge corpus covering core Python interview concepts
DEFAULT_PREP_CORPUS = [
    {
        "id": "py_concurrency_gil",
        "topic": "Concurrency & GIL",
        "content": (
            "Python Global Interpreter Lock (GIL): CPython's GIL is a mutex that prevents multiple native threads "
            "from executing Python bytecode simultaneously. For CPU-bound tasks, multithreading does not provide true "
            "parallelism and multiprocessing (separate memory spaces) must be used. For I/O-bound tasks, multithreading "
            "or asyncio (cooperative single-threaded multitasking via an event loop) releases the GIL during I/O operations."
        ),
    },
    {
        "id": "py_asyncio_event_loop",
        "topic": "Asyncio & Event Loop",
        "content": (
            "Asyncio & Event Loop: Python's asyncio runs a single-threaded cooperative event loop. Coroutines declared "
            "with 'async def' yield execution at 'await' expressions. Crucially, CPU-intensive synchronous operations or "
            "blocking I/O (like time.sleep or synchronous database drivers) block the entire event loop, freezing all concurrent "
            "tasks. Non-blocking libraries or run_in_executor must be used for blocking calls."
        ),
    },
    {
        "id": "py_memory_management",
        "topic": "Memory Management & Garbage Collection",
        "content": (
            "Python Memory Management: Python uses reference counting as its primary memory management mechanism; when an "
            "object's reference count drops to zero, its memory is immediately deallocated. To handle cyclic references (e.g., "
            "A references B and B references A), CPython includes a generational garbage collector with 3 generations (Gen 0, 1, 2) "
            "that periodically detects and frees unreferenced cycles."
        ),
    },
    {
        "id": "py_decorators_closures",
        "topic": "Decorators & Closures",
        "content": (
            "Decorators & Closures: A closure occurs when a nested function retains access to variables in its enclosing scope "
            "even after the enclosing function has finished executing. A decorator wraps a function to modify its behavior. "
            "Decorators taking arguments require three levels of nested functions: the outer decorator factory accepting arguments, "
            "the middle decorator accepting the function, and the inner wrapper executing the logic with functools.wraps."
        ),
    },
    {
        "id": "py_generators_iterators",
        "topic": "Generators & Iterators",
        "content": (
            "Generators vs Lists: Lists store all elements in memory simultaneously (eager evaluation). Generators produce "
            "items one at a time on demand using the 'yield' keyword (lazy evaluation). Generators maintain state between calls, "
            "consume O(1) memory, and are ideal for large streams, infinite sequences, or large file processing."
        ),
    },
    {
        "id": "redis_persistence_tradeoffs",
        "topic": "Redis Persistence & Reliability",
        "content": (
            "Redis Persistence Options: Redis offers RDB (point-in-time snapshots at configured intervals) and AOF (Append Only "
            "File logging every write command). RDB is compact and fast to restore but risks data loss between snapshots. AOF provides "
            "higher durability (fsync every sec or always) but produces larger files and higher disk I/O overhead. When used as a "
            "message broker, Redis has no built-in guaranteed delivery like RabbitMQ; if a worker crashes before ack, messages in simple "
            "queues can be lost unless Redis Streams with Consumer Groups are used."
        ),
    },
    {
        "id": "celery_task_guarantees",
        "topic": "Celery & Asynchronous Task Queues",
        "content": (
            "Celery Task Execution Guarantees: By default, Celery acknowledges tasks when they are fetched (acks_late=False). If a "
            "worker crashes mid-execution, the task is lost. Setting 'acks_late=True' acknowledges tasks after execution, providing "
            "at-least-once delivery; however, this requires all tasks to be idempotent (safe to run multiple times without duplicating "
            "effects, e.g., using unique transaction IDs)."
        ),
    },
    {
        "id": "django_orm_optimization",
        "topic": "Django ORM Query Optimization",
        "content": (
            "Django ORM Optimization (N+1 Problem): Accessing foreign key or many-to-many relationships in a loop generates an "
            "individual SQL query per row (the N+1 query problem). 'select_related' performs an SQL JOIN in a single query for "
            "ForeignKey and OneToOne fields. 'prefetch_related' executes a separate batch query using SQL 'WHERE IN' and joins in "
            "Python memory, suitable for ManyToMany and reverse ForeignKey relationships."
        ),
    },
    {
        "id": "fastapi_architecture",
        "topic": "FastAPI Architecture & Dependency Injection",
        "content": (
            "FastAPI Architecture: FastAPI is built on Starlette and Pydantic. It handles request validation, serialization, and "
            "OpenAPI documentation automatically. Its Dependency Injection system ('Depends') allows modular composition of database "
            "sessions, authentication, and security logic. Endpoints defined with 'def' run in a background threadpool, while 'async def' "
            "endpoints run directly on the event loop."
        ),
    },
]

# Configurable confidence threshold for accepting retrieved material
CONFIDENCE_THRESHOLD = 0.70


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class LocalPrepRetriever:
    """
    In-memory vector retriever for interview preparation material.
    Supports batch live embeddings via Gemini or mock embeddings for offline testing.
    """

    def __init__(self, corpus: Optional[List[dict]] = None, embedding_client: Optional[object] = None):
        self.corpus = corpus or DEFAULT_PREP_CORPUS
        self.embedding_client = embedding_client
        self._doc_embeddings: Optional[List[Tuple[dict, List[float]]]] = None

    def _get_embedding_client(self):
        if self.embedding_client is not None:
            return self.embedding_client

        from google import genai
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)

    def _embed_single(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for a single string."""
        client = self._get_embedding_client()
        if client is None:
            return None
        try:
            res = client.models.embed_content(model="gemini-embedding-001", contents=text)
            return res.embeddings[0].values
        except Exception as e:
            logging.getLogger("backend.rag").warning(f"Embedding failed: {e}")
            return None

    def _ensure_indexed(self):
        """Ensure all corpus documents have precomputed embeddings."""
        if self._doc_embeddings is not None:
            return

        from pathlib import Path
        cache_path = Path(__file__).resolve().parent.parent / "data" / "prep_embeddings.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                self._doc_embeddings = [(item["doc"], item["vector"]) for item in data]
                return
            except Exception:
                pass

        client = self._get_embedding_client()
        if client is None:
            self._doc_embeddings = []
            return

        try:
            texts = [f"{doc['topic']}: {doc['content']}" for doc in self.corpus]
            res = client.models.embed_content(model="gemini-embedding-001", contents=texts)
            self._doc_embeddings = [(doc, emb.values) for doc, emb in zip(self.corpus, res.embeddings)]
        except Exception as e:
            logging.getLogger("backend.rag").warning(f"Batch embedding indexing failed: {e}")
            self._doc_embeddings = []

    def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> dict:
        """
        Search corpus for query and return structured confidence-gated result.

        Returns a dictionary formatted with:
        - status: "HIGH_CONFIDENCE", "LOW_CONFIDENCE", or "EMPTY"
        - confidence: float score (0.0 to 1.0)
        - chunks: list of relevant text strings
        - guidance: explicit instructions for the agent
        """
        if not query or not query.strip():
            return {
                "status": "EMPTY",
                "confidence": 0.0,
                "chunks": [],
                "guidance": "Query was empty. Fall back to candidate profile and conversation history.",
            }

        clean_query = query.strip()
        if topic and topic.strip():
            clean_query = f"{topic.strip()}: {clean_query}"

        # Ensure index is ready
        self._ensure_indexed()

        best_doc = None
        best_score = 0.0

        if self._doc_embeddings:
            query_vec = self._embed_single(clean_query)
            if query_vec is not None:
                for doc, doc_vec in self._doc_embeddings:
                    sim = cosine_similarity(query_vec, doc_vec)
                    if sim > best_score:
                        best_score = sim
                        best_doc = doc

        # Fallback to token overlap if embeddings are unavailable (e.g. offline testing)
        if best_doc is None and not self._doc_embeddings:
            query_tokens = set(clean_query.lower().split())
            for doc in self.corpus:
                doc_tokens = set((doc["topic"] + " " + doc["content"]).lower().split())
                overlap = len(query_tokens.intersection(doc_tokens))
                keyword_score = min(1.0, overlap / max(1, len(query_tokens)))
                if keyword_score > best_score:
                    best_score = keyword_score
                    best_doc = doc

        best_score = round(best_score, 3)

        if best_score >= threshold and best_doc is not None:
            return {
                "status": "HIGH_CONFIDENCE",
                "confidence": best_score,
                "chunks": [f"[{best_doc['topic']}] {best_doc['content']}"],
                "guidance": (
                    f"High-confidence prep material retrieved (confidence: {best_score} >= {threshold}). "
                    "Use these facts to formulate a precise, targeted follow-up question."
                ),
            }
        else:
            return {
                "status": "LOW_CONFIDENCE",
                "confidence": best_score,
                "chunks": [],
                "guidance": (
                    f"Low confidence match ({best_score} < threshold {threshold}). "
                    "DO NOT use external material. Fall back to the candidate's profile and conversation history."
                ),
            }


# Default singleton instance
_DEFAULT_RETRIEVER = LocalPrepRetriever()


@tool
def retrieve_prep_material(query: str, topic: Optional[str] = None) -> str:
    """
    Retrieve curated interview-preparation knowledge and deep technical facts.
    Call this tool ONLY when you need verified technical concepts, trade-offs, or
    edge cases to probe a candidate's knowledge gap or ask a targeted follow-up question.
    Do NOT call this tool if the candidate gave a clear, confident, correct answer.
    """
    result = _DEFAULT_RETRIEVER.retrieve(query=query, topic=topic)
    return json.dumps(result, indent=2)

#!/usr/bin/env python3
"""
Hybrid RAG with Self-Learning
Tries local Ollama first, falls back to Claude, learns from Claude

Configuration via environment variables:
  CHROMADB_HOST - ChromaDB server IP (default: <ADD-IP-ADDRESS>)
  CHROMADB_PORT - ChromaDB port (default: 8000)
  OLLAMA_HOST   - Ollama server IP (default: <ADD-IP-ADDRESS>)
  OLLAMA_PORT   - Ollama port (default: 11434)
  OLLAMA_MODEL  - Model to use (default: mistral)
  ANTHROPIC_API_KEY - Claude API key
  RAG_CONFIDENCE_THRESHOLD - Confidence threshold 0.0-1.0 (default: 0.7)
  RAG_DISTANCE_THRESHOLD - Max embedding distance for good match (default: 1.0)
"""

import chromadb
from chromadb.utils import embedding_functions
import requests
import json
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

# =============================================================================
# CONFIGURATION - All configurable via environment variables
# =============================================================================

def load_config():
    """Load configuration from environment variables with sensible defaults"""
    config = {
        # ChromaDB settings
        "chromadb_host": os.getenv("CHROMADB_HOST", "<ADD-IP-ADDRESS>"),
        "chromadb_port": int(os.getenv("CHROMADB_PORT", "8000")),

        # Ollama settings
        "ollama_host": os.getenv("OLLAMA_HOST", "<ADD-IP-ADDRESS>"),
        "ollama_port": int(os.getenv("OLLAMA_PORT", "11434")),
        "ollama_model": os.getenv("OLLAMA_MODEL", "mistral"),

        # RAG settings
        "collection_name": os.getenv("RAG_COLLECTION", "knowledge_base"),
        "confidence_threshold": float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.7")),
        "distance_threshold": float(os.getenv("RAG_DISTANCE_THRESHOLD", "1.0")),
        "n_results": int(os.getenv("RAG_N_RESULTS", "5")),

        # Embedding model (for better quality)
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }

    # Load Claude API key
    config["claude_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    if not config["claude_api_key"]:
        # Try to load from file
        key_file = os.path.expanduser("~/.anthropic_api_key")
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                config["claude_api_key"] = f.read().strip()

    return config

# Load config at module level
CONFIG = load_config()


def get_embedding_function():
    """Get the embedding function for ChromaDB"""
    try:
        # Use sentence-transformers for better embeddings
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CONFIG["embedding_model"]
        )
    except Exception:
        # Fallback to default
        return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client():
    """Connect to ChromaDB"""
    return chromadb.HttpClient(
        host=CONFIG["chromadb_host"],
        port=CONFIG["chromadb_port"]
    )


def query_chromadb(client, question, n_results=None, filter_metadata=None):
    """
    Search vector database for relevant context

    Args:
        client: ChromaDB client
        question: Query string
        n_results: Number of results to return
        filter_metadata: Optional dict to filter by metadata (e.g., {"system": "unraid"})

    Returns:
        dict with documents, metadatas, distances, or None on error
    """
    if n_results is None:
        n_results = CONFIG["n_results"]

    try:
        collection = client.get_collection(
            name=CONFIG["collection_name"],
            embedding_function=get_embedding_function()
        )

        query_params = {
            "query_texts": [question],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        # Add metadata filter if provided
        if filter_metadata:
            query_params["where"] = filter_metadata

        results = collection.query(**query_params)
        return results
    except Exception as e:
        print(f"  ChromaDB query failed: {e}")
        return None


def calculate_retrieval_confidence(distances):
    """
    Calculate confidence score based on embedding distances
    Lower distance = better match = higher confidence

    Args:
        distances: List of distances from ChromaDB query

    Returns:
        float: Confidence score 0.0-1.0
    """
    if not distances or not distances[0]:
        return 0.0

    # Get the best (lowest) distance
    best_distance = min(distances[0])

    # Convert distance to confidence
    # distance 0 = confidence 1.0
    # distance >= threshold = confidence 0.0
    threshold = CONFIG["distance_threshold"]

    if best_distance >= threshold:
        return 0.0

    # Linear interpolation: closer to 0 = closer to 1.0 confidence
    confidence = 1.0 - (best_distance / threshold)

    return min(max(confidence, 0.0), 1.0)


def ask_ollama(prompt, context=""):
    """Query local Ollama LLM"""
    model = CONFIG["ollama_model"]

    if context:
        full_prompt = f"""Context from documentation:
{context}

Question: {prompt}

Answer based on the context above. If you're not confident about the answer, say "I don't have enough information to answer confidently."
"""
    else:
        full_prompt = prompt

    try:
        response = requests.post(
            f"http://{CONFIG['ollama_host']}:{CONFIG['ollama_port']}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return None
    except Exception as e:
        print(f"  Ollama query failed: {e}")
        return None


def evaluate_confidence(answer, retrieval_confidence=0.5):
    """
    Evaluate confidence of answer using heuristics + retrieval score

    Args:
        answer: The LLM's response text
        retrieval_confidence: Confidence from embedding distance (0.0-1.0)

    Returns:
        float: Combined confidence score 0.0-1.0
    """
    if not answer:
        return 0.0

    answer_lower = answer.lower()

    # Low confidence indicators (immediate penalty)
    low_confidence_phrases = [
        "i don't know",
        "i'm not sure",
        "i don't have enough information",
        "i cannot",
        "i can't answer",
        "not confident",
        "unclear",
        "uncertain",
        "i don't have information",
        "no information available"
    ]

    for phrase in low_confidence_phrases:
        if phrase in answer_lower:
            return 0.2  # Low confidence regardless of retrieval

    # High confidence indicators
    high_confidence_phrases = [
        "the answer is",
        "specifically",
        "according to",
        "here's what",
        "this is",
        "based on the",
        "the documentation shows",
        "as shown in",
        "the configuration"
    ]

    # Start with retrieval confidence as base (weighted 40%)
    heuristic_confidence = 0.5

    for phrase in high_confidence_phrases:
        if phrase in answer_lower:
            heuristic_confidence += 0.1

    # Check for specific details (IPs, paths, commands)
    if re.search(r'\d+\.\d+\.\d+\.\d+', answer):  # Has IP address
        heuristic_confidence += 0.1
    if re.search(r'/[a-z/]+', answer):  # Has file paths
        heuristic_confidence += 0.1
    if re.search(r'`[^`]+`', answer):  # Has code/commands
        heuristic_confidence += 0.1

    heuristic_confidence = min(heuristic_confidence, 1.0)

    # Combine: 40% retrieval quality + 60% answer heuristics
    combined = (retrieval_confidence * 0.4) + (heuristic_confidence * 0.6)

    return min(combined, 1.0)


def ask_claude(prompt):
    """Fallback to Claude API for complex questions"""
    if not CONFIG["claude_api_key"]:
        print("  Claude API key not configured!")
        return None

    try:
        client = Anthropic(api_key=CONFIG["claude_api_key"])

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text
    except Exception as e:
        print(f"  Claude API failed: {e}")
        return None


def store_learning(client, question, answer, source="claude"):
    """Store Claude's answer in ChromaDB for future learning"""
    try:
        collection = client.get_collection(
            name=CONFIG["collection_name"],
            embedding_function=get_embedding_function()
        )

        # Create unique ID with hash to prevent duplicates of same question
        question_hash = hash(question.lower().strip()) % 100000
        doc_id = f"learned-{question_hash}-{datetime.now().strftime('%Y%m%d')}"

        # Check if similar question already exists and delete it
        try:
            existing = collection.get(ids=[doc_id])
            if existing and existing['ids']:
                collection.delete(ids=[doc_id])
                print(f"  Updated existing learned answer")
        except:
            pass

        collection.add(
            documents=[f"Q: {question}\nA: {answer}"],
            metadatas=[{
                "source": source,
                "type": "learned",
                "date": datetime.now().isoformat(),
                "question": question[:500]  # Truncate for metadata
            }],
            ids=[doc_id]
        )

        print(f"  Learned and stored in ChromaDB (ID: {doc_id})")
        return True
    except Exception as e:
        print(f"  Failed to store learning: {e}")
        return False


def hybrid_query(question, filter_system=None, filter_topic=None):
    """
    Main hybrid query function:
    1. Try local Ollama + RAG
    2. Evaluate confidence (heuristics + embedding distance)
    3. Fallback to Claude if needed
    4. Learn from Claude

    Args:
        question: The user's question
        filter_system: Optional - filter by system ("local", "proxmox", "unraid")
        filter_topic: Optional - filter by topic ("docker", "networking", etc.)
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    if filter_system or filter_topic:
        filters = []
        if filter_system:
            filters.append(f"system={filter_system}")
        if filter_topic:
            filters.append(f"topic={filter_topic}")
        print(f"Filters: {', '.join(filters)}")
    print(f"{'='*60}\n")

    # Step 1: Connect to ChromaDB
    print(f"[1/5] Connecting to ChromaDB ({CONFIG['chromadb_host']})...")
    client = get_chroma_client()

    # Step 2: Get context from RAG with optional filtering
    print("[2/5] Searching for relevant context...")

    # Build metadata filter
    metadata_filter = None
    if filter_system or filter_topic:
        if filter_system and filter_topic:
            metadata_filter = {
                "$and": [
                    {"system": filter_system},
                    {"topics": {"$contains": filter_topic}}
                ]
            }
        elif filter_system:
            metadata_filter = {"system": filter_system}
        elif filter_topic:
            metadata_filter = {"topics": {"$contains": filter_topic}}

    rag_results = query_chromadb(client, question, filter_metadata=metadata_filter)

    context = ""
    retrieval_confidence = 0.0

    if rag_results and rag_results['documents'][0]:
        docs = rag_results['documents'][0]
        metadatas = rag_results['metadatas'][0]
        distances = rag_results.get('distances', [[]])

        # Calculate retrieval confidence from distances
        retrieval_confidence = calculate_retrieval_confidence(distances)

        print(f"  Found {len(docs)} relevant documents (retrieval confidence: {retrieval_confidence:.2f}):")
        for i, (metadata, dist) in enumerate(zip(metadatas, distances[0] if distances else [0]*len(metadatas)), 1):
            doc_type = metadata.get('type', 'doc')
            source = metadata.get('source', 'unknown')
            system = metadata.get('system', '?')

            if doc_type == 'learned':
                print(f"   {i}. {source} (learned) [dist: {dist:.3f}]")
            else:
                print(f"   {i}. {source} [{system}] [dist: {dist:.3f}]")

        context = "\n\n".join(docs)
    else:
        print("  No relevant context found in ChromaDB")

    # Step 3: Try local Ollama
    print(f"\n[3/5] Asking local Ollama ({CONFIG['ollama_model']} on {CONFIG['ollama_host']})...")
    ollama_answer = ask_ollama(question, context)

    if not ollama_answer:
        print("  Ollama failed, going straight to Claude...")
        if CONFIG["claude_api_key"]:
            claude_answer = ask_claude(question)
            if claude_answer:
                store_learning(client, question, claude_answer)
                print(f"\n[5/5] Answer (from Claude):")
                print(f"\n{claude_answer}\n")
                print(f"Cost: ~$0.002")
                return claude_answer
        else:
            print("  No fallback available (Claude API key not set)")
        return None

    # Step 4: Evaluate confidence (combined heuristics + retrieval)
    print(f"\n[4/5] Evaluating confidence...")
    confidence = evaluate_confidence(ollama_answer, retrieval_confidence)
    print(f"  Retrieval confidence: {retrieval_confidence:.2f}")
    print(f"  Combined confidence:  {confidence:.2f} (threshold: {CONFIG['confidence_threshold']})")

    if confidence >= CONFIG["confidence_threshold"]:
        print(f"  High confidence - using local answer!")
        print(f"\n[5/5] Answer (from local Ollama):")
        print(f"\n{ollama_answer}\n")
        print(f"Cost: $0.00 (local)")
        print(f"Source: {CONFIG['ollama_host']} -> Ollama -> {CONFIG['ollama_model']}")
        return ollama_answer

    # Step 5: Low confidence - fallback to Claude
    print(f"  Low confidence ({confidence:.2f}) - asking Claude for better answer...")

    if not CONFIG["claude_api_key"]:
        print("  Claude API key not set, returning local answer anyway")
        print(f"\n[5/5] Answer (from local Ollama - low confidence):")
        print(f"\n{ollama_answer}\n")
        return ollama_answer

    claude_answer = ask_claude(question)

    if claude_answer:
        print(f"  Got answer from Claude")

        # Learn from Claude
        print(f"  Storing Claude's answer in ChromaDB for future learning...")
        store_learning(client, question, claude_answer)

        print(f"\n[5/5] Answer (from Claude API):")
        print(f"\n{claude_answer}\n")
        print(f"Cost: ~$0.002")
        print(f"This answer is now stored - next time will be faster and free!")
        return claude_answer
    else:
        print(f"  Claude also failed, returning local answer anyway:")
        print(f"\n{ollama_answer}\n")
        return ollama_answer


def print_config():
    """Print current configuration"""
    print("\nCurrent Configuration:")
    print(f"  ChromaDB:   {CONFIG['chromadb_host']}:{CONFIG['chromadb_port']}")
    print(f"  Ollama:     {CONFIG['ollama_host']}:{CONFIG['ollama_port']} ({CONFIG['ollama_model']})")
    print(f"  Collection: {CONFIG['collection_name']}")
    print(f"  Confidence: {CONFIG['confidence_threshold']}")
    print(f"  Distance:   {CONFIG['distance_threshold']}")
    print(f"  Embedding:  {CONFIG['embedding_model']}")
    print(f"  Claude API: {'configured' if CONFIG['claude_api_key'] else 'NOT SET'}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid RAG with self-learning: Local LLM + Claude fallback"
    )
    parser.add_argument("question", nargs="*", help="Your question")
    parser.add_argument("--system", "-s", choices=["local", "proxmox", "unraid"],
                        help="Filter by system")
    parser.add_argument("--topic", "-t",
                        help="Filter by topic (docker, networking, gpu, etc.)")
    parser.add_argument("--config", "-c", action="store_true",
                        help="Show current configuration")

    args = parser.parse_args()

    print("="*60)
    print("HYBRID SELF-LEARNING RAG SYSTEM")
    print("Local Ollama -> Claude Fallback -> Automatic Learning")
    print("="*60)

    if args.config:
        print_config()
        return 0

    if not args.question:
        print("\nUsage: python3 hybrid-rag.py 'Your question here'")
        print("\nOptions:")
        print("  -s, --system   Filter by system (local, proxmox, unraid)")
        print("  -t, --topic    Filter by topic (docker, networking, gpu, etc.)")
        print("  -c, --config   Show current configuration")
        print("\nExamples:")
        print("  python3 hybrid-rag.py 'What is Docker?'")
        print("  python3 hybrid-rag.py -s unraid 'How do I check array status?'")
        print("  python3 hybrid-rag.py -t docker 'How do I view container logs?'")
        print("\nEnvironment Variables:")
        print("  CHROMADB_HOST, CHROMADB_PORT, OLLAMA_HOST, OLLAMA_PORT")
        print("  OLLAMA_MODEL, ANTHROPIC_API_KEY, RAG_CONFIDENCE_THRESHOLD")
        print("\nFirst time: May use Claude API")
        print("Second time: Will use local Ollama (learned!)")
        return 1

    question = " ".join(args.question)
    result = hybrid_query(question, filter_system=args.system, filter_topic=args.topic)

    if result:
        return 0
    else:
        print("\nFailed to get answer from any source")
        return 1


if __name__ == "__main__":
    sys.exit(main())

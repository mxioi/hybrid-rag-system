#!/usr/bin/env python3
"""
ChromaDB Helper for MCP and Local LLM Integration

Provides:
- Proper embedding-based queries (not raw HTTP)
- Hybrid search (semantic + keyword)
- Smart chunking for documents
- Metadata enrichment
- Collection management

Usage:
  python chromadb_helper.py query "your search query"
  python chromadb_helper.py add "document content" --source "source.md"
  python chromadb_helper.py index /path/to/file.md
  python chromadb_helper.py list-collections
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

CHROMADB_HOST = os.getenv("CHROMADB_HOST", "<ADD-IP-ADDRESS>")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
DEFAULT_COLLECTION = os.getenv("CHROMADB_COLLECTION", "infrastructure_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# =============================================================================
# LAZY IMPORTS (for faster CLI startup)
# =============================================================================

_chromadb = None
_embedding_function = None


def get_chromadb():
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        chromadb = get_chromadb()
        try:
            from chromadb.utils import embedding_functions
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        except Exception as e:
            print(f"Warning: SentenceTransformer not available ({e}), using default")
            from chromadb.utils import embedding_functions
            _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


def get_client():
    chromadb = get_chromadb()
    return chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)


def get_collection(name: str = None):
    client = get_client()
    collection_name = name or DEFAULT_COLLECTION
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"description": "Infrastructure and homelab documentation"}
    )


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    return text.strip()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better retrieval

    Returns list of dicts with:
    - content: the chunk text
    - chunk_index: position in document
    - char_start: character offset
    - char_end: character offset
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    text = clean_text(text)

    if len(text) <= chunk_size:
        return [{
            "content": text,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": len(text),
            "total_chunks": 1
        }]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for sentence end in last 20% of chunk
            search_start = end - int(chunk_size * 0.2)
            search_text = text[search_start:end]

            # Priority: paragraph > sentence > word
            break_patterns = [
                r'\n\n',           # Paragraph
                r'\.\s+',          # Sentence
                r'[;:]\s+',        # Clause
                r',\s+',           # Phrase
                r'\s+',            # Word
            ]

            for pattern in break_patterns:
                matches = list(re.finditer(pattern, search_text))
                if matches:
                    # Use the last match
                    end = search_start + matches[-1].end()
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "content": chunk_text,
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
            })
            chunk_index += 1

        start = end - overlap

    # Add total_chunks to each
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def extract_metadata_from_content(content: str, filepath: str = None) -> Dict[str, Any]:
    """Extract metadata from document content"""
    metadata = {}

    # Detect system from content
    system_patterns = {
        "unraid": [r"\bunraid\b", r"\b/mnt/user\b", r"\barray\b"],
        "proxmox": [r"\bproxmox\b", r"\bpve\b", r"\blxc\b", r"\bqemu\b"],
        "windows": [r"\bpowershell\b", r"\bactive directory\b", r"\bsccm\b"],
        "docker": [r"\bdocker\b", r"\bcontainer\b", r"\bcompose\b"],
        "network": [r"\bunifi\b", r"\bvlan\b", r"\bfirewall\b"],
    }

    content_lower = content.lower()
    detected_systems = []
    for system, patterns in system_patterns.items():
        if any(re.search(p, content_lower) for p in patterns):
            detected_systems.append(system)

    if detected_systems:
        metadata["system"] = detected_systems[0]  # Primary system
        metadata["systems"] = ",".join(detected_systems)

    # Detect category from filepath
    if filepath:
        path_lower = filepath.lower()
        if "troubleshoot" in path_lower:
            metadata["category"] = "troubleshooting"
        elif "howto" in path_lower or "guide" in path_lower:
            metadata["category"] = "howto"
        elif "reference" in path_lower or "api" in path_lower:
            metadata["category"] = "reference"
        elif "config" in path_lower:
            metadata["category"] = "configuration"
        else:
            metadata["category"] = "general"

        metadata["filepath"] = filepath

    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()[:200]

    return metadata


def generate_doc_id(content: str, source: str = "") -> str:
    """Generate a stable document ID"""
    hash_input = f"{source}:{content[:500]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# =============================================================================
# QUERY FUNCTIONS
# =============================================================================

def query(
    query_text: str,
    n_results: int = 5,
    collection_name: str = None,
    where_filter: Dict = None,
    include_distances: bool = True
) -> Dict[str, Any]:
    """
    Query ChromaDB with semantic search

    Args:
        query_text: The search query
        n_results: Number of results to return
        collection_name: Collection to search (default: infrastructure_docs)
        where_filter: Metadata filter (e.g., {"system": "unraid"})
        include_distances: Include distance scores

    Returns:
        Dict with documents, metadatas, distances, ids
    """
    collection = get_collection(collection_name)

    include = ["documents", "metadatas"]
    if include_distances:
        include.append("distances")

    query_params = {
        "query_texts": [query_text],
        "n_results": n_results,
        "include": include
    }

    if where_filter:
        query_params["where"] = where_filter

    try:
        results = collection.query(**query_params)
        return results
    except Exception as e:
        return {"error": str(e)}


def hybrid_query(
    query_text: str,
    n_results: int = 5,
    collection_name: str = None,
    keyword_boost: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining semantic and keyword matching

    Returns sorted list of results with combined scores
    """
    collection = get_collection(collection_name)

    # Semantic search
    semantic_results = collection.query(
        query_texts=[query_text],
        n_results=n_results * 2,  # Get more for merging
        include=["documents", "metadatas", "distances"]
    )

    # Extract keywords for filtering
    keywords = extract_keywords(query_text)

    # Score and merge results
    scored_results = []

    if semantic_results.get("documents") and semantic_results["documents"][0]:
        for i, doc in enumerate(semantic_results["documents"][0]):
            semantic_distance = semantic_results["distances"][0][i] if semantic_results.get("distances") else 1.0
            semantic_score = max(0, 1 - semantic_distance)  # Convert distance to similarity

            # Keyword score
            keyword_score = calculate_keyword_score(doc, keywords)

            # Combined score
            combined_score = semantic_score * (1 - keyword_boost) + keyword_score * keyword_boost

            scored_results.append({
                "document": doc,
                "metadata": semantic_results["metadatas"][0][i] if semantic_results.get("metadatas") else {},
                "id": semantic_results["ids"][0][i] if semantic_results.get("ids") else None,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
                "distance": semantic_distance
            })

    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)

    return scored_results[:n_results]


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text"""
    # Remove common words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and", "but", "if",
        "or", "because", "until", "while", "this", "that", "these", "those",
        "what", "which", "who", "whom", "my", "your", "his", "her", "its",
        "our", "their", "i", "me", "you", "he", "she", "it", "we", "they"
    }

    # Tokenize and filter
    words = re.findall(r'\b[a-z]+\b', text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    return list(set(keywords))


def calculate_keyword_score(document: str, keywords: List[str]) -> float:
    """Calculate keyword match score for a document"""
    if not keywords:
        return 0.0

    doc_lower = document.lower()
    matches = sum(1 for kw in keywords if kw in doc_lower)
    return matches / len(keywords)


# =============================================================================
# ADD/INDEX FUNCTIONS
# =============================================================================

def add_document(
    content: str,
    metadata: Dict[str, Any] = None,
    doc_id: str = None,
    collection_name: str = None
) -> Dict[str, Any]:
    """
    Add a single document to ChromaDB

    Args:
        content: Document content
        metadata: Optional metadata dict
        doc_id: Optional document ID (generated if not provided)
        collection_name: Target collection

    Returns:
        Dict with status and id
    """
    collection = get_collection(collection_name)

    if doc_id is None:
        source = metadata.get("source", "") if metadata else ""
        doc_id = generate_doc_id(content, source)

    # Enrich metadata
    full_metadata = extract_metadata_from_content(content)
    if metadata:
        full_metadata.update(metadata)
    full_metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()

    try:
        collection.add(
            documents=[content],
            metadatas=[full_metadata],
            ids=[doc_id]
        )
        return {"status": "success", "id": doc_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def index_file(
    filepath: str,
    collection_name: str = None,
    chunk: bool = True
) -> Dict[str, Any]:
    """
    Index a file into ChromaDB with chunking

    Args:
        filepath: Path to the file
        collection_name: Target collection
        chunk: Whether to split into chunks

    Returns:
        Dict with status and chunk count
    """
    path = Path(filepath)

    if not path.exists():
        return {"status": "error", "error": f"File not found: {filepath}"}

    # Read file
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"status": "error", "error": f"Failed to read file: {e}"}

    collection = get_collection(collection_name)

    # Base metadata
    base_metadata = {
        "source": path.name,
        "filepath": str(path.absolute()),
        "file_type": path.suffix.lstrip("."),
        "indexed_at": datetime.now(timezone.utc).isoformat()
    }
    base_metadata.update(extract_metadata_from_content(content, str(path)))

    if chunk and len(content) > CHUNK_SIZE:
        chunks = chunk_text(content)
        ids = []
        documents = []
        metadatas = []

        for i, chunk_data in enumerate(chunks):
            chunk_id = f"{generate_doc_id(content, str(path))}-{i:04d}"
            chunk_meta = base_metadata.copy()
            chunk_meta["chunk_index"] = chunk_data["chunk_index"]
            chunk_meta["total_chunks"] = chunk_data["total_chunks"]

            ids.append(chunk_id)
            documents.append(chunk_data["content"])
            metadatas.append(chunk_meta)

        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return {"status": "success", "chunks": len(chunks), "filepath": str(path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        # Single document
        doc_id = generate_doc_id(content, str(path))
        try:
            collection.add(
                documents=[content],
                metadatas=[base_metadata],
                ids=[doc_id]
            )
            return {"status": "success", "chunks": 1, "filepath": str(path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


def index_directory(
    directory: str,
    patterns: List[str] = None,
    collection_name: str = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Index all matching files in a directory

    Args:
        directory: Directory path
        patterns: List of glob patterns (default: ["*.md", "*.txt", "*.yaml"])
        collection_name: Target collection
        recursive: Search recursively

    Returns:
        Dict with status and file count
    """
    if patterns is None:
        patterns = ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"]

    dir_path = Path(directory)
    if not dir_path.exists():
        return {"status": "error", "error": f"Directory not found: {directory}"}

    results = {"status": "success", "indexed": 0, "failed": 0, "files": []}

    for pattern in patterns:
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        for filepath in files:
            result = index_file(str(filepath), collection_name)
            if result["status"] == "success":
                results["indexed"] += 1
                results["files"].append({
                    "path": str(filepath),
                    "chunks": result.get("chunks", 1)
                })
            else:
                results["failed"] += 1
                print(f"  Failed: {filepath}: {result.get('error', 'unknown')}")

    return results


# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def list_collections() -> List[Dict[str, Any]]:
    """List all ChromaDB collections"""
    client = get_client()
    collections = client.list_collections()

    result = []
    for col in collections:
        try:
            count = col.count()
        except:
            count = "unknown"

        result.append({
            "name": col.name,
            "count": count,
            "metadata": col.metadata
        })

    return result


def get_collection_stats(collection_name: str = None) -> Dict[str, Any]:
    """Get statistics for a collection"""
    collection = get_collection(collection_name)

    try:
        count = collection.count()

        # Sample some documents for metadata stats
        sample = collection.peek(limit=100)

        systems = {}
        categories = {}

        for meta in sample.get("metadatas", []):
            if meta:
                sys = meta.get("system", "unknown")
                cat = meta.get("category", "unknown")
                systems[sys] = systems.get(sys, 0) + 1
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "name": collection.name,
            "count": count,
            "systems_sample": systems,
            "categories_sample": categories
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChromaDB Helper")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Query command
    query_parser = subparsers.add_parser("query", help="Search the knowledge base")
    query_parser.add_argument("text", help="Search query")
    query_parser.add_argument("-n", "--n-results", type=int, default=5)
    query_parser.add_argument("-c", "--collection", default=DEFAULT_COLLECTION)
    query_parser.add_argument("--system", help="Filter by system (unraid, proxmox, windows, etc)")
    query_parser.add_argument("--hybrid", action="store_true", help="Use hybrid search")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a document")
    add_parser.add_argument("content", help="Document content")
    add_parser.add_argument("-s", "--source", help="Source identifier")
    add_parser.add_argument("-c", "--collection", default=DEFAULT_COLLECTION)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", help="File or directory path")
    index_parser.add_argument("-c", "--collection", default=DEFAULT_COLLECTION)
    index_parser.add_argument("-r", "--recursive", action="store_true", default=True)
    index_parser.add_argument("-p", "--patterns", nargs="+", help="File patterns to match")

    # List collections
    list_parser = subparsers.add_parser("list-collections", help="List all collections")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get collection statistics")
    stats_parser.add_argument("-c", "--collection", default=DEFAULT_COLLECTION)

    args = parser.parse_args()

    if args.command == "query":
        where_filter = None
        if args.system:
            where_filter = {"system": args.system}

        if args.hybrid:
            results = hybrid_query(args.text, args.n_results, args.collection)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\nHybrid search results for: {args.text}\n")
                for i, r in enumerate(results):
                    print(f"--- Result {i+1} (score: {r['combined_score']:.3f}) ---")
                    print(f"Source: {r['metadata'].get('source', 'unknown')}")
                    print(f"System: {r['metadata'].get('system', 'unknown')}")
                    print(f"{r['document'][:500]}...")
                    print()
        else:
            results = query(args.text, args.n_results, args.collection, where_filter)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\nSearch results for: {args.text}\n")
                if results.get("documents") and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                        dist = results["distances"][0][i] if results.get("distances") else "N/A"
                        print(f"--- Result {i+1} (distance: {dist:.3f if isinstance(dist, float) else dist}) ---")
                        print(f"Source: {meta.get('source', 'unknown')}")
                        print(f"System: {meta.get('system', 'unknown')}")
                        print(f"{doc[:500]}...")
                        print()
                else:
                    print("No results found")

    elif args.command == "add":
        metadata = {"source": args.source} if args.source else {}
        result = add_document(args.content, metadata, collection_name=args.collection)
        print(json.dumps(result, indent=2))

    elif args.command == "index":
        path = Path(args.path)
        if path.is_file():
            result = index_file(str(path), args.collection)
        else:
            result = index_directory(str(path), args.patterns, args.collection, args.recursive)
        print(json.dumps(result, indent=2))

    elif args.command == "list-collections":
        collections = list_collections()
        for col in collections:
            print(f"  - {col['name']} ({col['count']} documents)")

    elif args.command == "stats":
        stats = get_collection_stats(args.collection)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

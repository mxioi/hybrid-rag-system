#!/usr/bin/env python3
"""
MCP ChromaDB Bridge

Provides a simple interface for the MCP server to call ChromaDB operations
via subprocess, avoiding the embedding issues with direct curl calls.

Usage (from MCP server):
  python3 mcp_chromadb_bridge.py query "search text" --n-results 5
  python3 mcp_chromadb_bridge.py add "document content" --source "source.md"

Output is always JSON for easy parsing.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chromadb_helper import (
    query,
    hybrid_query,
    add_document,
    list_collections,
    get_collection_stats
)


def format_query_results(results: Dict[str, Any], max_doc_length: int = 800) -> Dict[str, Any]:
    """Format query results for MCP output"""
    if "error" in results:
        return {"success": False, "error": results["error"]}

    formatted = {
        "success": True,
        "count": 0,
        "results": []
    }

    if results.get("documents") and results["documents"][0]:
        formatted["count"] = len(results["documents"][0])

        for i, doc in enumerate(results["documents"][0]):
            result_item = {
                "index": i + 1,
                "document": doc[:max_doc_length] + ("..." if len(doc) > max_doc_length else ""),
                "full_length": len(doc)
            }

            if results.get("metadatas") and results["metadatas"][0]:
                result_item["metadata"] = results["metadatas"][0][i]

            if results.get("distances") and results["distances"][0]:
                result_item["distance"] = round(results["distances"][0][i], 4)
                # Convert to similarity score (0-1, higher is better)
                result_item["similarity"] = round(max(0, 1 - results["distances"][0][i]), 4)

            formatted["results"].append(result_item)

    return formatted


def format_hybrid_results(results: list, max_doc_length: int = 800) -> Dict[str, Any]:
    """Format hybrid query results for MCP output"""
    formatted = {
        "success": True,
        "count": len(results),
        "results": []
    }

    for i, r in enumerate(results):
        doc = r["document"]
        result_item = {
            "index": i + 1,
            "document": doc[:max_doc_length] + ("..." if len(doc) > max_doc_length else ""),
            "full_length": len(doc),
            "metadata": r.get("metadata", {}),
            "combined_score": round(r.get("combined_score", 0), 4),
            "semantic_score": round(r.get("semantic_score", 0), 4),
            "keyword_score": round(r.get("keyword_score", 0), 4)
        }
        formatted["results"].append(result_item)

    return formatted


def main():
    parser = argparse.ArgumentParser(description="MCP ChromaDB Bridge")
    subparsers = parser.add_subparsers(dest="command")

    # Query
    q = subparsers.add_parser("query")
    q.add_argument("text", help="Search query")
    q.add_argument("--n-results", type=int, default=5)
    q.add_argument("--collection", default="infrastructure_docs")
    q.add_argument("--system", help="Filter by system")
    q.add_argument("--hybrid", action="store_true")
    q.add_argument("--max-length", type=int, default=800)

    # Add
    a = subparsers.add_parser("add")
    a.add_argument("content", help="Document content")
    a.add_argument("--source", default="mcp")
    a.add_argument("--collection", default="infrastructure_docs")
    a.add_argument("--system", help="System tag")
    a.add_argument("--category", help="Category tag")

    # Collections
    c = subparsers.add_parser("collections")

    # Stats
    s = subparsers.add_parser("stats")
    s.add_argument("--collection", default="infrastructure_docs")

    args = parser.parse_args()

    try:
        if args.command == "query":
            where_filter = {"system": args.system} if args.system else None

            if args.hybrid:
                results = hybrid_query(args.text, args.n_results, args.collection)
                output = format_hybrid_results(results, args.max_length)
            else:
                results = query(args.text, args.n_results, args.collection, where_filter)
                output = format_query_results(results, args.max_length)

            print(json.dumps(output))

        elif args.command == "add":
            metadata = {"source": args.source}
            if args.system:
                metadata["system"] = args.system
            if args.category:
                metadata["category"] = args.category

            result = add_document(args.content, metadata, collection_name=args.collection)
            print(json.dumps(result))

        elif args.command == "collections":
            collections = list_collections()
            print(json.dumps({"success": True, "collections": collections}))

        elif args.command == "stats":
            stats = get_collection_stats(args.collection)
            print(json.dumps({"success": True, "stats": stats}))

        else:
            print(json.dumps({"success": False, "error": "Unknown command"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

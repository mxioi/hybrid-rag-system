#!/usr/bin/env python3
"""
Store Successful Chat Interactions in ChromaDB

This script stores Q&A pairs that worked correctly, so Mistral can learn
from past successes and provide better answers in the future.

Usage:
    python3 store-successful-chat.py "question" "answer" ["tools_used"]

Example:
    python3 store-successful-chat.py \
        "What is my Unraid storage?" \
        "3.7TB total, 1.8TB available" \
        "wrapper.sh fulltools"
"""

import chromadb
import sys
from datetime import datetime

# ChromaDB configuration
CHROMADB_HOST = "<ADD-IP-ADDRESS>"
CHROMADB_PORT = 8000
COLLECTION_NAME = "infrastructure_docs"


def store_successful_interaction(question, answer, tools_used="", confidence="high"):
    """
    Store a successful Q&A interaction in ChromaDB.

    Args:
        question: The user's question
        answer: The successful answer
        tools_used: What tools/scripts were used (optional)
        confidence: Confidence level (high/medium/low)
    """

    try:
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        collection = client.get_or_create_collection(COLLECTION_NAME)

        # Create a rich document that Mistral can learn from
        doc = f"""
========================================
VERIFIED SUCCESSFUL INTERACTION
========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {confidence}
Status: ✓ VERIFIED WORKING

Question:
{question}

Successful Answer:
{answer}

Tools/Methods Used:
{tools_used if tools_used else "Not specified"}

Learning Point:
When user asks similar questions, use this approach as it was verified successful.

========================================
"""

        # Generate unique ID
        doc_id = f"success_{int(datetime.now().timestamp() * 1000)}"

        # Store in ChromaDB with rich metadata
        collection.add(
            documents=[doc],
            metadatas=[
                {
                    "type": "successful_interaction",
                    "verified": True,
                    "confidence": confidence,
                    "date": datetime.now().isoformat(),
                    "question": question[:200],  # First 200 chars for searching
                    "tools": tools_used,
                }
            ],
            ids=[doc_id],
        )

        print("✓ Successfully stored interaction in ChromaDB")
        print(f"  ID: {doc_id}")
        print(f"  Question: {question[:60]}...")
        print(f"  Confidence: {confidence}")
        print("")
        print("Mistral will now learn from this successful interaction!")
        print("Similar future queries will benefit from this knowledge.")

        return True

    except Exception as e:
        print(f"❌ Error storing interaction: {str(e)}")
        print("")
        print("Troubleshooting:")
        print(f"1. Check ChromaDB is running: curl http://{CHROMADB_HOST}:{CHROMADB_PORT}/api/v2/heartbeat")
        print(f"2. Verify collection exists: Check ChromaDB at http://{CHROMADB_HOST}:{CHROMADB_PORT}")
        return False


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("\nRequired: question and answer")
        print("Optional: tools_used, confidence")
        sys.exit(1)

    question = sys.argv[1]
    answer = sys.argv[2]
    tools_used = sys.argv[3] if len(sys.argv) > 3 else ""
    confidence = sys.argv[4] if len(sys.argv) > 4 else "high"

    if not question.strip() or not answer.strip():
        print("Error: Question and answer cannot be empty")
        sys.exit(1)

    success = store_successful_interaction(question, answer, tools_used, confidence)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

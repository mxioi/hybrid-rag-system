#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Demo
Combines ChromaDB vector search with Ollama LLM
"""

import chromadb
import requests
import json
import sys
from pathlib import Path

# Configuration
CHROMADB_HOST = "http://<ADD-IP-ADDRESS>:8000"  # Now on Unraid
OLLAMA_HOST = "http://10.0.0.X:11434"  # Still on Proxmox
COLLECTION_NAME = "infrastructure_docs"

def get_chroma_client():
    """Connect to ChromaDB"""
    return chromadb.HttpClient(host="<ADD-IP-ADDRESS>", port=8000)

def create_collection(client):
    """Create or get collection for infrastructure docs"""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Using existing collection: {COLLECTION_NAME}")
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Infrastructure and homelab documentation"}
        )
        print(f"✓ Created new collection: {COLLECTION_NAME}")
    return collection

def index_documents(collection, documents):
    """Add documents to vector database"""
    print(f"\nIndexing {len(documents)} documents...")

    collection.add(
        documents=[doc["content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[doc["id"] for doc in documents]
    )

    print(f"✓ Indexed {len(documents)} documents")

def query_vector_db(collection, query, n_results=3):
    """Search for relevant documents"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results

def ask_ollama(prompt, model="mistral", context=""):
    """Query Ollama LLM"""
    if context:
        full_prompt = f"""Context information:
{context}

User question: {prompt}

Please answer the question based on the context provided above. If the context doesn't contain relevant information, say so."""
    else:
        full_prompt = prompt

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=120
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

def rag_query(collection, question, model="mistral", n_results=3):
    """
    Full RAG pipeline:
    1. Search vector database for relevant context
    2. Send context + question to LLM
    3. Return LLM response
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Step 1: Retrieve relevant documents
    print("\n[1/3] Searching vector database...")
    results = query_vector_db(collection, question, n_results=n_results)

    if not results['documents'][0]:
        print("✗ No relevant documents found")
        context = ""
    else:
        print(f"✓ Found {len(results['documents'][0])} relevant documents")

        # Build context from top results
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context_parts.append(f"[Document {i+1} - {metadata.get('source', 'unknown')}]\n{doc}")

        context = "\n\n".join(context_parts)

        # Show what was retrieved
        print("\n📄 Retrieved context:")
        for i, metadata in enumerate(results['metadatas'][0], 1):
            print(f"   {i}. {metadata.get('source', 'Unknown')}: {metadata.get('title', 'No title')}")

    # Step 2: Generate response with LLM
    print(f"\n[2/3] Generating answer with {model}...")
    answer = ask_ollama(question, model=model, context=context)

    # Step 3: Display answer
    print(f"\n[3/3] Answer:")
    print(f"\n{answer}\n")

    return answer

def load_sample_docs():
    """Load sample infrastructure documentation"""
    docs = [
        {
            "id": "proxmox-overview",
            "content": """Proxmox VE is an open-source server virtualization management solution.
            It combines KVM hypervisor and LXC containers, software-defined storage and networking,
            and a web-based management interface. Proxmox supports live migration, high availability
            clustering, and built-in backup/restore functionality. The system runs on Debian Linux
            and can manage multiple servers through a single interface.""",
            "metadata": {
                "source": "proxmox-docs",
                "title": "Proxmox Overview",
                "category": "virtualization"
            }
        },
        {
            "id": "ollama-setup",
            "content": """Ollama is a tool for running large language models locally.
            Container ID 300 runs Ubuntu 24.04 with Ollama v0.15.1. The service listens on
            10.0.0.X:11434 and is accessible from the network. Currently running Mistral 7B
            in CPU mode. GPU passthrough will be configured when AMD GPU arrives. Models are stored
            in /usr/share/ollama/.ollama/models with 42GB available storage.""",
            "metadata": {
                "source": "ollama-setup-guide",
                "title": "Ollama Configuration",
                "category": "ai"
            }
        },
        {
            "id": "lxc-containers",
            "content": """LXC (Linux Containers) are lightweight virtualization that shares the host kernel.
            Unlike VMs, containers don't require full OS overhead. Proxmox LXC containers offer near-native
            performance with minimal memory footprint. Containers can be privileged or unprivileged.
            Unprivileged containers map container root to unprivileged host user for security.
            Device passthrough requires specific cgroup permissions and mount entries in container config.""",
            "metadata": {
                "source": "proxmox-docs",
                "title": "LXC Container Concepts",
                "category": "virtualization"
            }
        },
        {
            "id": "gpu-passthrough",
            "content": """GPU passthrough allows containers or VMs to directly access physical GPU hardware.
            Requirements: IOMMU/VT-d enabled in BIOS and kernel (intel_iommu=on), GPU in separate IOMMU group,
            correct device permissions. For LXC, host must load GPU drivers, then pass /dev/dri/cardX and
            /dev/dri/renderDX to container via lxc.mount.entry. For AMD GPUs with ROCm, set HSA_OVERRIDE_GFX_VERSION
            based on architecture (11.0.0 for RDNA3, 10.3.0 for RDNA2).""",
            "metadata": {
                "source": "gpu-passthrough-guide",
                "title": "GPU Passthrough Configuration",
                "category": "hardware"
            }
        },
        {
            "id": "chromadb-rag",
            "content": """ChromaDB is an open-source vector database for embeddings.
            Running on Docker at <ADD-IP-ADDRESS>:8000 (v1.0.0). Used for RAG (Retrieval-Augmented Generation)
            by converting documents to embeddings, storing in vector DB, then retrieving similar documents
            based on query embedding. Integrates with Ollama for local LLM inference. Collections store
            documents with metadata, IDs, and vector embeddings for semantic search.""",
            "metadata": {
                "source": "rag-demo",
                "title": "ChromaDB and RAG",
                "category": "ai"
            }
        },
        {
            "id": "open-webui",
            "content": """Open WebUI provides a ChatGPT-like interface for Ollama.
            Running as Docker container on Proxmox host at port 3000. Accessible via
            http://<ADD-IP-ADDRESS>:3000. Connected to Ollama instance at 10.0.0.X:11434.
            Supports chat history, model selection, system prompts, and file uploads.
            First-time visitors create local account (no external authentication).""",
            "metadata": {
                "source": "ollama-setup-guide",
                "title": "Open WebUI",
                "category": "ai"
            }
        },
        {
            "id": "rocm-amd",
            "content": """ROCm (Radeon Open Compute) is AMD's GPU compute platform for ML/AI workloads.
            Install via AMD repository: repo.radeon.com/rocm/apt/6.3. Provides HIP (CUDA alternative)
            and supports PyTorch, TensorFlow. For Ollama, need rocm-hip-sdk and rocm-smi packages.
            Check GPU with rocm-smi command. May need HSA_OVERRIDE_GFX_VERSION environment variable
            for consumer GPUs not officially supported. RDNA2/3 architectures work well for inference.""",
            "metadata": {
                "source": "gpu-passthrough-guide",
                "title": "ROCm Installation",
                "category": "ai"
            }
        },
        {
            "id": "system-specs",
            "content": """Server hardware: Intel 12th Gen i9-12900H CPU with VT-x virtualization.
            Currently has Intel Alder Lake-P integrated graphics. AMD GPU pending installation.
            Storage: ~140GB free on local-lvm, 42GB free in Ollama container. Network: 192.168.5.x subnet.
            Proxmox host: <ADD-IP-ADDRESS>. Services: Ollama (10.0.0.X:11434),
            ChromaDB (<ADD-IP-ADDRESS>:8000), Open WebUI (<ADD-IP-ADDRESS>:3000).""",
            "metadata": {
                "source": "system-inventory",
                "title": "System Specifications",
                "category": "hardware"
            }
        }
    ]

    return docs

def main():
    """Main RAG demo"""
    print("="*60)
    print("RAG PIPELINE DEMO")
    print("Retrieval-Augmented Generation with ChromaDB + Ollama")
    print("="*60)

    # Connect to ChromaDB
    print("\n[Setup] Connecting to ChromaDB...")
    try:
        client = get_chroma_client()
        print("✓ Connected to ChromaDB")
    except Exception as e:
        print(f"✗ Failed to connect to ChromaDB: {e}")
        print(f"\nMake sure ChromaDB is running:")
        print(f"  docker ps | grep chromadb")
        return 1

    # Create/get collection
    collection = create_collection(client)

    # Check if we need to index documents
    count = collection.count()
    if count == 0:
        print(f"\n[Setup] Collection is empty, indexing sample documents...")
        docs = load_sample_docs()
        index_documents(collection, docs)
    else:
        print(f"✓ Collection has {count} documents")

    print(f"\n{'='*60}")
    print("READY TO ANSWER QUESTIONS")
    print(f"{'='*60}\n")

    # Example queries
    if len(sys.argv) > 1:
        # Question provided as command line argument
        question = " ".join(sys.argv[1:])
        rag_query(collection, question)
    else:
        # Run demo queries
        demo_questions = [
            "What is Ollama and where is it running?",
            "How do I set up GPU passthrough for AMD GPUs?",
            "What's the difference between LXC containers and VMs?",
            "What services are running and on which IP addresses?"
        ]

        print("Running demo queries...\n")

        for question in demo_questions:
            rag_query(collection, question)
            print("\n" + "-"*60 + "\n")
            input("Press Enter to continue...")

        print("\n[Demo Complete]")
        print(f"\nTo ask your own questions, run:")
        print(f"  python3 {sys.argv[0]} 'Your question here'")

    return 0

if __name__ == "__main__":
    sys.exit(main())

# Hybrid RAG Self-Learning AI System

A local-first AI system combining Ollama LLM inference with ChromaDB vector search, MCP server integration, and optional Claude API fallback.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Open WebUI    │────▶│   Ollama     │────▶│   Mistral/      │
│   :3001         │     │   :11434     │     │   Llama3.2      │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   MCP Server    │────▶│  ChromaDB    │
│   (20 tools)    │     │   :8000      │
└─────────────────┘     └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scripts                                   │
│  - ollama-full-toolset.py (agent with tools)                │
│  - hybrid-rag.py (RAG + Claude fallback)                    │
│  - improved-router.py (intent classification)               │
│  - chromadb_helper.py (sentence-transformer embeddings)     │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Local-first inference** using Ollama (Mistral, Llama3.2)
- **Semantic search** with sentence-transformer embeddings
- **Hybrid search** combining semantic + keyword matching
- **Intent classification** for smart query routing
- **MCP server** with 20 tools for Docker, system, ChromaDB, and Ollama
- **Optional Claude fallback** for complex queries
- **Self-learning** from successful interactions

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-only torch (smaller, works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export OLLAMA_HOST="<ADD-IP-ADDRESS>"      # Your Ollama server
export CHROMADB_HOST="<ADD-IP-ADDRESS>"       # Your ChromaDB server
export CHROMADB_COLLECTION="homelab_knowledge"
```

### 3. Index Your Documentation

```bash
# Index a single file
python scripts/chromadb_helper.py index /path/to/docs/file.md

# Index a directory
python scripts/chromadb_helper.py index /path/to/docs/ --patterns "*.md" "*.yaml"

# Query the knowledge base
python scripts/chromadb_helper.py query "docker containers" --hybrid --n-results 5
```

### 4. Run the Agent

```bash
# Full toolset mode (can execute commands, search files, etc.)
python scripts/ollama-full-toolset.py "check disk usage on unraid"

# Hybrid RAG mode (with Claude fallback)
python scripts/hybrid-rag.py "explain the docker network setup"

# Intent classification only (dry run)
ROUTER_DRY_RUN=1 python scripts/ollama-full-toolset.py "restart nginx"
```

## Directory Structure

```
├── scripts/
│   ├── chromadb_helper.py      # Embedding & search with sentence-transformers
│   ├── improved-router.py      # Intent classification & routing
│   ├── mcp_chromadb_bridge.py  # Python bridge for MCP server
│   ├── ollama-full-toolset.py  # Main agent with 12 tools
│   ├── hybrid-rag.py           # RAG with Claude fallback
│   ├── index-all-systems.py    # Multi-system documentation indexer
│   └── rag-demo.py             # Simple RAG demo
├── context/
│   ├── homelab-context.yaml    # Infrastructure inventory
│   └── infrastructure/         # Reference documentation
│       ├── bash-linux-reference.md
│       ├── docker-compose-patterns.md
│       ├── windows-powershell-reference.md
│       └── ...
├── docs/
│   └── LOCAL-LLM-IMPROVEMENTS.md  # Improvement roadmap
├── mcp-server/
│   └── src/index.ts            # MCP server (TypeScript)
└── requirements.txt
```

## MCP Server Tools

The MCP server (v2.2.0) provides 20 tools:

| Category | Tools |
|----------|-------|
| Docker | `docker_list`, `docker_start`, `docker_stop`, `docker_restart`, `docker_logs`, `docker_stats` |
| System | `system_info`, `disk_usage`, `array_status`, `network_info`, `shares_list` |
| ChromaDB | `chromadb_query`, `chromadb_add`, `chromadb_collections`, `chromadb_stats` |
| Ollama | `ollama_query`, `ollama_models` |
| Integration | `ask_local_ai`, `index_knowledge`, `run_command` |

## ChromaDB Collections

| Collection | Documents | Description |
|------------|-----------|-------------|
| `homelab_knowledge` | 57 | Main knowledge base with sentence-transformer embeddings |
| `infrastructure_docs` | 1700+ | Legacy collection (default embeddings) |

## Roadmap

See [docs/LOCAL-LLM-IMPROVEMENTS.md](docs/LOCAL-LLM-IMPROVEMENTS.md) for the full roadmap.

**Completed:**
- [x] Intent classification router
- [x] ChromaDB helper with sentence-transformers
- [x] MCP Python bridge
- [x] Context files (homelab, PowerShell, Bash, Docker)

**In Progress:**
- [ ] Integrate improved-router into ollama-full-toolset
- [ ] Self-learning from Claude answers
- [ ] System-specific prompt templates
- [ ] Fine-tuning data collection

## License

MIT License - see [LICENSE](LICENSE)

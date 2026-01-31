# Local LLM Improvement Roadmap

## Current Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Open WebUI    │────▶│   Ollama     │────▶│   Mistral/      │
│   :3001         │     │   :11434     │     │   Llama3.2      │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   MCP Server    │────▶│  ChromaDB    │
│   (19 tools)    │     │   :8000      │
└─────────────────┘     └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scripts                                   │
│  - ollama-full-toolset.py (12 tools)                        │
│  - hybrid-rag.py (RAG + Claude fallback)                    │
│  - improved-router.py (NEW: intent classification)          │
│  - chromadb_helper.py (NEW: proper embeddings)              │
└─────────────────────────────────────────────────────────────┘
```

## Priority Improvements

### 1. [DONE] Intent Classification Router
**File:** `improved-router.py`

Features:
- Classifies queries into: TOOL_REQUIRED, RAG_LOOKUP, CONVERSATIONAL, HYBRID
- Detects target systems (Unraid, Proxmox, Windows AD, etc.)
- Calculates complexity score
- Generates context-aware system prompts
- Logs all decisions for fine-tuning data collection

Usage:
```bash
python3 improved-router.py "check disk space on unraid"
# Output: Intent=tool_required, Systems=[unraid], Tools=[Bash]
```

### 2. [DONE] ChromaDB Helper with Proper Embeddings
**File:** `chromadb_helper.py`

Features:
- Uses SentenceTransformer embeddings (not raw HTTP)
- Hybrid search (semantic + keyword)
- Smart text chunking with overlap
- Metadata extraction from content
- Collection management

Usage:
```bash
# Query
python3 chromadb_helper.py query "docker container management" --n-results 5

# Hybrid search (better for mixed queries)
python3 chromadb_helper.py query "restart nginx" --hybrid

# Index a file
python3 chromadb_helper.py index /path/to/docs/file.md

# Index a directory
python3 chromadb_helper.py index /path/to/docs/ --patterns "*.md" "*.yaml"
```

### 3. [DONE] MCP ChromaDB Bridge
**File:** `mcp_chromadb_bridge.py`

Allows MCP server to call ChromaDB via Python subprocess instead of curl.
Returns JSON for easy parsing.

### 4. [TODO] Update MCP Server to Use Python Bridge

**Edit:** `/mnt/user/data/docker/unraid-mcp-server/src/index.ts`

Replace the curl-based ChromaDB query with:
```typescript
case "chromadb_query": {
  const { query, n_results = 5, collection = "infrastructure_docs" } = args as any;
  const script = `${SCRIPTS_PATH}/mcp_chromadb_bridge.py`;

  const result = await runCommand(
    `cd ${SCRIPTS_PATH} && source ../venv/bin/activate && python3 ${script} query "${query.replace(/"/g, '\\"')}" --n-results ${n_results} --collection ${collection}`,
    60000
  );

  try {
    const data = JSON.parse(result);
    if (data.success) {
      let output = `Found ${data.count} results:\n\n`;
      data.results.forEach((r: any) => {
        output += `--- Result ${r.index} (similarity: ${r.similarity}) ---\n`;
        output += `Source: ${r.metadata?.source || 'unknown'}\n`;
        output += `${r.document}\n\n`;
      });
      return { content: [{ type: "text", text: output }] };
    }
    return { content: [{ type: "text", text: `Query failed: ${data.error}` }] };
  } catch (e: any) {
    return { content: [{ type: "text", text: result }] };
  }
}
```

---

## Advanced Improvements

### 5. Tool Execution Enforcement

Add to `ollama-full-toolset.py`:

```python
# After classifying query intent
if classification.intent == QueryIntent.TOOL_REQUIRED:
    if not tools_executed:
        # Force tool execution before responding
        suggested_tool = classification.tool_hints[0] if classification.tool_hints else "Bash"
        # Execute the tool...
```

### 6. Self-Learning from Claude Fallback

When Claude provides a good answer, automatically:
1. Index the Q&A pair in ChromaDB
2. Log as training data for fine-tuning
3. Update confidence thresholds

```python
def learn_from_claude(question: str, claude_answer: str, context_used: list):
    # Create a verified Q&A document
    qa_doc = f"""
Question: {question}

Answer: {claude_answer}

Context: This answer was verified by Claude and should be trusted.
"""
    # Add to ChromaDB with high confidence metadata
    add_document(
        qa_doc,
        metadata={
            "source": "claude_verified",
            "question": question[:200],
            "verified": True,
            "confidence": "high"
        },
        collection_name="verified_answers"
    )
```

### 7. Context Window Optimization

For long documents, use smart context selection:

```python
def select_relevant_context(query: str, documents: list, max_tokens: int = 2000) -> str:
    """Select most relevant parts of documents within token budget"""
    # Score each document chunk by relevance
    scored_chunks = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size=500)
        for chunk in chunks:
            score = calculate_relevance(query, chunk["content"])
            scored_chunks.append((score, chunk["content"]))

    # Take highest scoring chunks until token limit
    scored_chunks.sort(reverse=True)
    selected = []
    total_tokens = 0

    for score, chunk in scored_chunks:
        chunk_tokens = len(chunk) // 4  # Rough estimate
        if total_tokens + chunk_tokens > max_tokens:
            break
        selected.append(chunk)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(selected)
```

### 8. Fine-Tuning Data Collection

The router already logs decisions. To create fine-tuning data:

```bash
# Extract successful interactions
cat logs/training-examples.jsonl | \
  jq -c 'select(.success == true)' | \
  head -1000 > finetune-data.jsonl

# Convert to training format
python3 scripts/create_finetune_data.py finetune-data.jsonl
```

Training format for Ollama/LoRA:
```json
{
  "instruction": "Check the disk usage on the Unraid server",
  "input": "",
  "output": "I'll check the disk usage on Unraid.\n\n```bash\ndf -h | grep -E '^/dev|mnt'\n```\n\n[Tool Output]\n...\n\nThe Unraid array is at 54% capacity..."
}
```

### 9. System Prompt Templates

Create system prompts optimized for each system:

**For Unraid queries:**
```
You are managing an Unraid server (<ADD-HOSTNAME>, <ADD-IP-ADDRESS>).
Available tools: docker_list, docker_restart, disk_usage, array_status
Array: 2 data disks, 1 parity, 3.7TB total (54% used)
Cache: 477GB NVMe (6% used)
Critical containers: nginx-proxy-manager, lldap, authelia, chromadb

When checking status, always run the appropriate tool first.
```

**For Windows AD queries:**
```
You are managing Active Directory for homelab.local.
- DC1: <ADD-IP-ADDRESS> (primary, RSAT installed)
- DC2: <ADD-IP-ADDRESS> (secondary)

Access via: ssh DC1 "powershell -Command '...'"

Common commands:
- Get-ADUser -Filter * | Select Name,Enabled
- repadmin /replsummary
- Get-Service NTDS,DNS,DHCP
```

### 10. Observability Dashboard

Create a simple log viewer:

```bash
# Recent router decisions
tail -20 logs/router-decisions.jsonl | jq .

# Tool execution success rate
cat logs/tool-output/*.json | jq -s 'group_by(.success) | map({success: .[0].success, count: length})'

# Most common query intents
cat logs/router-decisions.jsonl | jq -s 'group_by(.intent) | map({intent: .[0].intent, count: length})'
```

---

## Implementation Order

1. **Immediate (already done):**
   - [x] Intent classification router
   - [x] ChromaDB helper with proper embeddings
   - [x] MCP bridge script

2. **Short-term (this week):**
   - [ ] Update MCP server to use Python bridge
   - [ ] Fix onnxruntime on Unraid (or use alternative)
   - [ ] Index new context files (homelab-context.yaml, windows-powershell-reference.md)

3. **Medium-term (next 2 weeks):**
   - [ ] Integrate router into ollama-full-toolset.py
   - [ ] Add self-learning from Claude answers
   - [ ] Create system-specific prompt templates

4. **Long-term (next month):**
   - [ ] Collect 500+ training examples
   - [ ] Fine-tune LoRA adapter on homelab data
   - [ ] Build observability dashboard

---

## Quick Wins

### Fix onnxruntime
```bash
# On Unraid, use an older compatible version
ssh unraid 'cd /path/to/hybrid-rag-system && source venv/bin/activate && pip install onnxruntime==1.16.0'
```

### Test the new router
```bash
ssh unraid 'cd /path/to/hybrid-rag-system/scripts && python3 improved-router.py "restart the nginx container"'
```

### Test ChromaDB helper
```bash
ssh unraid 'cd /path/to/hybrid-rag-system && source venv/bin/activate && python3 scripts/chromadb_helper.py query "docker containers" --n-results 3'
```

### Index new context files
```bash
ssh unraid 'cd /path/to/hybrid-rag-system && source venv/bin/activate && \
  python3 scripts/chromadb_helper.py index context/homelab-context.yaml && \
  python3 scripts/chromadb_helper.py index context/infrastructure/windows-powershell-reference.md'
```

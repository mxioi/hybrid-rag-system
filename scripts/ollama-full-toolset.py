#!/usr/bin/env python3
"""
Local LLM Agent w/ Claude-style "context pack" + optional ChromaDB RAG + tools.

UPDATED (Jan 31, 2026) – Integrated sentence-transformer embeddings via chromadb_helper – Hardened for real homelab use:
- Ollama preflight healthcheck (fast fail + clear error)
- Safer Ollama HTTP timeouts (connect/read) + retry/backoff
- Router/tool-run logging to JSONL:
    logs/router-decisions.jsonl
    logs/tool-output/<run_id>.json
- Default Ollama host updated to <ADD-IP-ADDRESS> (your fixed/static IP)

Dependencies (venv):
  pip install requests chromadb
Optional:
  pip install ddgs beautifulsoup4
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ConnectTimeout, ReadTimeout

# Import improved ChromaDB helper (uses sentence-transformers)
try:
    from chromadb_helper import query as chromadb_helper_query, hybrid_query as chromadb_helper_hybrid
    CHROMADB_HELPER_AVAILABLE = True
except ImportError:
    CHROMADB_HELPER_AVAILABLE = False

# -----------------------------
# PATHS (repo-aware)
# -----------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent  # /path/to/hybrid-rag-system
LOGS_DIR = REPO_ROOT / "logs"
TOOL_OUT_DIR = LOGS_DIR / "tool-output"
ROUTER_LOG_PATH = LOGS_DIR / "router-decisions.jsonl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
TOOL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
# NOTE: default updated from .10 -> .210
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "<ADD-IP-ADDRESS>")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

CHROMADB_HOST = os.getenv("CHROMADB_HOST", "<ADD-IP-ADDRESS>")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "homelab_knowledge")

# Your existing directory-RAG pack (already present)
CONTEXT_DIR = Path(os.getenv("CONTEXT_DIR", str(REPO_ROOT / "context")))

# Optional: additional context from directory search (cheap retrieval)
DIR_RAG_SEARCH_GLOBS = [
    "rules/*.md",
    "infrastructure/*.md",
    "prompts/*.md",
    "tools/README.md",
]

# Router/context selection
ROUTER_MAX_CONTEXT_TAGS = 2

CONTEXT_TAG_TO_GLOBS = {
    "rules": ["rules/*.md"],
    "prompts": ["prompts/*.md"],
    "tools": ["tools/*.md", "tools/README.md"],
    "infra:unraid": ["infrastructure/unraid*.md", "infrastructure/*unraid*.md"],
    "infra:proxmox": ["infrastructure/proxmox*.md", "infrastructure/*proxmox*.md"],
    "infra:dc": ["infrastructure/*dc*.md", "infrastructure/*domain*.md"],
    "infra:sccm": ["infrastructure/*sccm*.md", "infrastructure/*mecm*.md"],
    "infra:udm": ["infrastructure/*udm*.md", "infrastructure/*unifi*.md"],
    "infra:network": ["infrastructure/*network*.md", "infrastructure/*vlan*.md"],
    "infra:storage": ["infrastructure/*storage*.md", "infrastructure/*zfs*.md", "infrastructure/*ceph*.md"],
}

FAST_TRIVIAL_PATTERNS = [
    r"^\s*(hi|hello|hey)\s*$",
    r"^\s*what does .* mean\??\s*$",
    r"^\s*define\s+\w+.*$",
    r"^\s*cmdlet to .*",
]

TOOL_REQUIRED_PATTERNS = [
    r"\b(check|verify|confirm|show)\b.*\b(status|health|running|logs?)\b",
    r"\b(disk|storage|space|remaining|free)\b",
    r"\b(df -h|docker ps|systemctl|journalctl)\b",
    r"\bssh\b|\brun\b|\bexecute\b",
]

CONTEXT_REQUIRED_KEYWORDS = {
    "infra:unraid": ["unraid", "/mnt/user", "/mnt/disk", "array"],
    "infra:proxmox": ["proxmox", "pve", "lxc", "qemu"],
    "infra:sccm": ["sccm", "mecm", "configmgr", "dp", "pxe"],
    "infra:dc": ["domain controller", "adcs", "ldap", "gpo", "homelab.local"],
    "infra:udm": ["udm", "unifi", "vlans", "ssid"],
    "infra:network": ["vlan", "subnet", "dhcp", "dns", "gateway"],
    "infra:storage": ["zfs", "ceph", "parity", "cache"],
    "prompts": ["prompt", "system prompt", "router", "context pack"],
    "tools": ["tool", "ollama-full-toolset.py", "script", "function"],
}

# Budgets / limits
MAX_BASELINE_CHARS = 14000
MAX_DIR_RAG_CHARS = 9000
MAX_CHROMA_CHARS = 9000
MAX_TOOL_RESULT_CHARS = 6000


def _is_truthy_env(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
DEFAULT_MAX_ITER = int(os.getenv("MAX_ITERATIONS", "8"))

# OLLAMA_TIMEOUT = total read timeout in seconds (for non-stream calls)
# We split connect/read to avoid hanging on connect.
OLLAMA_CONNECT_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
OLLAMA_READ_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_READ_TIMEOUT", "900"))  # 15 min default
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_CTX = int(os.getenv("OLLAMA_CTX", "4096"))

# If true: prefer deterministic tool first for storage questions
PREFER_DETERMINISTIC_STORAGE = True
RETRIEVAL_PREFERENCE = os.getenv("RETRIEVAL_PREFERENCE", "dir").strip().lower()


# -----------------------------
# UTIL
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    # readable + unique
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_tool_output(run_id: str, payload: Dict[str, Any]) -> Path:
    p = TOOL_OUT_DIR / f"{run_id}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 200] + "\n...[truncated]...\n" + s[-200:]


def _format_file_block(p: Path, content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    return f"--- FILE: {p} ---\n{content}\n"


def _is_storage_question(q: str) -> bool:
    ql = q.lower()
    keywords = ["unraid", "disk", "disk1", "array", "capacity", "free", "remaining", "storage", "space"]
    return any(k in ql for k in keywords)


def _run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


# -----------------------------
# DIRECTORY RAG (Claude-like context pack)
# -----------------------------
def load_context_by_tags(tags: List[str]) -> str:
    if not CONTEXT_DIR.exists() or not tags:
        return ""

    globs: List[str] = []
    for t in tags:
        globs.extend(CONTEXT_TAG_TO_GLOBS.get(t, []))

    blocks: List[str] = []
    for pattern in globs:
        for p in sorted(CONTEXT_DIR.glob(pattern)):
            text = _read_text(p)
            b = _format_file_block(p, text)
            if b:
                blocks.append(b)

    blob = "\n".join(blocks)
    return _clip(blob, MAX_BASELINE_CHARS)


def heuristic_route(q: str) -> Dict[str, Any]:
    ql = q.strip().lower()

    if not ql:
        return {
            "route": "FAST",
            "needs_tools": False,
            "tool": "",
            "needs_context": False,
            "context_tags": [],
            "notes": "empty",
        }

    if any(re.search(p, ql) for p in FAST_TRIVIAL_PATTERNS):
        return {
            "route": "FAST",
            "needs_tools": False,
            "tool": "",
            "needs_context": False,
            "context_tags": [],
            "notes": "trivial",
        }

    needs_tools = any(re.search(p, ql) for p in TOOL_REQUIRED_PATTERNS)

    tags: List[str] = []
    for tag, kws in CONTEXT_REQUIRED_KEYWORDS.items():
        if any(k in ql for k in kws):
            tags.append(tag)

    tags = tags[:ROUTER_MAX_CONTEXT_TAGS]
    needs_context = len(tags) > 0

    if needs_tools or needs_context:
        return {
            "route": "AGENT",
            "needs_tools": needs_tools,
            "tool": "",
            "needs_context": needs_context,
            "context_tags": tags,
            "notes": "needs tools/context",
        }

    return {
        "route": "FAST",
        "needs_tools": False,
        "tool": "",
        "needs_context": False,
        "context_tags": [],
        "notes": "general",
    }


def dir_rag_retrieve(question: str, max_files: int = 8) -> str:
    """Cheap retrieval: grep the context folder for keywords and include matching files."""
    if not CONTEXT_DIR.exists():
        return ""

    rg = shutil.which("rg")
    matches: List[Path] = []

    tokens = [t for t in re.findall(r"[a-zA-Z0-9\-_]{3,}", question.lower())]
    tokens = list(dict.fromkeys(tokens))[:10]  # dedupe + cap
    if not tokens:
        return ""

    candidates: List[Path] = []
    for g in DIR_RAG_SEARCH_GLOBS:
        candidates.extend(CONTEXT_DIR.glob(g))
    candidates = sorted(set(candidates))

    if rg:
        pattern = "|".join(re.escape(t) for t in tokens[:6])
        cmd = [rg, "-l", "-i", pattern, str(CONTEXT_DIR)]
        rc, out, _ = _run(cmd, timeout=10)
        if rc == 0 and out.strip():
            for line in out.splitlines():
                p = Path(line.strip())
                if p.exists() and p.suffix.lower() in {".md", ".txt"}:
                    matches.append(p)
    else:
        for p in candidates:
            txt = _read_text(p).lower()
            if any(t in txt for t in tokens[:6]):
                matches.append(p)

    def score(p: Path) -> Tuple[int, str]:
        s = 3
        if "infrastructure" in p.parts:
            s = 0
        elif "rules" in p.parts:
            s = 1
        elif "prompts" in p.parts:
            s = 2
        return (s, str(p))

    matches = sorted(set(matches), key=score)[:max_files]

    blocks: List[str] = []
    for p in matches:
        blocks.append(_format_file_block(p, _read_text(p)))

    blob = "\n".join(blocks)
    return _clip(blob, MAX_DIR_RAG_CHARS)


# -----------------------------
# CHROMA RAG
# -----------------------------
def chroma_retrieve(question: str, n_results: int = 5, use_hybrid: bool = True) -> str:
    """Retrieve relevant docs from ChromaDB using sentence-transformer embeddings."""
    try:
        if CHROMADB_HELPER_AVAILABLE and use_hybrid:
            # Use improved hybrid search with sentence-transformers
            results = chromadb_helper_hybrid(question, n_results, CHROMADB_COLLECTION)
            blocks: List[str] = []
            for i, r in enumerate(results):
                meta = r.get("metadata", {})
                src = meta.get("source") or "chroma"
                score = r.get("combined_score", 0)
                doc = str(r.get("document", "")).strip()
                blocks.append(f"--- CHROMA CHUNK ({i+1}) source={src} score={score:.3f} ---\n{doc}\n")
            blob = "\n".join(blocks)
            return _clip(blob, MAX_CHROMA_CHARS)

        # Fallback: use chromadb_helper semantic search
        if CHROMADB_HELPER_AVAILABLE:
            res = chromadb_helper_query(question, n_results, CHROMADB_COLLECTION)
            docs = res.get("documents", [[]])[0] or []
            metas = res.get("metadatas", [[]])[0] or []
            dists = res.get("distances", [[]])[0] or []

            blocks: List[str] = []
            for i, d in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else 0
                src = meta.get("source") or "chroma"
                blocks.append(f"--- CHROMA CHUNK ({i+1}) source={src} dist={dist:.3f} ---\n{str(d).strip()}\n")
            blob = "\n".join(blocks)
            return _clip(blob, MAX_CHROMA_CHARS)

        # Final fallback: direct chromadb client
        import chromadb  # type: ignore
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        coll = client.get_collection(CHROMADB_COLLECTION)
        res = coll.query(query_texts=[question], n_results=n_results)
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []

        blocks: List[str] = []
        for i, d in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            src = meta.get("source") or meta.get("path") or "chroma"
            blocks.append(f"--- CHROMA CHUNK ({i+1}) source={src} ---\n{str(d).strip()}\n")
        blob = "\n".join(blocks)
        return _clip(blob, MAX_CHROMA_CHARS)

    except Exception:
        return ""


# -----------------------------
# TOOL DEFINITIONS
# -----------------------------
AVAILABLE_TOOLS: Dict[str, Dict[str, Any]] = {
    # Deterministic homelab helpers
    "UnraidDiskFree": {
        "description": "Get size/used/available for an Unraid disk mount (default /mnt/disk1).",
        "parameters": {"path": "Mount path like /mnt/disk1 (default /mnt/disk1)"},
    },

    # File Ops
    "Read": {"description": "Read file contents", "parameters": {"file_path": "Absolute path"}},
    "Glob": {"description": "Find files matching a glob", "parameters": {"pattern": "Glob pattern"}},
    "Grep": {
        "description": "Search for text in files (uses rg if available).",
        "parameters": {"pattern": "Regex/text", "path": "Path to search (optional)"},
    },

    # System Ops
    "Bash": {
        "description": "Execute a bash command locally (on Unraid where this script runs).",
        "parameters": {"command": "Shell command", "timeout_ms": "Optional timeout ms"},
    },
    "SSH": {
        "description": "Execute command over SSH (requires keys).",
        "parameters": {"host": "Host", "command": "Command", "timeout_s": "Optional timeout seconds"},
    },

    # RAG
    "ChromaDBQuery": {
        "description": "Query ChromaDB for relevant docs.",
        "parameters": {"query": "Search query", "n_results": "Optional int"},
    },
}

# -----------------------------
# TOOL IMPLEMENTATIONS
# -----------------------------
def tool_unraid_disk_free(path: str = "/mnt/disk1") -> Dict[str, Any]:
    cmd = f"df -h {shlex.quote(path)} --output=size,used,avail,pcent,target"
    return tool_bash(command=cmd, timeout_ms=15000)


def tool_read(file_path: str) -> Dict[str, Any]:
    p = Path(os.path.expanduser(file_path))
    if not p.exists():
        return {"success": False, "error": f"File not found: {p}"}
    return {"success": True, "content": _clip(_read_text(p), 12000), "file_path": str(p)}


def tool_glob(pattern: str) -> Dict[str, Any]:
    matches = sorted([str(p) for p in Path(".").glob(pattern)])
    return {"success": True, "matches": matches, "count": len(matches)}


def tool_grep(pattern: str, path: str = ".") -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"success": False, "error": f"Path not found: {p}"}

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", pattern, str(p)]
        rc, out, err = _run(cmd, timeout=15)
        return {"success": True, "exit_code": rc, "stdout": _clip(out, 12000), "stderr": _clip(err, 4000)}
    else:
        cmd = ["grep", "-R", "-n", pattern, str(p)]
        rc, out, err = _run(cmd, timeout=15)
        return {"success": True, "exit_code": rc, "stdout": _clip(out, 12000), "stderr": _clip(err, 4000)}


def tool_bash(command: str, timeout_ms: int = 120000) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000,
        )
        return {
            "success": True,
            "exit_code": p.returncode,
            "stdout": _clip(p.stdout, 12000),
            "stderr": _clip(p.stderr, 4000),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout_ms}ms"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_ssh(host: str, command: str, timeout_s: int = 30) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            ["ssh", host, command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "success": True,
            "exit_code": p.returncode,
            "stdout": _clip(p.stdout, 12000),
            "stderr": _clip(p.stderr, 4000),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_chromadb_query(query: str, n_results: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
    blob = chroma_retrieve(query, n_results=n_results, use_hybrid=use_hybrid)
    if not blob:
        return {"success": False, "error": "Chroma retrieval failed or returned no results."}
    return {"success": True, "content": blob, "count": n_results, "hybrid": use_hybrid}


TOOL_FUNCTIONS = {
    "UnraidDiskFree": tool_unraid_disk_free,
    "Read": tool_read,
    "Glob": tool_glob,
    "Grep": tool_grep,
    "Bash": tool_bash,
    "SSH": tool_ssh,
    "ChromaDBQuery": tool_chromadb_query,
}


def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    try:
        return fn(**parameters)
    except TypeError as e:
        return {"success": False, "error": f"Invalid parameters: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Tool failed: {e}"}


# -----------------------------
# OLLAMA (health + chat)
# -----------------------------
def ollama_healthcheck(session: requests.Session, timeout_s: int = 2) -> Tuple[bool, str]:
    """Fast preflight so we fail cleanly when Ollama is down/unreachable."""
    try:
        r = session.get(f"{OLLAMA_URL}/api/version", timeout=(timeout_s, timeout_s))
        r.raise_for_status()
        return True, r.text.strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def call_ollama(session: requests.Session, messages: List[Dict[str, str]], model: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Returns: (ok, content_or_error, raw_json_if_ok)
    Uses connect/read split timeouts + retry/backoff.
    """
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": OLLAMA_CTX},
    }

    last_err: Optional[str] = None
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            r = session.post(
                url,
                json=payload,
                timeout=(OLLAMA_CONNECT_TIMEOUT_SECONDS, OLLAMA_READ_TIMEOUT_SECONDS),
            )
            r.raise_for_status()
            data = r.json()
            content = data["message"]["content"]
            return True, content, data

        except (ConnectTimeout, ReadTimeout, RequestsConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < OLLAMA_MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            return False, f"Ollama request failed after retries. {last_err}", None

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            return False, f"Ollama request failed. {last_err}", None

    return False, f"Ollama request failed. {last_err or 'unknown error'}", None


# -----------------------------
# OLLAMA AGENT LOOP
# -----------------------------
def build_system_prompt() -> str:
    tools_list = "\n".join(
        f"- {name}: {info['description']} | params={list(info['parameters'].keys())}"
        for name, info in AVAILABLE_TOOLS.items()
    )

    return f"""You are a homelab assistant with tool access (Claude Code style).

You will receive:
- SELECTED CONTEXT (only what is provided)
- OPTIONAL CONTEXT (directory RAG + Chroma chunks)
- USER QUESTION

RULES:
- Prefer deterministic tools over guessing paths.
- For Unraid storage questions: use UnraidDiskFree first (default /mnt/disk1).
- Use tools only when needed. Otherwise answer directly.
- When using a tool, output ONE LINE JSON only:
{{"tool":"ToolName","parameters":{{}},"reason":"<=8 words"}}

Available tools:
{tools_list}
"""


def build_fast_system_prompt() -> str:
    return (
        "Answer briefly.\n"
        "- No tools.\n"
        "- No repo context.\n"
        "- Max 3 sentences.\n"
        'If tools/context are required, reply exactly: "Needs agent route."\n'
    )


def extract_first_tool_json(text: str) -> Optional[Dict[str, Any]]:
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts[:20]:
        for e in range(len(text), s + 1, -1):
            chunk = text[s:e].strip()
            if not chunk.endswith("}"):
                continue
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict) and "tool" in obj and "parameters" in obj:
                    return obj
            except Exception:
                continue
    return None


def validate_tool_req(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj.get("tool"), str) or not obj["tool"]:
        return "tool must be non-empty string"
    if obj["tool"] not in AVAILABLE_TOOLS:
        return "tool not in catalog"
    if not isinstance(obj.get("parameters", {}), dict):
        return "parameters must be object"
    return None


def build_tool_repair_prompt(bad_output: str) -> str:
    tool_names = list(AVAILABLE_TOOLS.keys())
    return (
        "Return ONLY one JSON object on one line.\n"
        'Schema:\n{"tool":"<ToolName>","parameters":{},"reason":"<=8 words"}\n'
        f"Allowed ToolName values: {tool_names}\n"
        "Do not include any other text.\n\n"
        "Bad output to repair:\n"
        + _clip(bad_output, 1500)
    )


def assemble_prompt(
    question: str,
    selected_context: str = "",
    use_dir_rag: bool = True,
    use_chroma: bool = True,
    dir_rag_max_files: int = 8,
    chroma_n_results: int = 5,
) -> str:
    dir_ctx = dir_rag_retrieve(question, max_files=dir_rag_max_files) if use_dir_rag else ""
    chroma_ctx = chroma_retrieve(question, n_results=chroma_n_results) if use_chroma else ""

    parts: List[str] = []
    if selected_context:
        parts.append("SELECTED CONTEXT (read-only):\n" + selected_context)
    if dir_ctx:
        parts.append("DIRECTORY RAG CONTEXT (matched files):\n" + dir_ctx)
    if chroma_ctx:
        parts.append("CHROMADB CONTEXT (semantic matches):\n" + chroma_ctx)

    parts.append("USER QUESTION:\n" + question)
    return "\n\n".join(parts)


def retrieval_policy_from_tags(tags: List[str]) -> Tuple[bool, bool]:
    """
    Returns: (allow_dir_rag, allow_chroma)
    Deterministic: based only on tags.
    """
    tags = tags or []
    has_infra = any(t.startswith("infra:") for t in tags)
    has_promptish = any(t in ("prompts", "tools") for t in tags)

    allow_dir = has_promptish
    allow_chroma = has_infra
    return allow_dir, allow_chroma


def ask_agent(
    question: str,
    model: str = DEFAULT_MODEL,
    max_iterations: int = DEFAULT_MAX_ITER,
    use_dir_rag: bool = True,
    use_chroma: bool = True,
    source: str = "cli",
) -> str:
    run_id = make_run_id()
    start_ts = utc_now_iso()
    start_time = time.time()
    dry_run = _is_truthy_env(os.getenv("ROUTER_DRY_RUN", ""))

    route = heuristic_route(question)
    if dry_run:
        retrieval_guard_applied = "none"
        use_dir_rag_effective = False
        use_chroma_effective = False
        if route.get("needs_context"):
            allow_dir, allow_chroma = retrieval_policy_from_tags(route.get("context_tags", []))
            use_dir_rag_effective = bool(use_dir_rag and allow_dir)
            use_chroma_effective = bool(use_chroma and allow_chroma)
            if use_dir_rag_effective and use_chroma_effective:
                if RETRIEVAL_PREFERENCE == "chroma":
                    use_dir_rag_effective = False
                    retrieval_guard_applied = "disabled_dir"
                else:
                    use_chroma_effective = False
                    retrieval_guard_applied = "disabled_chroma"
        retrieval = {
            "dir_rag": use_dir_rag_effective,
            "chroma": use_chroma_effective,
            "guard_applied": retrieval_guard_applied,
        }
        if route["route"] == "FAST":
            retrieval = {
                "dir_rag": False,
                "chroma": False,
                "guard_applied": "none",
            }
        prompt_chars = len(question)
        context_loaded_chars = 0
        append_jsonl(
            ROUTER_LOG_PATH,
            {
                "ts": start_ts,
                "run_id": run_id,
                "source": source,
                "input": {"text": question},
                "route": route,
                "retrieval": retrieval,
                "prompt_chars": prompt_chars,
                "context_loaded": context_loaded_chars,
                "iterations": 0,
                "agent_loop": route["route"] == "AGENT",
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "policy_version": "ollama-full-toolset@2026-01-27",
                "dry_run": True,
            },
        )
        print(json.dumps(route, ensure_ascii=True))
        raise SystemExit(0)

    session = requests.Session()

    # Preflight check (fast fail)
    ok, ver_or_err = ollama_healthcheck(session, timeout_s=2)
    if not ok:
        append_jsonl(
            ROUTER_LOG_PATH,
            {
                "ts": start_ts,
                "run_id": run_id,
                "source": source,
                "input": {"text": question},
                "prompt_chars": 0,
                "context_loaded": 0,
                "retrieval": {"dir_rag": False, "chroma": False},
                "iterations": 0,
                "agent_loop": False,
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "routing": {
                    "mode": "agentic",
                    "chosen_intent": "explain_only",
                    "confidence": 0.0,
                    "reasons": [{"type": "backend_down", "value": ver_or_err}],
                    "policy_version": "ollama-full-toolset@2026-01-27",
                    "candidates": [],
                },
                "execution": {
                    "tool": None,
                    "status": "error",
                    "error": "ollama_unreachable",
                    "details": ver_or_err,
                    "ollama_url": OLLAMA_URL,
                },
            },
        )
        return (
            "Ollama backend is unreachable right now.\n"
            f"- URL: {OLLAMA_URL}\n"
            f"- Error: {ver_or_err}\n\n"
            "Fix tip: confirm the container/host is up and `curl http://<ollama-ip>:11434/api/version` works."
        )

    if route["route"] == "FAST":
        retrieval = {
            "dir_rag": False,
            "chroma": False,
            "guard_applied": "none",
        }
        prompt_chars = len(question)
        append_jsonl(
            ROUTER_LOG_PATH,
            {
                "ts": start_ts,
                "run_id": run_id,
                "source": source,
                "input": {"text": question},
                "route": route,
                "retrieval": retrieval,
                "prompt_chars": prompt_chars,
                "context_loaded": 0,
                "iterations": 0,
                "agent_loop": False,
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "policy_version": "ollama-full-toolset@2026-01-27",
            },
        )
        # FAST path: do not load any context or RAG.
        convo_fast = [
            {"role": "system", "content": build_fast_system_prompt()},
            {"role": "user", "content": question},
        ]
        llm_ok, assistant_or_err, _raw = call_ollama(session, convo_fast, model=model)
        return assistant_or_err if llm_ok else assistant_or_err
    system_prompt = build_system_prompt()
    use_dir_rag_effective = False
    use_chroma_effective = False
    retrieval_guard_applied = "none"
    if route.get("needs_context"):
        allow_dir, allow_chroma = retrieval_policy_from_tags(route.get("context_tags", []))
        use_dir_rag_effective = bool(use_dir_rag and allow_dir)
        use_chroma_effective = bool(use_chroma and allow_chroma)
        if use_dir_rag_effective and use_chroma_effective:
            if RETRIEVAL_PREFERENCE == "chroma":
                use_dir_rag_effective = False
                retrieval_guard_applied = "disabled_dir"
            else:
                use_chroma_effective = False
                retrieval_guard_applied = "disabled_chroma"
    selected_context = ""
    if route.get("needs_context"):
        selected_context = load_context_by_tags(route.get("context_tags", []))
    context_loaded_chars = len(selected_context)
    retrieval = {
        "dir_rag": use_dir_rag_effective,
        "chroma": use_chroma_effective,
        "guard_applied": retrieval_guard_applied,
    }
    prompt = assemble_prompt(
        question,
        selected_context=selected_context,
        use_dir_rag=use_dir_rag_effective,
        use_chroma=use_chroma_effective,
        dir_rag_max_files=6,
        chroma_n_results=3,
    )
    prompt_chars = len(prompt)
    append_jsonl(
        ROUTER_LOG_PATH,
        {
            "ts": start_ts,
            "run_id": run_id,
            "source": source,
            "input": {"text": question},
            "route": route,
            "retrieval": retrieval,
            "prompt_chars": prompt_chars,
            "context_loaded": context_loaded_chars,
            "iterations": 0,
            "agent_loop": True,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "policy_version": "ollama-full-toolset@2026-01-27",
        },
    )

    # Optional nudge: avoid wandering for storage
    if PREFER_DETERMINISTIC_STORAGE and _is_storage_question(question):
        prompt = (
            "NOTE: This looks like an Unraid storage question. "
            "Use UnraidDiskFree first (default /mnt/disk1) to get remaining capacity.\n\n"
            + prompt
        )

    convo: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Track tool calls for logging
    tool_calls: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    last_assistant: Optional[str] = None
    last_failed_tool: Optional[str] = None
    iterations_run = 0

    for it in range(max_iterations):
        iterations_run += 1
        llm_ok, assistant_or_err, _raw = call_ollama(session, convo, model=model)
        if not llm_ok:
            append_jsonl(
                ROUTER_LOG_PATH,
                {
                    "ts": utc_now_iso(),
                    "run_id": run_id,
                    "source": source,
                    "input": {"text": question},
                    "prompt_chars": prompt_chars,
                    "context_loaded": context_loaded_chars,
                    "retrieval": {"dir_rag": use_dir_rag_effective, "chroma": use_chroma_effective},
                    "iterations": iterations_run,
                    "agent_loop": True,
                    "elapsed_ms": int((time.time() - start_time) * 1000),
                    "routing": {
                        "mode": "agentic",
                        "chosen_intent": "explain_only",
                        "confidence": 0.0,
                        "reasons": [{"type": "ollama_error", "value": assistant_or_err}],
                        "policy_version": "ollama-full-toolset@2026-01-27",
                        "candidates": [],
                    },
                    "execution": {
                        "tool": None,
                        "status": "error",
                        "error": "ollama_call_failed",
                        "details": assistant_or_err,
                        "ollama_url": OLLAMA_URL,
                        "model": model,
                        "iteration": it,
                    },
                    "tools": tool_calls,
                },
            )
            return (
                "Ollama call failed (handled gracefully; no crash).\n"
                f"- Details: {assistant_or_err}\n\n"
                "If this keeps happening, reduce model/context or confirm the Ollama host has enough RAM/CPU."
            )

        assistant = assistant_or_err
        last_assistant = assistant

        tool_req = extract_first_tool_json(assistant)
        if not tool_req:
            final_answer = assistant.strip()
            break
        err = validate_tool_req(tool_req)
        if err:
            repair_prompt = build_tool_repair_prompt(assistant)
            convo_repair = [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": repair_prompt},
            ]
            llm_ok, repair_or_err, _raw = call_ollama(session, convo_repair, model=model)
            if not llm_ok:
                final_answer = "Invalid tool request."
                break
            tool_req = extract_first_tool_json(repair_or_err or "")
            err = validate_tool_req(tool_req or {})
            if err:
                final_answer = "Invalid tool request."
                break

        tool = str(tool_req.get("tool"))
        params = tool_req.get("parameters", {}) or {}
        reason = str(tool_req.get("reason", ""))

        if last_failed_tool and tool == last_failed_tool:
            final_answer = "Invalid tool request."
            break

        t_start = time.time()
        result = execute_tool(tool, params)
        t_end = time.time()

        tool_out_path = save_tool_output(
            run_id=f"{run_id}-it{it+1}",
            payload={
                "tool": tool,
                "parameters": params,
                "reason": reason,
                "result": result,
            },
        )

        tool_success = bool(result.get("success", True)) if isinstance(result, dict) else True
        tool_calls.append(
            {
                "tool": tool,
                "parameters": params,
                "reason": reason,
                "duration_ms": int((t_end - t_start) * 1000),
                "success": tool_success,
                "saved_path": str(tool_out_path),
                "sha256": sha256_text(json.dumps(result, sort_keys=True, ensure_ascii=False)) if isinstance(result, dict) else None,
            }
        )

        # Truncate tool output to keep ctx sane
        result_text = _clip(json.dumps(result, indent=2, ensure_ascii=False), MAX_TOOL_RESULT_CHARS)

        convo.append({"role": "assistant", "content": assistant})
        convo.append(
            {
                "role": "user",
                "content": (
                    f"Tool result:\n```json\n{result_text}\n```\n\n"
                    "Continue. If you have enough info, answer directly without tools."
                ),
            }
        )
        if not tool_success:
            last_failed_tool = tool
            convo.append(
                {
                    "role": "user",
                    "content": (
                        "Tool failed. Do NOT retry the same tool. "
                        "Use a different tool or answer with what you have."
                    ),
                }
            )

    end_ts = utc_now_iso()

    if final_answer is None:
        final_answer = "Max iterations reached. Provide best effort answer with what you have."
        if last_assistant and last_assistant.strip():
            final_answer += "\n\nLast assistant message:\n" + _clip(last_assistant.strip(), 2500)

    append_jsonl(
        ROUTER_LOG_PATH,
        {
            "ts": start_ts,
            "ts_end": end_ts,
            "run_id": run_id,
            "source": source,
            "input": {"text": question},
            "prompt_chars": prompt_chars,
            "context_loaded": context_loaded_chars,
            "retrieval": {"dir_rag": use_dir_rag_effective, "chroma": use_chroma_effective},
            "iterations": iterations_run,
            "agent_loop": True,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "routing": {
                "mode": "agentic",
                "chosen_intent": "tool_or_answer",
                "confidence": 0.5,
                "reasons": [{"type": "ollama_version", "value": ver_or_err}],
                "policy_version": "ollama-full-toolset@2026-01-27",
                "candidates": [],
            },
            "execution": {
                "status": "ok",
                "ollama_url": OLLAMA_URL,
                "model": model,
                "max_iterations": max_iterations,
                "use_dir_rag": use_dir_rag,
                "use_chroma": use_chroma,
            },
            "tools": tool_calls,
            "output": {
                "bytes": len(final_answer.encode("utf-8", errors="ignore")),
                "sha256": sha256_text(final_answer),
            },
        },
    )

    return final_answer


# -----------------------------
# MAIN
# -----------------------------
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="User question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-dir-rag", action="store_true", help="Disable directory RAG extra context")
    ap.add_argument("--no-chroma", action="store_true", help="Disable Chroma semantic retrieval")
    ap.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--source", default="cli", help="Log source tag (cli/webui/n8n/discord)")
    args = ap.parse_args()

    q = " ".join(args.question).strip()
    out = ask_agent(
        q,
        model=args.model,
        max_iterations=args.max_iter,
        use_dir_rag=not args.no_dir_rag,
        use_chroma=not args.no_chroma,
        source=args.source,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

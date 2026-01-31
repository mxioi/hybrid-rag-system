#!/usr/bin/env python3
"""
Improved Local LLM Router with:
- Intent classification (tool-required vs RAG vs direct)
- Query complexity scoring
- System-specific routing
- Confidence-based fallback
- Structured logging for fine-tuning data collection

Author: Claude Code
Version: 1.0.0
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "<ADD-IP-ADDRESS>")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "<ADD-IP-ADDRESS>")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

LOGS_DIR = Path(os.getenv("LOGS_DIR", "/path/to/hybrid-rag-system/logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

class QueryIntent(Enum):
    TOOL_REQUIRED = "tool_required"      # Must execute a tool/command
    RAG_LOOKUP = "rag_lookup"            # Documentation lookup
    CONVERSATIONAL = "conversational"    # General chat/explanation
    HYBRID = "hybrid"                    # RAG + possible tool
    REFUSED = "refused"                  # Blocked query


class QueryComplexity(Enum):
    TRIVIAL = "trivial"       # Single fact lookup
    SIMPLE = "simple"         # Single step task
    MODERATE = "moderate"     # Multi-step, single system
    COMPLEX = "complex"       # Multi-system or multi-step


class TargetSystem(Enum):
    UNRAID = "unraid"
    PROXMOX = "proxmox"
    WINDOWS_AD = "windows_ad"
    WINDOWS_SCCM = "windows_sccm"
    WINDOWS_EXCHANGE = "windows_exchange"
    NETWORK = "network"
    OLLAMA = "ollama"
    CHROMADB = "chromadb"
    GENERAL = "general"


@dataclass
class QueryClassification:
    """Classification result for a query"""
    intent: QueryIntent
    complexity: QueryComplexity
    target_systems: List[TargetSystem]
    confidence: float
    tool_hints: List[str]
    context_tags: List[str]
    reasoning: str


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Patterns that REQUIRE tool execution (not just RAG)
TOOL_REQUIRED_PATTERNS = [
    # Status checks
    (r"\b(check|verify|show|get|list)\b.*\b(status|health|running|state)\b", ["Bash", "SSH"]),
    (r"\bis\b.*\b(running|up|down|healthy|online|offline)\b", ["Bash", "SSH"]),

    # Resource queries
    (r"\b(disk|storage|space|memory|cpu|ram)\b.*\b(usage|free|available|remaining)\b", ["Bash"]),
    (r"\bhow much (space|memory|ram|disk)\b", ["Bash"]),

    # Container operations
    (r"\b(docker|container)\b.*\b(logs?|restart|stop|start|status)\b", ["Bash"]),
    (r"\brestart\b.*\b(container|service|docker)\b", ["Bash"]),

    # Service operations
    (r"\b(service|daemon)\b.*\b(start|stop|restart|status)\b", ["Bash", "SSH"]),

    # File operations
    (r"\b(read|show|cat|display)\b.*\b(file|config|log)\b", ["Read", "Bash"]),
    (r"\b(list|ls|find)\b.*\b(files?|directories|folders?)\b", ["Glob", "Bash"]),

    # Network checks
    (r"\b(ping|traceroute|curl|wget|nc)\b", ["Bash"]),
    (r"\bcan.*\b(reach|connect|access)\b", ["Bash"]),

    # Process queries
    (r"\bwhat.*\b(process|running|listening)\b", ["Bash"]),
    (r"\bport\b.*\b(open|listening|used)\b", ["Bash"]),
]

# Patterns for RAG-only queries
RAG_PATTERNS = [
    r"\bwhat is\b",
    r"\bhow (do|does|to|can)\b",
    r"\bexplain\b",
    r"\bwhy\b.*\b(does|is|do)\b",
    r"\bdescribe\b",
    r"\bwhat are the\b.*\b(steps|options|ways)\b",
    r"\bdocumentation\b",
    r"\bguide\b",
    r"\btutorial\b",
]

# Conversational patterns (no tools or RAG needed)
CONVERSATIONAL_PATTERNS = [
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay)\s*[!.]?\s*$",
    r"^\s*(yes|no|sure|great)\s*[!.]?\s*$",
    r"^\s*what can you do\b",
    r"^\s*who are you\b",
]

# Patterns that should be refused
REFUSAL_PATTERNS = [
    (r"\b(rm -rf|delete|remove)\b.*\b(/|all|everything|root)\b", "Destructive operation"),
    (r"\b(password|secret|credential|api.?key|token)\b.*\b(show|display|print|get)\b", "Credential exposure"),
    (r"\bdrop\b.*\b(database|table)\b", "Destructive database operation"),
    (r"\bformat\b.*\b(disk|drive|partition)\b", "Disk format operation"),
]

# System detection patterns
SYSTEM_PATTERNS = {
    TargetSystem.UNRAID: [
        r"\bunraid\b", r"\b/mnt/(user|cache|disk)\b", r"\barray\b", r"\bshares?\b",
        r"\bdocker\b", r"\bparity\b", r"\bappdata\b",
    ],
    TargetSystem.PROXMOX: [
        r"\bproxmox\b", r"\bpve\b", r"\blxc\b", r"\bqemu\b", r"\bvm\b",
        r"\bpct\b", r"\bqm\b", r"\bpbs\b", r"\bbackup server\b",
    ],
    TargetSystem.WINDOWS_AD: [
        r"\bactive directory\b", r"\bad\b", r"\bdomain controller\b", r"\bdc[12]?\b",
        r"\bldap\b", r"\bgpo\b", r"\bgroup policy\b", r"\breplication\b",
        r"\bfsmo\b", r"\bntds\b", r"\bdns\b", r"\bdhcp\b",
    ],
    TargetSystem.WINDOWS_SCCM: [
        r"\bsccm\b", r"\bmecm\b", r"\bconfigmgr\b", r"\bapp1\b",
        r"\bsms_\b", r"\bwsus\b", r"\bdeployment\b", r"\bpackage\b",
    ],
    TargetSystem.WINDOWS_EXCHANGE: [
        r"\bexchange\b", r"\bexc1\b", r"\bmailbox\b", r"\bemail\b",
        r"\bmail queue\b", r"\bownership\b",
    ],
    TargetSystem.NETWORK: [
        r"\budm\b", r"\bunifi\b", r"\bvlan\b", r"\bsubnet\b",
        r"\bfirewall\b", r"\brouter\b", r"\bswitch\b", r"\bap\b",
    ],
    TargetSystem.OLLAMA: [
        r"\bollama\b", r"\bmodel\b", r"\bllm\b", r"\binference\b",
    ],
    TargetSystem.CHROMADB: [
        r"\bchromadb\b", r"\bvector\b", r"\bembedding\b", r"\brag\b",
        r"\bcollection\b", r"\bindex\b",
    ],
}

# Context tags for RAG filtering
CONTEXT_KEYWORDS = {
    "docker": ["docker", "container", "image", "compose", "dockerfile"],
    "storage": ["disk", "storage", "zfs", "array", "parity", "cache", "share"],
    "networking": ["network", "vlan", "ip", "dns", "dhcp", "firewall", "port"],
    "backup": ["backup", "pbs", "restore", "snapshot", "replicate"],
    "security": ["auth", "ssl", "certificate", "password", "permission", "acl"],
    "monitoring": ["monitor", "alert", "log", "metric", "grafana", "prometheus"],
}


# =============================================================================
# CLASSIFIER
# =============================================================================

def classify_query(query: str) -> QueryClassification:
    """
    Classify a query to determine:
    - Intent (tool required, RAG lookup, conversational)
    - Complexity
    - Target systems
    - Recommended tools
    - Context tags for RAG filtering
    """
    query_lower = query.lower().strip()

    # Check for refusal patterns first
    for pattern, reason in REFUSAL_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                intent=QueryIntent.REFUSED,
                complexity=QueryComplexity.TRIVIAL,
                target_systems=[],
                confidence=0.95,
                tool_hints=[],
                context_tags=[],
                reasoning=f"Query blocked: {reason}"
            )

    # Check for conversational patterns
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                intent=QueryIntent.CONVERSATIONAL,
                complexity=QueryComplexity.TRIVIAL,
                target_systems=[TargetSystem.GENERAL],
                confidence=0.9,
                tool_hints=[],
                context_tags=[],
                reasoning="Conversational query, no tools or RAG needed"
            )

    # Detect target systems
    target_systems = []
    for system, patterns in SYSTEM_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                if system not in target_systems:
                    target_systems.append(system)
                break

    if not target_systems:
        target_systems = [TargetSystem.GENERAL]

    # Check for tool-required patterns
    tool_hints = []
    tool_required = False
    for pattern, tools in TOOL_REQUIRED_PATTERNS:
        if re.search(pattern, query_lower):
            tool_required = True
            for tool in tools:
                if tool not in tool_hints:
                    tool_hints.append(tool)

    # Check for RAG patterns
    rag_likely = any(re.search(p, query_lower) for p in RAG_PATTERNS)

    # Determine intent
    if tool_required and rag_likely:
        intent = QueryIntent.HYBRID
    elif tool_required:
        intent = QueryIntent.TOOL_REQUIRED
    elif rag_likely:
        intent = QueryIntent.RAG_LOOKUP
    else:
        # Default to hybrid for ambiguous queries
        intent = QueryIntent.HYBRID

    # Determine complexity
    complexity = determine_complexity(query_lower, target_systems, tool_hints)

    # Determine context tags for RAG filtering
    context_tags = []
    for tag, keywords in CONTEXT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            context_tags.append(tag)

    # Calculate confidence based on pattern match strength
    confidence = calculate_classification_confidence(
        query_lower, intent, tool_hints, context_tags
    )

    # Build reasoning
    reasoning = build_reasoning(intent, target_systems, tool_hints, context_tags)

    return QueryClassification(
        intent=intent,
        complexity=complexity,
        target_systems=target_systems,
        confidence=confidence,
        tool_hints=tool_hints,
        context_tags=context_tags,
        reasoning=reasoning
    )


def determine_complexity(query: str, systems: List[TargetSystem], tools: List[str]) -> QueryComplexity:
    """Determine query complexity based on various factors"""

    # Multiple systems = complex
    if len(systems) > 1 and TargetSystem.GENERAL not in systems:
        return QueryComplexity.COMPLEX

    # Multiple tools = moderate to complex
    if len(tools) >= 3:
        return QueryComplexity.COMPLEX
    elif len(tools) >= 2:
        return QueryComplexity.MODERATE

    # Long queries tend to be more complex
    word_count = len(query.split())
    if word_count > 20:
        return QueryComplexity.COMPLEX
    elif word_count > 10:
        return QueryComplexity.MODERATE

    # Check for multi-step indicators
    multi_step_patterns = [
        r"\band\s+then\b",
        r"\bfirst.*then\b",
        r"\bafter.*do\b",
        r"\bmultiple\b",
        r"\ball\b.*\b(servers?|systems?|vms?)\b",
    ]
    if any(re.search(p, query) for p in multi_step_patterns):
        return QueryComplexity.MODERATE

    # Simple patterns
    simple_patterns = [
        r"^(what|show|list|get)\s+\w+$",
        r"^is\s+\w+\s+(running|up|down)\??$",
    ]
    if any(re.search(p, query) for p in simple_patterns):
        return QueryComplexity.TRIVIAL

    return QueryComplexity.SIMPLE


def calculate_classification_confidence(
    query: str,
    intent: QueryIntent,
    tools: List[str],
    tags: List[str]
) -> float:
    """Calculate confidence in the classification"""
    confidence = 0.5  # Base confidence

    # Boost for clear patterns
    if intent == QueryIntent.TOOL_REQUIRED and tools:
        confidence += 0.2
    if intent == QueryIntent.RAG_LOOKUP and tags:
        confidence += 0.15

    # Boost for specific system mentions
    specific_system_patterns = [
        r"\bunraid\b", r"\bproxmox\b", r"\bdc[12]\b",
        r"\bapp1\b", r"\bexc1\b", r"\budm\b"
    ]
    if any(re.search(p, query) for p in specific_system_patterns):
        confidence += 0.15

    # Reduce for very short or vague queries
    if len(query.split()) < 3:
        confidence -= 0.1

    return min(max(confidence, 0.1), 0.99)


def build_reasoning(
    intent: QueryIntent,
    systems: List[TargetSystem],
    tools: List[str],
    tags: List[str]
) -> str:
    """Build human-readable reasoning for the classification"""
    parts = []

    parts.append(f"Intent: {intent.value}")

    if systems and systems != [TargetSystem.GENERAL]:
        system_names = [s.value for s in systems]
        parts.append(f"Systems: {', '.join(system_names)}")

    if tools:
        parts.append(f"Suggested tools: {', '.join(tools)}")

    if tags:
        parts.append(f"Context tags: {', '.join(tags)}")

    return "; ".join(parts)


# =============================================================================
# QUERY ROUTER
# =============================================================================

@dataclass
class RoutingDecision:
    """Decision on how to handle a query"""
    classification: QueryClassification
    use_rag: bool
    use_tools: bool
    rag_filter: Optional[Dict[str, Any]]
    tool_sequence: List[str]
    fallback_to_claude: bool
    run_id: str
    timestamp: str


def route_query(query: str) -> RoutingDecision:
    """
    Route a query to the appropriate handling strategy
    """
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    classification = classify_query(query)

    # Determine RAG usage
    use_rag = classification.intent in [
        QueryIntent.RAG_LOOKUP,
        QueryIntent.HYBRID
    ]

    # Determine tool usage
    use_tools = classification.intent in [
        QueryIntent.TOOL_REQUIRED,
        QueryIntent.HYBRID
    ]

    # Build RAG filter based on context tags and systems
    rag_filter = None
    if use_rag and (classification.context_tags or
                    classification.target_systems != [TargetSystem.GENERAL]):
        rag_filter = build_rag_filter(classification)

    # Determine tool sequence
    tool_sequence = classification.tool_hints if use_tools else []

    # Determine if Claude fallback should be enabled
    fallback_to_claude = (
        classification.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MODERATE] or
        classification.confidence < 0.7
    )

    decision = RoutingDecision(
        classification=classification,
        use_rag=use_rag,
        use_tools=use_tools,
        rag_filter=rag_filter,
        tool_sequence=tool_sequence,
        fallback_to_claude=fallback_to_claude,
        run_id=run_id,
        timestamp=timestamp
    )

    # Log the decision
    log_routing_decision(decision, query)

    return decision


def build_rag_filter(classification: QueryClassification) -> Dict[str, Any]:
    """Build ChromaDB filter based on classification"""
    filters = {}

    # Filter by system if specific
    if classification.target_systems and classification.target_systems != [TargetSystem.GENERAL]:
        system_values = [s.value for s in classification.target_systems]
        if len(system_values) == 1:
            filters["system"] = system_values[0]
        else:
            filters["$or"] = [{"system": s} for s in system_values]

    # Could also filter by category/context_tags if metadata supports it

    return filters if filters else None


def log_routing_decision(decision: RoutingDecision, query: str):
    """Log routing decision for analysis and fine-tuning data collection"""
    log_entry = {
        "run_id": decision.run_id,
        "timestamp": decision.timestamp,
        "query": query[:500],  # Truncate long queries
        "intent": decision.classification.intent.value,
        "complexity": decision.classification.complexity.value,
        "target_systems": [s.value for s in decision.classification.target_systems],
        "confidence": decision.classification.confidence,
        "tool_hints": decision.classification.tool_hints,
        "context_tags": decision.classification.context_tags,
        "use_rag": decision.use_rag,
        "use_tools": decision.use_tools,
        "fallback_enabled": decision.fallback_to_claude,
        "reasoning": decision.classification.reasoning,
    }

    log_file = LOGS_DIR / "router-decisions.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log routing decision: {e}")


# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

def build_system_prompt(classification: QueryClassification) -> str:
    """Build a dynamic system prompt based on query classification"""

    base_prompt = """You are an expert homelab systems administrator assistant. You help manage infrastructure including:
- Unraid NAS (Docker, storage, shares)
- Proxmox virtualization (VMs, LXCs, PBS backups)
- Windows Server (Active Directory, SCCM, Exchange)
- Network infrastructure (UniFi, VLANs, DNS)

Guidelines:
1. Be concise and actionable
2. Provide exact commands when relevant
3. Explain potential impacts of operations
4. Ask for clarification if the request is ambiguous
"""

    # Add system-specific context
    system_contexts = {
        TargetSystem.UNRAID: """
For Unraid:
- Array is at /mnt/user (user shares) and /mnt/disk* (individual disks)
- Docker containers are managed via docker CLI or Unraid UI
- Appdata is at /mnt/user/appdata
- Server hostname: <ADD-HOSTNAME> (<ADD-IP-ADDRESS>)
""",
        TargetSystem.PROXMOX: """
For Proxmox:
- Access via: ssh proxmox (10.0.0.X)
- Use pvesh, qm, pct commands
- PBS is in LXC 200
- Ollama is in LXC 300 (<ADD-IP-ADDRESS>)
""",
        TargetSystem.WINDOWS_AD: """
For Active Directory:
- DC1: <ADD-IP-ADDRESS> (primary, has RSAT)
- DC2: <ADD-IP-ADDRESS> (secondary)
- Domain: homelab.local
- Access via: ssh DC1 "powershell -Command '...'"
- Use RSAT cmdlets: Get-ADUser, Get-ADComputer, Get-ADDomainController
""",
        TargetSystem.WINDOWS_SCCM: """
For SCCM/MECM:
- Server: APP1 (10.0.0.X)
- Services: SMS_*, WSUS, SQL Server
- Access via: ssh APP1 "powershell -Command '...'"
""",
    }

    # Append relevant system contexts
    for system in classification.target_systems:
        if system in system_contexts:
            base_prompt += system_contexts[system]

    # Add tool guidance if tools are required
    if classification.intent == QueryIntent.TOOL_REQUIRED:
        base_prompt += """
IMPORTANT: This query requires executing actual commands. You MUST:
1. Execute the appropriate tool/command
2. Show the actual output
3. Do not guess or assume the result
"""

    return base_prompt


# =============================================================================
# FINE-TUNING DATA COLLECTION
# =============================================================================

@dataclass
class TrainingExample:
    """A training example for fine-tuning"""
    query: str
    response: str
    classification: Dict[str, Any]
    success: bool
    feedback_score: Optional[float]
    tools_used: List[str]
    context_used: List[str]
    timestamp: str


def log_training_example(
    query: str,
    response: str,
    classification: QueryClassification,
    success: bool,
    tools_used: List[str] = None,
    context_used: List[str] = None,
    feedback_score: float = None
):
    """Log a training example for future fine-tuning"""

    example = TrainingExample(
        query=query,
        response=response,
        classification=asdict(classification),
        success=success,
        feedback_score=feedback_score,
        tools_used=tools_used or [],
        context_used=context_used or [],
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    # Convert enums to strings for JSON
    example_dict = asdict(example)
    example_dict["classification"]["intent"] = classification.intent.value
    example_dict["classification"]["complexity"] = classification.complexity.value
    example_dict["classification"]["target_systems"] = [s.value for s in classification.target_systems]

    log_file = LOGS_DIR / "training-examples.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(example_dict) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log training example: {e}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI interface for testing the router"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python improved-router.py 'your query'")
        print("\nExamples:")
        print("  python improved-router.py 'check disk space on unraid'")
        print("  python improved-router.py 'what is AD replication'")
        print("  python improved-router.py 'restart the nginx container'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f"\nQuery: {query}")
    print("=" * 60)

    decision = route_query(query)

    print(f"\nClassification:")
    print(f"  Intent:      {decision.classification.intent.value}")
    print(f"  Complexity:  {decision.classification.complexity.value}")
    print(f"  Systems:     {[s.value for s in decision.classification.target_systems]}")
    print(f"  Confidence:  {decision.classification.confidence:.2f}")

    print(f"\nRouting Decision:")
    print(f"  Use RAG:     {decision.use_rag}")
    print(f"  Use Tools:   {decision.use_tools}")
    print(f"  Tools:       {decision.tool_sequence}")
    print(f"  RAG Filter:  {decision.rag_filter}")
    print(f"  Fallback:    {decision.fallback_to_claude}")

    print(f"\nReasoning: {decision.classification.reasoning}")
    print(f"Run ID: {decision.run_id}")

    # Generate system prompt
    print(f"\n{'=' * 60}")
    print("Generated System Prompt (first 500 chars):")
    print("-" * 40)
    prompt = build_system_prompt(decision.classification)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)


if __name__ == "__main__":
    main()

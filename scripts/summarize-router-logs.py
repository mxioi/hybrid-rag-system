#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class DataHealth:
    missing_file: bool = False
    empty_file: bool = False
    parse_errors: int = 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    vals_sorted = sorted(vals)
    if p <= 0:
        return vals_sorted[0]
    if p >= 100:
        return vals_sorted[-1]
    n = len(vals_sorted)
    k = max(0, min(n - 1, math.ceil((p / 100) * n) - 1))
    return vals_sorted[k]


def _avg(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _extract_request_text(obj: Dict[str, Any]) -> str:
    v = obj.get("request_text")
    if isinstance(v, str):
        return v
    if "input" in obj and isinstance(obj["input"], dict):
        txt = obj["input"].get("text")
        if isinstance(txt, str):
            return txt
    for key in ("input_text", "question", "prompt", "request"):
        v = obj.get(key)
        if isinstance(v, str):
            return v
    return ""


def _extract_route(obj: Dict[str, Any]) -> str:
    r = obj.get("route")
    if isinstance(r, str):
        val = r.strip().lower()
        if val in ("fast", "agent"):
            return val.upper()
        if val in ("agentic", "tool_or_answer"):
            return "AGENT"
        if val in ("explain_only",):
            return "FAST"
        return r.strip().upper()
    if isinstance(r, dict):
        rr = r.get("route")
        if isinstance(rr, str):
            val = rr.strip().lower()
            if val in ("fast", "agent"):
                return val.upper()
            if val in ("agentic", "tool_or_answer"):
                return "AGENT"
            if val in ("explain_only",):
                return "FAST"
            return rr.strip().upper()
    routing = obj.get("routing")
    if isinstance(routing, dict):
        rr = routing.get("chosen_intent") or routing.get("route") or routing.get("mode")
        if isinstance(rr, str):
            val = rr.strip().lower()
            if val in ("fast", "agent"):
                return val.upper()
            if val in ("agentic", "tool_or_answer"):
                return "AGENT"
            if val in ("explain_only",):
                return "FAST"
            return rr.strip().upper()
    return ""


def _extract_bool(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[bool]:
    for key in keys:
        v = obj.get(key)
        if isinstance(v, bool):
            return v
    return None


def _extract_nested_bool(obj: Dict[str, Any], path: Tuple[str, ...]) -> Optional[bool]:
    cur: Any = obj
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur if isinstance(cur, bool) else None


def _extract_prompt_chars(obj: Dict[str, Any]) -> Optional[int]:
    v = obj.get("prompt_chars")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if "prompt" in obj and isinstance(obj["prompt"], dict):
        v2 = obj["prompt"].get("chars")
        if isinstance(v2, (int, float)):
            return int(v2)
    return None


def _extract_tool_calls(obj: Dict[str, Any]) -> Optional[int]:
    v = obj.get("tool_calls")
    if isinstance(v, int):
        return v
    if isinstance(v, list):
        return len(v)
    if "tools" in obj and isinstance(obj["tools"], list):
        return len(obj["tools"])
    return None


def _extract_iterations(obj: Dict[str, Any]) -> Optional[int]:
    v = obj.get("iterations")
    if isinstance(v, int):
        return v
    if "execution" in obj and isinstance(obj["execution"], dict):
        v2 = obj["execution"].get("iterations")
        if isinstance(v2, int):
            return v2
    return None


def _extract_retrieval(obj: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    if "retrieval" in obj and isinstance(obj["retrieval"], dict):
        dir_rag = obj["retrieval"].get("dir_rag")
        chroma = obj["retrieval"].get("chroma")
        return (dir_rag if isinstance(dir_rag, bool) else None, chroma if isinstance(chroma, bool) else None)
    return (None, None)


def _extract_latency(obj: Dict[str, Any], key: str) -> Optional[float]:
    v = obj.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _format_float(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _format_int(v: Optional[int]) -> str:
    return "n/a" if v is None else str(v)


def _coverage(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(count / total) * 100:.1f}%"


def summarize_router_logs(log_path: Path) -> Tuple[Dict[str, Any], DataHealth]:
    health = DataHealth()
    if not log_path.exists():
        health.missing_file = True
        return {}, health

    raw_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not raw_lines:
        health.empty_file = True
        return {}, health

    total_entries = 0
    fast_count = 0
    agent_count = 0
    other_route_count = 0

    prompt_chars_vals: List[int] = []
    context_flags: List[bool] = []
    dir_rag_flags: List[bool] = []
    chroma_flags: List[bool] = []
    tool_calls_vals: List[int] = []
    iterations_vals: List[int] = []
    agent_loop_flags: List[bool] = []

    latency_ms_vals: List[float] = []
    duration_ms_vals: List[float] = []
    elapsed_ms_vals: List[float] = []

    request_prefixes = Counter()
    first_words = Counter()
    longest_requests: List[Tuple[int, str]] = []

    for line in raw_lines:
        if not line.strip():
            continue
        obj = _safe_json_loads(line.strip())
        if obj is None:
            health.parse_errors += 1
            continue

        total_entries += 1

        route = _extract_route(obj)
        if route == "FAST":
            fast_count += 1
        elif route == "AGENT":
            agent_count += 1
        else:
            other_route_count += 1

        pc = _extract_prompt_chars(obj)
        if pc is not None:
            prompt_chars_vals.append(pc)

        context_loaded = _extract_bool(obj, ("context_loaded", "needs_context"))
        if context_loaded is None:
            context_loaded = _extract_nested_bool(obj, ("route", "needs_context"))
        if context_loaded is not None:
            context_flags.append(context_loaded)

        dir_rag, chroma = _extract_retrieval(obj)
        if dir_rag is None:
            dir_rag = _extract_nested_bool(obj, ("retrieval", "dir_rag"))
        if chroma is None:
            chroma = _extract_nested_bool(obj, ("retrieval", "chroma"))
        if dir_rag is not None:
            dir_rag_flags.append(dir_rag)
        if chroma is not None:
            chroma_flags.append(chroma)

        tc = _extract_tool_calls(obj)
        if tc is not None:
            tool_calls_vals.append(tc)

        iters = _extract_iterations(obj)
        if iters is not None:
            iterations_vals.append(iters)

        agent_loop = _extract_bool(obj, ("agent_loop",))
        if agent_loop is not None:
            agent_loop_flags.append(agent_loop)

        lm = _extract_latency(obj, "latency_ms")
        if lm is not None:
            latency_ms_vals.append(lm)
        dm = _extract_latency(obj, "duration_ms")
        if dm is not None:
            duration_ms_vals.append(dm)
        em = _extract_latency(obj, "elapsed_ms")
        if em is not None:
            elapsed_ms_vals.append(em)

        req_text = _extract_request_text(obj)
        req_norm = _normalize_whitespace(req_text)
        if req_norm:
            prefix = req_norm[:24]
            request_prefixes[prefix] += 1
            first_word = req_norm.split(" ", 1)[0].lower()
            first_words[first_word] += 1
            longest_requests.append((len(req_norm), req_norm))

    longest_requests = sorted(longest_requests, key=lambda x: x[0], reverse=True)[:10]

    metrics = {
        "total_entries": total_entries,
        "fast_count": fast_count,
        "agent_count": agent_count,
        "other_route_count": other_route_count,
        "prompt_chars_vals": prompt_chars_vals,
        "context_flags": context_flags,
        "dir_rag_flags": dir_rag_flags,
        "chroma_flags": chroma_flags,
        "tool_calls_vals": tool_calls_vals,
        "iterations_vals": iterations_vals,
        "agent_loop_flags": agent_loop_flags,
        "latency_ms_vals": latency_ms_vals,
        "duration_ms_vals": duration_ms_vals,
        "elapsed_ms_vals": elapsed_ms_vals,
        "request_prefixes": request_prefixes,
        "first_words": first_words,
        "longest_requests": longest_requests,
    }
    return metrics, health


def summarize_violations(violations_dir: Path) -> Dict[str, Any]:
    out = {
        "present": False,
        "total_lines": 0,
        "parse_errors": 0,
        "counts_by_type": Counter(),
        "most_recent_file": "none",
    }
    if not violations_dir.exists() or not violations_dir.is_dir():
        return out

    files = sorted(violations_dir.glob("*.jsonl"))
    if not files:
        return out

    out["present"] = True
    most_recent = max(files, key=lambda p: p.stat().st_mtime)
    out["most_recent_file"] = most_recent.name

    for fp in files:
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            obj = _safe_json_loads(line.strip())
            if obj is None:
                out["parse_errors"] += 1
                continue
            out["total_lines"] += 1
            vtype = obj.get("violation_type") or obj.get("type") or "unknown"
            out["counts_by_type"][str(vtype)] += 1

    return out


def render_report(metrics: Dict[str, Any], health: DataHealth, violations: Dict[str, Any], ts: datetime) -> str:
    total_entries = metrics.get("total_entries", 0)
    fast_count = metrics.get("fast_count", 0)
    agent_count = metrics.get("agent_count", 0)
    other_route_count = metrics.get("other_route_count", 0)

    fast_ratio = (fast_count / total_entries) if total_entries else 0.0
    agent_ratio = (agent_count / total_entries) if total_entries else 0.0

    prompt_chars_vals = metrics.get("prompt_chars_vals", [])
    context_flags = metrics.get("context_flags", [])
    dir_rag_flags = metrics.get("dir_rag_flags", [])
    chroma_flags = metrics.get("chroma_flags", [])
    tool_calls_vals = metrics.get("tool_calls_vals", [])
    iterations_vals = metrics.get("iterations_vals", [])
    agent_loop_flags = metrics.get("agent_loop_flags", [])

    latency_ms_vals = metrics.get("latency_ms_vals", [])
    duration_ms_vals = metrics.get("duration_ms_vals", [])
    elapsed_ms_vals = metrics.get("elapsed_ms_vals", [])

    request_prefixes: Counter = metrics.get("request_prefixes", Counter())
    first_words: Counter = metrics.get("first_words", Counter())
    longest_requests: List[Tuple[int, str]] = metrics.get("longest_requests", [])

    report = []
    report.append("# Router Metrics Report")
    report.append("")
    report.append(f"- Generated (UTC): `{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    report.append("")

    report.append("## Volume & routing")
    report.append("")
    report.append("| metric | value |")
    report.append("|---|---:|")
    report.append(f"| total_entries | {total_entries} |")
    report.append(f"| fast_count | {fast_count} |")
    report.append(f"| agent_count | {agent_count} |")
    report.append(f"| other_route_count | {other_route_count} |")
    report.append(f"| fast_ratio | {_format_float(fast_ratio * 100)}% |")
    report.append(f"| agent_ratio | {_format_float(agent_ratio * 100)}% |")
    report.append("")

    report.append("## Prompt size")
    report.append("")
    report.append("| metric | value |")
    report.append("|---|---:|")
    report.append(f"| avg_prompt_chars | {_format_float(_avg(prompt_chars_vals))} |")
    report.append(f"| p50_prompt_chars | {_format_float(_percentile(prompt_chars_vals, 50))} |")
    report.append(f"| p95_prompt_chars | {_format_float(_percentile(prompt_chars_vals, 95))} |")
    report.append(f"| max_prompt_chars | {_format_int(max(prompt_chars_vals) if prompt_chars_vals else None)} |")
    report.append("")

    report.append("## Context / retrieval / tools / iterations")
    report.append("")
    report.append("| metric | value |")
    report.append("|---|---:|")
    report.append(f"| context_loaded_true | {context_flags.count(True)} |")
    report.append(f"| context_loaded_false | {context_flags.count(False)} |")
    report.append(f"| dir_rag_true | {dir_rag_flags.count(True)} |")
    report.append(f"| chroma_true | {chroma_flags.count(True)} |")
    report.append(f"| tool_calls_sum | {sum(tool_calls_vals) if tool_calls_vals else 0} |")
    report.append(f"| tool_calls_avg | {_format_float(_avg([float(v) for v in tool_calls_vals]))} |")
    report.append(
        f"| tool_calls_entries_pct | {_format_float((sum(1 for v in tool_calls_vals if v > 0) / total_entries) * 100) if total_entries else 'n/a'} |"
    )
    report.append(f"| iterations_avg | {_format_float(_avg([float(v) for v in iterations_vals]))} |")
    report.append(f"| iterations_max | {_format_int(max(iterations_vals) if iterations_vals else None)} |")
    report.append(
        f"| agent_loop_true_pct | {_format_float((agent_loop_flags.count(True) / total_entries) * 100) if total_entries else 'n/a'} |"
    )
    report.append("")

    report.append("## Timing (ms)")
    report.append("")
    report.append("| metric | p50 | p95 |")
    report.append("|---|---:|---:|")
    report.append(
        f"| latency_ms | {_format_float(_percentile(latency_ms_vals, 50))} | {_format_float(_percentile(latency_ms_vals, 95))} |"
    )
    report.append(
        f"| duration_ms | {_format_float(_percentile(duration_ms_vals, 50))} | {_format_float(_percentile(duration_ms_vals, 95))} |"
    )
    report.append(
        f"| elapsed_ms | {_format_float(_percentile(elapsed_ms_vals, 50))} | {_format_float(_percentile(elapsed_ms_vals, 95))} |"
    )
    report.append("")

    report.append("## Top request shapes")
    report.append("")
    report.append("### Top 10 prefixes (first 24 chars)")
    report.append("")
    report.append("| prefix | count |")
    report.append("|---|---:|")
    for prefix, cnt in request_prefixes.most_common(10):
        report.append(f"| `{prefix}` | {cnt} |")
    if not request_prefixes:
        report.append("| n/a | n/a |")
    report.append("")

    report.append("### Top 10 first words")
    report.append("")
    report.append("| word | count |")
    report.append("|---|---:|")
    for word, cnt in first_words.most_common(10):
        report.append(f"| `{word}` | {cnt} |")
    if not first_words:
        report.append("| n/a | n/a |")
    report.append("")

    report.append("### Top 10 longest requests")
    report.append("")
    report.append("| length | request |")
    report.append("|---:|---|")
    for ln, txt in longest_requests:
        report.append(f"| {ln} | `{txt}` |")
    if not longest_requests:
        report.append("| n/a | n/a |")
    report.append("")

    report.append("## Data health")
    report.append("")
    report.append("| item | value |")
    report.append("|---|---:|")
    report.append(f"| missing_file | {health.missing_file} |")
    report.append(f"| empty_file | {health.empty_file} |")
    report.append(f"| parse_errors | {health.parse_errors} |")
    report.append(f"| coverage.prompt_chars | {_coverage(len(prompt_chars_vals), total_entries)} |")
    report.append(f"| coverage.context_loaded | {_coverage(len(context_flags), total_entries)} |")
    report.append(f"| coverage.dir_rag | {_coverage(len(dir_rag_flags), total_entries)} |")
    report.append(f"| coverage.chroma | {_coverage(len(chroma_flags), total_entries)} |")
    report.append(f"| coverage.tool_calls | {_coverage(len(tool_calls_vals), total_entries)} |")
    report.append(f"| coverage.iterations | {_coverage(len(iterations_vals), total_entries)} |")
    report.append(f"| coverage.agent_loop | {_coverage(len(agent_loop_flags), total_entries)} |")
    report.append(f"| coverage.latency_ms | {_coverage(len(latency_ms_vals), total_entries)} |")
    report.append(f"| coverage.duration_ms | {_coverage(len(duration_ms_vals), total_entries)} |")
    report.append(f"| coverage.elapsed_ms | {_coverage(len(elapsed_ms_vals), total_entries)} |")
    report.append("")

    report.append("## Violations summary")
    report.append("")
    if not violations.get("present"):
        report.append("No violations data found.")
    else:
        report.append("| metric | value |")
        report.append("|---|---:|")
        report.append(f"| total_violation_lines | {violations.get('total_lines', 0)} |")
        report.append(f"| parse_errors | {violations.get('parse_errors', 0)} |")
        report.append(f"| most_recent_file | `{violations.get('most_recent_file', 'none')}` |")
        report.append("")
        report.append("### Counts by violation_type")
        report.append("")
        report.append("| violation_type | count |")
        report.append("|---|---:|")
        counts: Counter = violations.get("counts_by_type", Counter())
        for vt, cnt in counts.most_common():
            report.append(f"| `{vt}` | {cnt} |")
        if not counts:
            report.append("| n/a | n/a |")
    report.append("")

    return "\n".join(report)


def main() -> int:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        logs_dir = repo_root / "logs"
        report_dir = repo_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        router_log = logs_dir / "router-decisions.jsonl"
        violations_dir = logs_dir / "violations"

        metrics, health = summarize_router_logs(router_log)
        violations = summarize_violations(violations_dir)

        ts = _utc_now()
        report_name = f"router-metrics-{ts.strftime('%Y%m%d-%H%M%S')}.md"
        report_path = report_dir / report_name

        content = render_report(metrics, health, violations, ts)
        report_path.write_text(content, encoding="utf-8")

        rel_path = report_path.relative_to(repo_root)
        print(rel_path.as_posix())
        return 0
    except Exception as e:
        msg = str(e).strip().replace("\n", " ")
        if not msg:
            msg = "unknown error"
        sys.stderr.write(msg + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast peek/index/prioritizer utilities for Snake Agent"""
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

SAMPLE_HEAD_BYTES = 8192


def peek_summary(file_path: Path, head_bytes: int = SAMPLE_HEAD_BYTES) -> Dict[str, Any]:
    """Return a compact summary of a file by reading only the head portion.

    Summary includes: size, mtime, imports, top-level defs, TODOs, docstring (first triple-quote), and simple keyword flags.
    """
    summary: Dict[str, Any] = {
        "path": str(file_path),
        "size": 0,
        "mtime": 0.0,
        "imports": [],
        "defs": [],
        "todos": [],
        "docstring": None,
        "flags": [],
    }

    try:
        summary["size"] = file_path.stat().st_size
        summary["mtime"] = file_path.stat().st_mtime

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            head = f.read(head_bytes)

        # Extract imports
        imports = re.findall(
            r"^\s*(?:from\s+(\S+)\s+import|import\s+([\w\.]+))", head, re.M)
        flat_imports = set()
        for a, b in imports:
            if a:
                flat_imports.add(a.split('.')[0])
            if b:
                flat_imports.add(b.split('.')[0])
        summary["imports"] = sorted(flat_imports)

        # Extract top-level defs (function/class signatures)
        defs = re.findall(
            r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", head, re.M)
        summary["defs"] = [f"{kind} {name}" for kind, name in defs]

        # TODOs and FIXMEs
        todos = re.findall(r"(?i)\b(TODO|FIXME)[:\s-]*(.*)$", head, re.M)
        summary["todos"] = [t[1].strip() for t in todos]

        # Docstring (first triple-quoted block)
        ds = re.search(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', head, re.S)
        if ds:
            summary["docstring"] = (ds.group(1) or ds.group(
                2) or '').strip().splitlines()[0][:200]

        # Keyword flags
        flags = []
        if re.search(r"\beval\s*\(|\bexec\s*\(|subprocess\.|socket\.|requests\.", head):
            flags.append("dangerous_ops")
        if len(summary["defs"]) == 0 and summary["size"] > 3000:
            flags.append("large_no_defs")
        if summary["todos"]:
            flags.append("has_todos")
        summary["flags"] = flags

    except Exception:
        # Best-effort summary; return whatever we managed
        pass

    return summary


def simple_score(summary: Dict[str, Any], repo_root: Path = None) -> Tuple[float, Dict[str, float]]:
    """Compute a simple explainable score from a peek summary.

    Returns (score, breakdown)
    """
    breakdown: Dict[str, float] = {}
    score = 0.0

    # Weight signals
    size_kb = summary.get("size", 0) / 1024.0
    breakdown["size_penalty"] = 1.0 / \
        (1.0 + (size_kb / 50.0))  # large files penalized

    # Recent activity boost (mtime within 24 hours)
    now = time.time()
    mtime = summary.get("mtime", 0.0)
    recent = 1.0 if (now - mtime) < (24 * 3600) else 0.0
    breakdown["recent_change"] = recent * 3.0

    # imports importance (core libs less important)
    imp_count = len(summary.get("imports", []))
    breakdown["imports"] = min(imp_count, 5) * 0.3

    # TODOs
    todo_count = len(summary.get("todos", []))
    breakdown["todos"] = min(todo_count, 3) * 1.0

    # flags
    flags = summary.get("flags", [])
    breakdown["dangerous"] = 2.5 if "dangerous_ops" in flags else 0.0
    breakdown["large_no_defs"] = 1.0 if "large_no_defs" in flags else 0.0

    # defs: more defs means this file has many entry points; raise attention
    defs_count = len(summary.get("defs", []))
    breakdown["defs"] = min(defs_count, 10) * 0.2

    # Combine
    raw = (breakdown["recent_change"] + breakdown["imports"] + breakdown["todos"] +
           breakdown["dangerous"] + breakdown["large_no_defs"] + breakdown["defs"])

    # Apply size penalty
    score = raw * breakdown["size_penalty"]

    return score, breakdown


def index_repo(root: Path, out_path: Path = None, max_files: int = 2000) -> List[Dict[str, Any]]:
    """Quick one-shot index: peek summaries for repository files (non-recursive heavy filtering).

    Writes to out_path JSON if provided, and returns list of summaries with scores.
    """
    summaries: List[Dict[str, Any]] = []
    root = Path(root)

    exts = {'.py', '.md', '.json'}
    for p in root.rglob('**/*'):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                summ = peek_summary(p)
                score, breakdown = simple_score(summ, root)
                summ["score"] = score
                summ["breakdown"] = breakdown
                summaries.append(summ)
                if len(summaries) >= max_files:
                    break
            except Exception:
                continue

    summaries.sort(key=lambda s: s.get("score", 0.0), reverse=True)

    if out_path:
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2)
        except Exception:
            pass

    return summaries

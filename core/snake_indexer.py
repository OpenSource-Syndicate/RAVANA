"""Persistent indexer for Snake peek/prioritizer

Creates a lightweight JSON index (.snake_index.json) containing peek summaries and git metadata.
"""
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.snake_peek import peek_summary, simple_score


def _git_commit_count(path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD", "--", str(path)], stderr=subprocess.DEVNULL)
        return int(out.decode().strip())
    except Exception:
        return None


def _git_last_commit_ts(path: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--", str(path)], stderr=subprocess.DEVNULL)
        return int(out.decode().strip())
    except Exception:
        return None


def build_index(root: Path, index_file: Path = None, exts: Optional[set] = None, max_files: int = 2000) -> List[Dict[str, Any]]:
    root = Path(root)
    if exts is None:
        exts = {'.py', '.md', '.json'}

    entries: List[Dict[str, Any]] = []
    scanned = 0
    for p in root.rglob('**/*'):
        if scanned >= max_files:
            break
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        try:
            summ = peek_summary(p)
            score, breakdown = simple_score(summ, root)
            summ['score'] = score
            summ['breakdown'] = breakdown

            # Git metadata (best-effort)
            commit_count = _git_commit_count(p)
            last_commit = _git_last_commit_ts(p)
            if commit_count is not None:
                summ['git_commit_count'] = commit_count
            if last_commit is not None:
                summ['git_last_commit_ts'] = last_commit

            entries.append(summ)
            scanned += 1
        except Exception:
            continue

    entries.sort(key=lambda s: s.get('score', 0.0), reverse=True)

    if index_file:
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({'generated_at': int(time.time()),
                          'entries': entries}, f, indent=2)
        except Exception:
            pass

    return entries


def load_index(index_file: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('entries', [])
    except Exception:
        return None

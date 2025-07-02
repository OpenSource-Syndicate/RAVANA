import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../memory-3')))
from llm import call_llm
from compression_prompts import COMPRESSION_PROMPT
from compressed_memory import save_summary, load_summaries

def compress_knowledge(logs):
    """Summarize accumulated knowledge/logs using the LLM."""
    prompt = COMPRESSION_PROMPT.format(logs=json.dumps(logs, indent=2))
    summary = call_llm(prompt)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    }
    save_summary(entry)
    return entry

def scheduled_compression(logs, schedule="weekly"):
    """
    Run knowledge compression on a schedule (weekly/monthly).
    This function can be called by a scheduler (e.g., cron, Windows Task Scheduler).
    Args:
        logs: List of log entries to summarize.
        schedule: 'weekly' or 'monthly'.
    Returns:
        The summary entry created.
    """
    # In a real deployment, this would be triggered by a scheduler.
    return compress_knowledge(logs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge Compression Module")
    parser.add_argument('--logs', type=str, required=True, help='Path to logs/reflections JSON file')
    args = parser.parse_args()
    with open(args.logs, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    entry = compress_knowledge(logs)
    print(json.dumps(entry, indent=2))

if __name__ == "__main__":
    main()
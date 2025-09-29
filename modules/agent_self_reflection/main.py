from reflection_db import save_reflection, load_reflections
from reflection_prompts import REFLECTION_PROMPT
from core.llm import call_llm, run_langchain_reflection
import os
import sys
import json
from datetime import datetime, timezone
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../episodic_memory')))

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False


def reflect_on_task(task_summary, outcome, blog_scheduler=None):
    """Generate a self-reflection using the LLM."""
    prompt = REFLECTION_PROMPT.format(
        task_summary=task_summary, outcome=outcome)
    reflection = call_llm(prompt)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_summary": task_summary,
        "outcome": outcome,
        "reflection": reflection
    }
    save_reflection(entry)

    # Check if this reflection should trigger a blog post
    if BLOG_SCHEDULER_AVAILABLE and blog_scheduler:
        _check_reflection_blog_trigger(entry, blog_scheduler)

    return entry


def _check_reflection_blog_trigger(entry, blog_scheduler):
    """Check if a reflection should trigger a blog post."""
    try:
        reflection_text = entry.get('reflection', '')
        task_summary = entry.get('task_summary', '')
        outcome = entry.get('outcome', '')

        # Determine if the reflection contains significant insights
        insight_keywords = ['learned', 'discovered', 'realized',
                            'understood', 'insight', 'breakthrough', 'pattern', 'connection']
        insight_score = sum(
            1 for keyword in insight_keywords if keyword.lower() in reflection_text.lower())

        # Check for emotional depth
        emotional_keywords = ['feel', 'felt', 'emotional',
                              'frustrated', 'excited', 'proud', 'disappointed', 'surprised']
        emotional_score = sum(
            1 for keyword in emotional_keywords if keyword.lower() in reflection_text.lower())

        # Calculate importance
        importance_score = 0.3  # Base score
        # Up to 0.3 for insights
        importance_score += min(0.3, insight_score * 0.1)
        # Up to 0.2 for emotional depth
        importance_score += min(0.2, emotional_score * 0.05)
        # Up to 0.2 for detailed reflections
        importance_score += min(0.2, len(reflection_text) / 1000)

        # Only blog if significant enough
        if importance_score < 0.6:
            return

        # Determine emotional valence
        positive_words = ['success', 'good', 'well',
                          'effective', 'learned', 'growth', 'improvement']
        negative_words = ['failed', 'error', 'mistake',
                          'difficult', 'challenge', 'struggle']

        positive_count = sum(
            1 for word in positive_words if word in reflection_text.lower())
        negative_count = sum(
            1 for word in negative_words if word in reflection_text.lower())

        if positive_count > negative_count:
            emotional_valence = 0.4
        elif negative_count > positive_count:
            emotional_valence = -0.2  # Negative but still valuable for learning
        else:
            emotional_valence = 0.1

        reasoning_why = f"""This reflection on '{task_summary}' revealed important insights about my 
performance and decision-making process. Self-reflection is essential for continuous improvement 
and helps me understand patterns in my behavior and outcomes."""

        reasoning_how = f"""Through systematic reflection using structured prompts, I analyzed the task, 
its outcome, and the underlying factors that contributed to the result. This metacognitive process 
helps me extract actionable insights from experiences."""

        # Extract key phrases for tags
        task_words = [word.lower()
                      for word in task_summary.split() if len(word) > 3]
        tags = ['reflection', 'self-analysis',
                'learning', 'metacognition'] + task_words[:4]

        import asyncio
        asyncio.create_task(blog_scheduler.register_learning_event(
            trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
            topic=f"Task Reflection: {task_summary[:50]}{'...' if len(task_summary) > 50 else ''}",
            context=f"Self-reflection on task outcome: {outcome[:100]}{'...' if len(outcome) > 100 else ''}",
            learning_content=f"Task: {task_summary}\nOutcome: {outcome}\nReflection: {reflection_text[:500]}{'...' if len(reflection_text) > 500 else ''}",
            reasoning_why=reasoning_why,
            reasoning_how=reasoning_how,
            emotional_valence=emotional_valence,
            importance_score=importance_score,
            tags=tags[:10],
            metadata={
                'reflection_type': 'task_reflection',
                'insight_score': insight_score,
                'emotional_score': emotional_score,
                'reflection_length': len(reflection_text)
            }
        ))

        print(
            f"Registered reflection blog trigger with importance {importance_score:.2f}")

    except Exception as e:
        print(f"Failed to register reflection blog trigger: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Self-Reflection & Self-Modification Module")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Reflection command
    reflect_parser = subparsers.add_parser(
        'reflect', help='Generate a reflection')
    reflect_parser.add_argument(
        '--task', type=str, required=True, help='Task summary')
    reflect_parser.add_argument(
        '--outcome', type=str, required=True, help='Outcome description')
    reflect_parser.add_argument('--use-langchain', action='store_true',
                                help='Use LangChain for Planning → Execution → Reflection')

    # Self-modification command
    modify_parser = subparsers.add_parser(
        'modify', help='Run self-modification on reflection logs')

    args = parser.parse_args()

    if args.command == 'reflect':
        if args.use_langchain:
            entry = run_langchain_reflection(args.task, args.outcome)
        else:
            entry = reflect_on_task(args.task, args.outcome)
        print(json.dumps(entry, indent=2))
    elif args.command == 'modify':
        from self_modification import run_self_modification
        run_self_modification()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

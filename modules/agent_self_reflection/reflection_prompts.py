REFLECTION_PROMPT = (
    "You are an AI agent journaling after a major task. "
    "Given the following summary and outcome, answer these questions in a structured way:\n"
    "1. What worked?\n2. What failed?\n3. What surprised you?\n4. What do you still need to learn?\n"
    "Task Summary: {task_summary}\nOutcome: {outcome}\n"
    "Respond in a clear, numbered format."
) 
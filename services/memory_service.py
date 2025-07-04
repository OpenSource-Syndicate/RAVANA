from modules.episodic_memory.memory import (
    extract_memories_api,
    save_memories,
    get_relevant_memories_api,
    consolidate_memories_api,
)
import asyncio

class MemoryService:
    async def get_relevant_memories(self, query_text: str):
        return await get_relevant_memories_api({"query_text": query_text})

    async def save_memories(self, memories):
        await asyncio.to_thread(save_memories, memories)

    async def extract_memories(self, user_input: str, ai_output: str):
        return await extract_memories_api({"user_input": user_input, "ai_output": ai_output})

    async def consolidate_memories(self):
        from modules.episodic_memory.memory import ConsolidateRequest
        return await consolidate_memories_api(ConsolidateRequest()) 
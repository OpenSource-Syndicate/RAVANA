import asyncio
import logging
from typing import Any, Dict

from core.config import Config
from modules.decision_engine.llm import generate_hypothetical_scenarios
from modules.information_processing.trend_analysis.trend_engine import analyze_trends
from modules.information_processing.youtube_transcription.youtube_transcription import transcribe_youtube_video

logger = logging.getLogger(__name__)

class ActionManager:
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.actions = {
            "fetch_and_analyze_trends": self._action_fetch_and_analyze_trends,
            "transcribe_youtube": self._action_transcribe_youtube,
            "run_hypothetical_scenario": self._action_run_hypothetical_scenario,
            "compress_knowledge": self._action_compress_knowledge,
        }

    async def execute_action(self, decision: Dict[str, Any]) -> Any:
        """Executes the action determined by the decision engine."""
        action_type = decision.get("action")
        params = decision.get("params", {})

        action_func = self.actions.get(action_type)
        if action_func:
            return await action_func(params)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return "No action taken."

    async def _action_fetch_and_analyze_trends(self, params: Dict[str, Any]) -> Any:
        return await asyncio.to_thread(analyze_trends, Config.FEED_URLS)

    async def _action_transcribe_youtube(self, params: Dict[str, Any]) -> Any:
        return await asyncio.to_thread(transcribe_youtube_video, params.get("url"))

    async def _action_run_hypothetical_scenario(self, params: Dict[str, Any]) -> Any:
        return await generate_hypothetical_scenarios(params.get("scenario_description"))

    async def _action_compress_knowledge(self, params: Dict[str, Any]) -> Any:
        return await asyncio.to_thread(self.agi_system.knowledge_service.compress_and_save_knowledge) 
from core.actions.action import Action
from typing import Any, Dict, List


class ProcessImageAction(Action):
    @property
    def name(self) -> str:
        return "process_image"

    @property
    def description(self) -> str:
        return "Process and analyze an image file."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "image_path", "type": "str",
                "description": "Path to the image file.", "required": True},
            {"name": "analysis_prompt", "type": "str",
                "description": "Analysis prompt (optional).", "required": False}
        ]

    async def execute(self, image_path: str, analysis_prompt: str = None) -> Any:
        return await self.system.action_manager.process_image_action(image_path, analysis_prompt)


class ProcessAudioAction(Action):
    @property
    def name(self) -> str:
        return "process_audio"

    @property
    def description(self) -> str:
        return "Process and analyze an audio file."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "audio_path", "type": "str",
                "description": "Path to the audio file.", "required": True},
            {"name": "analysis_prompt", "type": "str",
                "description": "Analysis prompt (optional).", "required": False}
        ]

    async def execute(self, audio_path: str, analysis_prompt: str = None) -> Any:
        return await self.system.action_manager.process_audio_action(audio_path, analysis_prompt)


class AnalyzeDirectoryAction(Action):
    @property
    def name(self) -> str:
        return "analyze_directory"

    @property
    def description(self) -> str:
        return "Analyze all media files in a directory."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "directory_path", "type": "str",
                "description": "Path to the directory.", "required": True},
            {"name": "recursive", "type": "bool",
                "description": "Whether to search recursively (optional).", "required": False}
        ]

    async def execute(self, directory_path: str, recursive: bool = False) -> Any:
        return await self.system.action_manager.analyze_directory_action(directory_path, recursive)


class CrossModalAnalysisAction(Action):
    @property
    def name(self) -> str:
        return "cross_modal_analysis"

    @property
    def description(self) -> str:
        return "Perform cross-modal analysis on multiple content types."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "content_paths", "type": "list",
                "description": "List of file paths.", "required": True},
            {"name": "analysis_prompt", "type": "str",
                "description": "Analysis prompt (optional).", "required": False}
        ]

    async def execute(self, content_paths: list, analysis_prompt: str = None) -> Any:
        return await self.system.action_manager.cross_modal_analysis_action(content_paths, analysis_prompt)

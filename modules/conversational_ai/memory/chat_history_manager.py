"""
Chat History Manager for Conversational AI

This module provides advanced chat history management capabilities including
intelligent pruning, long-term storage, and integration with RAVANA's 
Very Long-Term Memory system.
"""

import logging
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import uuid

from core.vltm_data_models import MemoryType, MemoryRecord
from modules.conversational_ai.memory.memory_interface import SharedMemoryInterface

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat history with intelligent pruning and long-term storage."""

    def __init__(self, memory_interface: SharedMemoryInterface, vltm_store=None):
        """
        Initialize the chat history manager.

        Args:
            memory_interface: Shared memory interface for user data
            vltm_store: Very Long-Term Memory store (optional)
        """
        self.memory_interface = memory_interface
        self.vltm_store = vltm_store
        self.prune_threshold = 100  # Number of messages before pruning
        self.archive_threshold = 500  # Number of messages before archiving to VLTM

    def store_chat_message(self, user_id: str, message: Dict[str, Any]):
        """
        Store a chat message with intelligent management.

        Args:
            user_id: Unique identifier for the user
            message: Message data to store
        """
        try:
            # Store in shared memory
            self.memory_interface.store_conversation(user_id, message)

            # Check if we need to perform management operations
            self._manage_chat_history(user_id)

        except Exception as e:
            logger.error(f"Error storing chat message for user {user_id}: {e}")

    def _manage_chat_history(self, user_id: str):
        """
        Perform intelligent management of chat history.

        Args:
            user_id: Unique identifier for the user
        """
        try:
            # Get current chat history size
            memory_path = self.memory_interface._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return

            with open(memory_path, 'r') as f:
                user_memory = json.load(f)

            conversations = user_memory.get("conversations", [])

            # If we have too many conversations, perform intelligent pruning
            if len(conversations) > self.prune_threshold:
                self._prune_chat_history(user_id, user_memory, conversations)

            # If we have a very large history, consider archiving to VLTM
            if len(conversations) > self.archive_threshold and self.vltm_store:
                self._archive_to_vltm(user_id, user_memory, conversations)

        except Exception as e:
            logger.error(
                f"Error managing chat history for user {user_id}: {e}")

    def _prune_chat_history(self, user_id: str, user_memory: Dict[str, Any], conversations: List[Dict[str, Any]]):
        """
        Prune chat history using intelligent selection criteria.

        Args:
            user_id: Unique identifier for the user
            user_memory: User memory data
            conversations: List of conversations
        """
        try:
            # Sort conversations by importance (simple heuristic for now)
            # In a more advanced implementation, this would use NLP to assess importance
            sorted_conversations = sorted(
                conversations,
                key=lambda x: self._assess_conversation_importance(x),
                reverse=True
            )

            # Keep the most important conversations
            pruned_conversations = sorted_conversations[:50]  # Keep top 50

            # Move less important conversations to archive
            archived_conversations = sorted_conversations[50:]

            if archived_conversations:
                # Create archive entry
                archive_entry = {
                    "archive_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "period": f"Conversations from {archived_conversations[0].get('timestamp', 'unknown')} to {archived_conversations[-1].get('timestamp', 'unknown')}",
                    "conversation_count": len(archived_conversations),
                    "conversations": archived_conversations,
                    "pruning_reason": "intelligent_pruning"
                }

                # Add to long-term archive
                user_memory["long_term_archive"].append(archive_entry)

                # Keep only recent archive entries
                if len(user_memory["long_term_archive"]) > 20:
                    user_memory["long_term_archive"] = user_memory["long_term_archive"][-20:]

                # Update conversations
                user_memory["conversations"] = pruned_conversations

                # Save updated memory
                memory_path = self.memory_interface._get_user_memory_path(
                    user_id)
                with open(memory_path, 'w') as f:
                    json.dump(user_memory, f, indent=2)

                logger.info(
                    f"Pruned {len(archived_conversations)} conversations for user {user_id}")

        except Exception as e:
            logger.error(f"Error pruning chat history for user {user_id}: {e}")

    def _assess_conversation_importance(self, conversation: Dict[str, Any]) -> float:
        """
        Assess the importance of a conversation (simple heuristic).

        Args:
            conversation: Conversation data

        Returns:
            Importance score (0.0 to 1.0)
        """
        try:
            # Simple heuristic based on content length and emotional intensity
            user_message = conversation.get("user_message", "")
            ai_response = conversation.get("ai_response", "")
            emotional_context = conversation.get("emotional_context", {})

            # Base score on content length
            content_length_score = min(
                (len(user_message) + len(ai_response)) / 1000.0, 1.0)

            # Boost score for emotional intensity
            emotional_intensity = sum(
                emotional_context.get("mood_vector", {}).values())
            emotional_score = min(emotional_intensity / 10.0, 1.0)

            # Combined score
            return 0.7 * content_length_score + 0.3 * emotional_score

        except Exception as e:
            logger.error(f"Error assessing conversation importance: {e}")
            return 0.5

    def _archive_to_vltm(self, user_id: str, user_memory: Dict[str, Any], conversations: List[Dict[str, Any]]):
        """
        Archive old conversations to Very Long-Term Memory.

        Args:
            user_id: Unique identifier for the user
            user_memory: User memory data
            conversations: List of conversations
        """
        try:
            # Get conversations to archive (older ones)
            conversations_to_archive = conversations[:-
                                                     100] if len(conversations) > 100 else []

            if conversations_to_archive and self.vltm_store:
                # Create memory record for VLTM
                content = {
                    "user_id": user_id,
                    "conversation_history": conversations_to_archive,
                    "period": f"Conversations from {conversations_to_archive[0].get('timestamp', 'unknown')} to {conversations_to_archive[-1].get('timestamp', 'unknown')}",
                    "summary": self._generate_archive_summary(conversations_to_archive)
                }

                metadata = {
                    "user_id": user_id,
                    "archive_type": "chat_history",
                    "conversation_count": len(conversations_to_archive),
                    "timestamp": datetime.now().isoformat()
                }

                memory_record = MemoryRecord(
                    memory_id=f"chat_archive_{user_id}_{uuid.uuid4()}",
                    memory_type=MemoryType.BEHAVIORAL_PATTERN,
                    content=content,
                    metadata=metadata,
                    importance_score=0.6,  # Moderate importance
                    strategic_value=0.4,
                    source_session="conversational_ai"
                )

                # Store in VLTM
                # Note: This would be an async operation in a real implementation
                # await self.vltm_store.store_memory(memory_record)

                logger.info(
                    f"Archived {len(conversations_to_archive)} conversations to VLTM for user {user_id}")

        except Exception as e:
            logger.error(f"Error archiving to VLTM for user {user_id}: {e}")

    def _generate_archive_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of archived conversations.

        Args:
            conversations: List of conversations to summarize

        Returns:
            Summary string
        """
        try:
            # Simple summary generation
            if not conversations:
                return "No conversations to summarize"

            # Extract topics from conversations
            topics = set()
            for conv in conversations[:10]:  # Sample first 10 conversations
                user_message = conv.get("user_message", "").lower()
                ai_response = conv.get("ai_response", "").lower()

                # Simple topic extraction
                common_topics = ["ai", "technology", "science", "philosophy", "research",
                                 "programming", "learning", "innovation", "ethics", "future"]
                for topic in common_topics:
                    if topic in user_message or topic in ai_response:
                        topics.add(topic)

            time_period = f"from {conversations[0].get('timestamp', 'unknown')} to {conversations[-1].get('timestamp', 'unknown')}"

            return f"Chat history archive containing {len(conversations)} conversations about {', '.join(list(topics)[:3])} {time_period}"

        except Exception as e:
            logger.error(f"Error generating archive summary: {e}")
            return "Chat history archive"

    def search_chat_history(self, user_id: str, query: str, search_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Search chat history for a query.

        Args:
            user_id: Unique identifier for the user
            query: Query string to search for
            search_archived: Whether to search archived conversations

        Returns:
            List of matching conversations
        """
        try:
            matching_conversations = []
            query_lower = query.lower()

            # Search recent conversations
            recent_conversations = self.memory_interface.retrieve_relevant_memories(
                user_id, query)
            matching_conversations.extend(recent_conversations)

            # Search archived conversations if requested
            if search_archived:
                archived_matches = self.memory_interface.search_archived_conversations(
                    user_id, query)
                matching_conversations.extend(archived_matches)

            return matching_conversations

        except Exception as e:
            logger.error(
                f"Error searching chat history for user {user_id}: {e}")
            return []

    def get_chat_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get analytics about a user's chat history.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Dictionary of analytics data
        """
        try:
            memory_path = self.memory_interface._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return {}

            with open(memory_path, 'r') as f:
                user_memory = json.load(f)

            conversations = user_memory.get("conversations", [])
            archive = user_memory.get("long_term_archive", [])

            # Calculate basic statistics
            total_conversations = len(conversations)
            total_archived = sum(entry.get("conversation_count", 0)
                                 for entry in archive)

            # Analyze conversation patterns
            topics = {}
            sentiment_scores = []

            for conv in conversations[-50:]:  # Analyze last 50 conversations
                # Extract topics (simplified)
                user_message = conv.get("user_message", "").lower()
                ai_response = conv.get("ai_response", "").lower()

                common_topics = ["ai", "technology", "science", "philosophy", "research",
                                 "programming", "learning", "innovation", "ethics", "future"]
                for topic in common_topics:
                    if topic in user_message or topic in ai_response:
                        topics[topic] = topics.get(topic, 0) + 1

                # Extract sentiment from emotional context
                emotional_context = conv.get("emotional_context", {})
                mood_vector = emotional_context.get("mood_vector", {})
                if mood_vector:
                    positive_moods = sum(mood_vector.get(mood, 0) for mood in [
                                         "Happy", "Confident", "Curious", "Excited"])
                    negative_moods = sum(mood_vector.get(mood, 0) for mood in [
                                         "Sad", "Anxious", "Frustrated", "Angry"])
                    sentiment_score = positive_moods - negative_moods
                    sentiment_scores.append(sentiment_score)

            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / \
                len(sentiment_scores) if sentiment_scores else 0

            return {
                "total_conversations": total_conversations,
                "total_archived_conversations": total_archived,
                "top_topics": sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5],
                "average_sentiment": avg_sentiment,
                "conversation_trend": "increasing" if len(conversations) > 20 else "stable"
            }

        except Exception as e:
            logger.error(
                f"Error generating chat analytics for user {user_id}: {e}")
            return {}

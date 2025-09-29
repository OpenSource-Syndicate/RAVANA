import logging
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SharedMemoryInterface:
    def __init__(self, memory_path: str = "shared_memory"):
        """
        Initialize the shared memory interface.

        Args:
            memory_path: Path to the shared memory storage
        """
        self.memory_path = memory_path
        self._ensure_memory_directory()

    def _ensure_memory_directory(self):
        """Ensure the memory directory exists."""
        if not os.path.exists(self.memory_path):
            os.makedirs(self.memory_path)
            logger.info(f"Created memory directory: {self.memory_path}")

    def _get_user_memory_path(self, user_id: str) -> str:
        """Get the file path for a user's memory data."""
        return os.path.join(self.memory_path, f"user_{user_id}_memory.json")

    def _get_user_archive_path(self, user_id: str) -> str:
        """Get the file path for a user's archived memory data."""
        return os.path.join(self.memory_path, f"user_{user_id}_archive.json")

    def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get context for a user from shared memory.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Dictionary containing user context
        """
        try:
            # In a real implementation, this would query the actual RAVANA memory system
            # For now, we'll return a basic context structure

            # Try to load user-specific memory if it exists
            memory_path = self._get_user_memory_path(user_id)
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, 'r') as f:
                        user_memory = json.load(f)
                    return user_memory.get("context", {})
                except Exception as e:
                    logger.error(
                        f"Error loading user memory for {user_id}: {e}")

            # Return default context
            return {
                "user_id": user_id,
                "recent_interactions": [],
                "user_interests": [],
                "shared_knowledge": [],
                "emotional_context": {}
            }
        except Exception as e:
            logger.error(f"Error getting context for user {user_id}: {e}")
            return {}

    def store_conversation(self, user_id: str, conversation: Dict[str, Any]):
        """
        Store a conversation in shared memory.

        Args:
            user_id: Unique identifier for the user
            conversation: Conversation data to store
        """
        try:
            # In a real implementation, this would store in the actual RAVANA memory system
            # For now, we'll store in a user-specific file

            memory_path = self._get_user_memory_path(user_id)

            # Load existing memory or create new structure
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, 'r') as f:
                        user_memory = json.load(f)
                except Exception as e:
                    logger.error(
                        f"Error loading existing memory for {user_id}: {e}")
                    user_memory = self._create_user_memory_structure(user_id)
            else:
                user_memory = self._create_user_memory_structure(user_id)

            # Add conversation to memory
            user_memory["conversations"].append(conversation)

            # Implement intelligent pruning - move old conversations to archive when threshold is reached
            # Higher threshold for shared memory
            if len(user_memory["conversations"]) > 100:
                self._archive_old_conversations(user_id, user_memory)

            # Keep only recent conversations (last 50)
            if len(user_memory["conversations"]) > 50:
                user_memory["conversations"] = user_memory["conversations"][-50:]

            # Update context with recent information
            self._update_context_from_conversation(user_memory, conversation)

            # Save updated memory
            with open(memory_path, 'w') as f:
                json.dump(user_memory, f, indent=2)

            logger.debug(f"Stored conversation for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing conversation for user {user_id}: {e}")

    def _create_user_memory_structure(self, user_id: str) -> Dict[str, Any]:
        """Create the initial memory structure for a user."""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "context": {
                "user_id": user_id,
                "recent_interactions": [],
                "user_interests": [],
                "shared_knowledge": [],
                "emotional_context": {}
            },
            "conversations": [],
            "knowledge_fragments": [],
            "emotional_history": [],
            "chat_summaries": [],  # Add chat summaries storage
            "long_term_archive": []  # Add long-term archive storage
        }

    def _update_context_from_conversation(self, user_memory: Dict[str, Any], conversation: Dict[str, Any]):
        """Update the user context based on a new conversation."""
        context = user_memory["context"]

        # Add to recent interactions
        context["recent_interactions"].append({
            "timestamp": conversation.get("timestamp", datetime.now().isoformat()),
            "user_message": conversation.get("user_message", "")[:100] + "..." if len(conversation.get("user_message", "")) > 100 else conversation.get("user_message", ""),
            "ai_response": conversation.get("ai_response", "")[:100] + "..." if len(conversation.get("ai_response", "")) > 100 else conversation.get("ai_response", "")
        })

        # Keep only recent interactions (last 10)
        if len(context["recent_interactions"]) > 10:
            context["recent_interactions"] = context["recent_interactions"][-10:]

        # Update emotional context
        emotional_context = conversation.get("emotional_context", {})
        if emotional_context:
            context["emotional_context"] = emotional_context
            # Add to emotional history
            user_memory["emotional_history"].append({
                "timestamp": conversation.get("timestamp", datetime.now().isoformat()),
                "emotional_state": emotional_context
            })

            # Keep only recent emotional history (last 20)
            if len(user_memory["emotional_history"]) > 20:
                user_memory["emotional_history"] = user_memory["emotional_history"][-20:]

        # Update last updated timestamp
        user_memory["last_updated"] = datetime.now().isoformat()

    def _archive_old_conversations(self, user_id: str, user_memory: Dict[str, Any]):
        """
        Archive old conversations to long-term storage.

        Args:
            user_id: Unique identifier for the user
            user_memory: User memory dictionary
        """
        try:
            # Get conversations to archive (older than the last 50)
            conversations_to_archive = user_memory["conversations"][:-50] if len(
                user_memory["conversations"]) > 50 else []

            if conversations_to_archive:
                # Create archive entry
                archive_entry = {
                    "archive_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "period": f"Conversations from {conversations_to_archive[0].get('timestamp', 'unknown')} to {conversations_to_archive[-1].get('timestamp', 'unknown')}",
                    "conversation_count": len(conversations_to_archive),
                    "conversations": conversations_to_archive
                }

                # Add to long-term archive
                user_memory["long_term_archive"].append(archive_entry)

                # Keep only recent archive entries (last 20)
                if len(user_memory["long_term_archive"]) > 20:
                    user_memory["long_term_archive"] = user_memory["long_term_archive"][-20:]

                logger.info(
                    f"Archived {len(conversations_to_archive)} conversations for user {user_id}")

        except Exception as e:
            logger.error(
                f"Error archiving conversations for user {user_id}: {e}")

    def retrieve_relevant_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a user based on a query.

        Args:
            user_id: Unique identifier for the user
            query: Query to search for relevant memories
            top_k: Number of memories to retrieve

        Returns:
            List of relevant memories
        """
        try:
            # In a real implementation, this would query the actual RAVANA memory system
            # using vector similarity search or other retrieval methods

            memory_path = self._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return []

            try:
                with open(memory_path, 'r') as f:
                    user_memory = json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory for {user_id}: {e}")
                return []

            # Simple keyword matching for demonstration
            # In a real implementation, this would use embeddings and similarity search
            relevant_memories = []
            conversations = user_memory.get("conversations", [])

            query_lower = query.lower()
            for conv in conversations:
                user_message = conv.get("user_message", "").lower()
                ai_response = conv.get("ai_response", "").lower()

                # Simple relevance scoring based on keyword matching
                score = 0
                for word in query_lower.split():
                    if word in user_message or word in ai_response:
                        score += 1

                if score > 0:
                    relevant_memories.append({
                        "conversation": conv,
                        "relevance_score": score
                    })

            # Sort by relevance and return top_k
            relevant_memories.sort(
                key=lambda x: x["relevance_score"], reverse=True)
            return [mem["conversation"] for mem in relevant_memories[:top_k]]

        except Exception as e:
            logger.error(
                f"Error retrieving relevant memories for user {user_id}: {e}")
            return []

    def store_knowledge_fragment(self, user_id: str, fragment: Dict[str, Any]):
        """
        Store a knowledge fragment in the user's memory.

        Args:
            user_id: Unique identifier for the user
            fragment: Knowledge fragment to store
        """
        try:
            memory_path = self._get_user_memory_path(user_id)

            # Load existing memory or create new structure
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, 'r') as f:
                        user_memory = json.load(f)
                except Exception as e:
                    logger.error(
                        f"Error loading existing memory for {user_id}: {e}")
                    user_memory = self._create_user_memory_structure(user_id)
            else:
                user_memory = self._create_user_memory_structure(user_id)

            # Add knowledge fragment
            user_memory["knowledge_fragments"].append(fragment)

            # Keep only recent fragments (last 100)
            if len(user_memory["knowledge_fragments"]) > 100:
                user_memory["knowledge_fragments"] = user_memory["knowledge_fragments"][-100:]

            # Save updated memory
            with open(memory_path, 'w') as f:
                json.dump(user_memory, f, indent=2)

            logger.debug(f"Stored knowledge fragment for user {user_id}")
        except Exception as e:
            logger.error(
                f"Error storing knowledge fragment for user {user_id}: {e}")

    def get_user_knowledge(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge fragments for a user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            List of knowledge fragments
        """
        try:
            memory_path = self._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return []

            try:
                with open(memory_path, 'r') as f:
                    user_memory = json.load(f)
                return user_memory.get("knowledge_fragments", [])
            except Exception as e:
                logger.error(f"Error loading knowledge for {user_id}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting user knowledge for {user_id}: {e}")
            return []

    def summarize_chat_history(self, user_id: str, max_messages: int = 50) -> Optional[Dict[str, Any]]:
        """
        Summarize chat history using LLM-based approach when it becomes large.

        Args:
            user_id: Unique identifier for the user
            max_messages: Maximum number of messages to summarize

        Returns:
            Summary dictionary or None if summarization failed
        """
        try:
            # Import LLM module for summarization
            from core.llm import safe_call_llm

            # Load user memory
            memory_path = self._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return None

            try:
                with open(memory_path, 'r') as f:
                    user_memory = json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory for {user_id}: {e}")
                return None

            # Get conversations to summarize
            conversations = user_memory.get("conversations", [])
            if len(conversations) < max_messages:
                return None  # Not enough messages to summarize

            # Get the messages to summarize
            messages_to_summarize = conversations[-max_messages:]

            # Create a prompt for summarization
            summary_prompt = f"""
You are an AI assistant tasked with summarizing a conversation history. 
Please provide a concise summary of the following conversation that captures the key topics, 
themes, and any important context or decisions made.

Conversation History:
{json.dumps(messages_to_summarize, indent=2)}

Summary Instructions:
1. Identify the main topics discussed
2. Note any important decisions or agreements
3. Capture the overall tone and sentiment
4. Highlight any recurring themes or interests
5. Keep the summary concise but informative
6. Include any significant user interests or goals expressed

Provide only the summary, nothing else.
"""

            # Generate summary using LLM
            summary = safe_call_llm(summary_prompt, timeout=30, retries=3)

            if summary:
                # Create a summary entry
                summary_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "period": f"Last {max_messages} messages up to {datetime.now().isoformat()}",
                    "summary": summary,
                    "message_count": len(messages_to_summarize),
                    "topics": self._extract_topics_from_summary(summary)
                }

                # Add to chat summaries
                user_memory["chat_summaries"].append(summary_entry)

                # Keep only recent summaries (last 5)
                if len(user_memory["chat_summaries"]) > 5:
                    user_memory["chat_summaries"] = user_memory["chat_summaries"][-5:]

                # Save updated memory
                with open(memory_path, 'w') as f:
                    json.dump(user_memory, f, indent=2)

                logger.info(f"Created chat summary for user {user_id}")
                return summary_entry

            return None

        except Exception as e:
            logger.error(
                f"Error summarizing chat history for user {user_id}: {e}")
            return None

    def _extract_topics_from_summary(self, summary: str) -> List[str]:
        """
        Extract key topics from a conversation summary.

        Args:
            summary: Conversation summary text

        Returns:
            List of extracted topics
        """
        try:
            # Import LLM module for topic extraction
            from core.llm import safe_call_llm

            # Create a prompt for topic extraction
            topic_prompt = f"""
From the following conversation summary, extract 3-5 key topics or themes discussed.
Return only a JSON array of topic strings.

Summary:
{summary}

Example response format:
["AI research", "philosophy", "technology trends"]

Response:
"""

            # Generate topics using LLM
            topics_response = safe_call_llm(
                topic_prompt, timeout=20, retries=2)

            if topics_response:
                try:
                    topics = json.loads(topics_response)
                    if isinstance(topics, list):
                        return topics[:5]  # Limit to 5 topics
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract topics using simple keyword approach
                    pass

            # Fallback to simple keyword extraction
            common_topics = ["AI", "technology", "science", "philosophy", "research",
                             "programming", "learning", "innovation", "ethics", "future"]
            extracted_topics = []
            summary_lower = summary.lower()

            for topic in common_topics:
                if topic.lower() in summary_lower:
                    extracted_topics.append(topic)

            return extracted_topics[:5]  # Limit to 5 topics

        except Exception as e:
            logger.error(f"Error extracting topics from summary: {e}")
            return []

    def get_chat_summaries(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get chat summaries for a user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            List of chat summaries
        """
        try:
            memory_path = self._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return []

            try:
                with open(memory_path, 'r') as f:
                    user_memory = json.load(f)
                return user_memory.get("chat_summaries", [])
            except Exception as e:
                logger.error(
                    f"Error loading chat summaries for {user_id}: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting chat summaries for {user_id}: {e}")
            return []

    def search_archived_conversations(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Search archived conversations for a query.

        Args:
            user_id: Unique identifier for the user
            query: Query string to search for

        Returns:
            List of matching archived conversations
        """
        try:
            memory_path = self._get_user_memory_path(user_id)
            if not os.path.exists(memory_path):
                return []

            try:
                with open(memory_path, 'r') as f:
                    user_memory = json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory for {user_id}: {e}")
                return []

            archive = user_memory.get("long_term_archive", [])

            query_lower = query.lower()
            matching_entries = []

            for entry in archive:
                # Check conversations
                conversations = entry.get("conversations", [])
                for conv in conversations:
                    user_message = conv.get("user_message", "").lower()
                    ai_response = conv.get("ai_response", "").lower()

                    if query_lower in user_message or query_lower in ai_response:
                        matching_entries.append(entry)
                        break

            return matching_entries
        except Exception as e:
            logger.error(
                f"Error searching archived conversations for user {user_id}: {e}")
            return []

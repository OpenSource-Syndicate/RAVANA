"""
Hierarchical Memory Architecture for Enhanced Context-Aware Retrieval
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.config import Config
from modules.episodic_memory.memory import chroma_collection, sentence_transformer_ef

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HierarchicalMemoryStore:
    """Enhanced memory store with hierarchical organization and context-aware retrieval"""
    
    def __init__(self):
        self.episodic_memory = EpisodicMemoryStore()
        self.semantic_memory = SemanticMemoryStore()
        self.procedural_memory = ProceduralMemoryStore()
        self.meta_memory = MetaMemoryStore()
        self.context_cache = {}
    
    async def contextual_retrieval(self, query: str, context: Dict[str, Any], mood: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories with context-aware attention"""
        try:
            # Create cache key
            cache_key = f"{query}_{hash(str(context))}"
            
            # Check cache first
            if cache_key in self.context_cache:
                cached_result = self.context_cache[cache_key]
                # Check if cache is still valid (5 minutes)
                if (datetime.now() - cached_result['timestamp']).total_seconds() < 300:
                    logging.debug("Using cached memory retrieval result")
                    return cached_result['memories']
            
            # Compute context relevance scores
            context_scores = await self._compute_context_relevance(query, context)
            
            # Apply mood influence if provided
            if mood:
                mood_influence = await self._compute_mood_influence(mood)
                context_scores = self._apply_mood_modulation(context_scores, mood_influence)
            
            # Retrieve top-k memories based on combined scores
            memories = await self._retrieve_top_k_memories(context_scores, k=10)
            
            # Consolidate and summarize retrieved memories
            consolidated = await self._consolidate_memories(memories)
            
            # Cache the result
            self.context_cache[cache_key] = {
                'memories': consolidated,
                'timestamp': datetime.now()
            }
            
            # Clean up old cache entries
            self._cleanup_cache()
            
            return consolidated
            
        except Exception as e:
            logging.error(f"Error in contextual memory retrieval: {e}")
            # Fallback to simple retrieval
            return await self.episodic_memory.get_relevant_memories(query)
    
    async def _compute_context_relevance(self, query: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Compute relevance scores based on context"""
        try:
            # Extract context elements
            recent_events = context.get('recent_events', [])
            dominant_mood = context.get('dominant_mood', '')
            mood_vector = context.get('mood_vector', {})
            search_results = context.get('search_results', [])
            
            # Get all memories
            all_memories = await self.episodic_memory.get_all_memories()
            
            # Compute relevance scores
            relevance_scores = {}
            for memory in all_memories:
                memory_id = memory.get('id', '')
                memory_text = memory.get('text', '')
                
                score = 0.0
                
                # Query similarity (base score)
                query_similarity = self._compute_text_similarity(query, memory_text)
                score += query_similarity * 0.4
                
                # Recent events relevance
                for event in recent_events:
                    event_context = event.get('context', '')
                    event_similarity = self._compute_text_similarity(event_context, memory_text)
                    score += event_similarity * 0.2
                
                # Mood relevance
                if dominant_mood:
                    mood_keywords = self._get_mood_keywords(dominant_mood)
                    for keyword in mood_keywords:
                        if keyword.lower() in memory_text.lower():
                            score += 0.1
                
                # Search results relevance
                for result in search_results:
                    result_text = str(result)[:500]  # Limit length for performance
                    result_similarity = self._compute_text_similarity(result_text, memory_text)
                    score += result_similarity * 0.3
                
                relevance_scores[memory_id] = min(1.0, score)  # Cap at 1.0
            
            return relevance_scores
            
        except Exception as e:
            logging.error(f"Error computing context relevance: {e}")
            return {}
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        try:
            # Simple word overlap approach for now
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        except Exception as e:
            logging.error(f"Error computing text similarity: {e}")
            return 0.0
    
    async def _compute_mood_influence(self, mood: Dict[str, float]) -> Dict[str, float]:
        """Compute mood influence on memory retrieval"""
        # Simple approach: use dominant mood to influence retrieval
        if not mood:
            return {}
        
        # Find dominant mood
        dominant_mood = max(mood.items(), key=lambda x: x[1])[0] if mood else ''
        
        # Map moods to keywords that might influence memory retrieval
        mood_keywords = self._get_mood_keywords(dominant_mood)
        
        return {keyword: 0.1 for keyword in mood_keywords}
    
    def _get_mood_keywords(self, mood: str) -> List[str]:
        """Get keywords associated with a mood"""
        mood_keyword_map = {
            'Confident': ['success', 'achievement', 'capability', 'skill'],
            'Curious': ['learn', 'discover', 'explore', 'question', 'investigate'],
            'Frustrated': ['difficulty', 'challenge', 'problem', 'struggle', 'obstacle'],
            'Excited': ['opportunity', 'potential', 'new', 'innovation'],
            'Reflective': ['think', 'consider', 'analyze', 'evaluate', 'review'],
            'Bored': ['repetitive', 'routine', 'monotonous', 'uninteresting'],
            'Inspired': ['creativity', 'idea', 'innovation', 'breakthrough'],
            'Anxious': ['uncertainty', 'risk', 'concern', 'worry'],
            'Satisfied': ['accomplishment', 'completion', 'fulfillment']
        }
        
        return mood_keyword_map.get(mood, [])
    
    def _apply_mood_modulation(self, scores: Dict[str, float], mood_influence: Dict[str, float]) -> Dict[str, float]:
        """Apply mood influence to relevance scores"""
        if not mood_influence:
            return scores
        
        # Modulate scores based on mood keywords
        modulated_scores = scores.copy()
        
        # This would require access to memory content to check for keywords
        # For now, we'll apply a simple global modulation
        mood_factor = sum(mood_influence.values()) / len(mood_influence) if mood_influence else 0
        mood_factor = min(0.2, mood_factor)  # Cap the influence
        
        for memory_id in modulated_scores:
            modulated_scores[memory_id] *= (1.0 + mood_factor)
            modulated_scores[memory_id] = min(1.0, modulated_scores[memory_id])  # Cap at 1.0
        
        return modulated_scores
    
    async def _retrieve_top_k_memories(self, scores: Dict[str, float], k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k memories based on scores"""
        try:
            # Sort memories by score
            sorted_memories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_k_ids = [mem_id for mem_id, score in sorted_memories[:k]]
            
            # Retrieve full memory objects
            top_memories = []
            for mem_id in top_k_ids:
                memory = await self.episodic_memory.get_memory_by_id(mem_id)
                if memory:
                    top_memories.append(memory)
            
            return top_memories
        except Exception as e:
            logging.error(f"Error retrieving top-k memories: {e}")
            return []
    
    async def _consolidate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate and summarize retrieved memories"""
        try:
            # Group similar memories
            grouped_memories = self._group_similar_memories(memories)
            
            # For each group, create a consolidated representation
            consolidated = []
            for group in grouped_memories:
                if len(group) == 1:
                    # Single memory, keep as is
                    consolidated.append(group[0])
                else:
                    # Multiple similar memories, create summary
                    summary = self._summarize_memory_group(group)
                    consolidated.append(summary)
            
            return consolidated
        except Exception as e:
            logging.error(f"Error consolidating memories: {e}")
            return memories
    
    def _group_similar_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar memories together"""
        groups = []
        used_indices = set()
        
        for i, memory1 in enumerate(memories):
            if i in used_indices:
                continue
                
            current_group = [memory1]
            used_indices.add(i)
            
            for j, memory2 in enumerate(memories):
                if j in used_indices or i == j:
                    continue
                
                # Check similarity
                similarity = self._compute_text_similarity(
                    memory1.get('text', ''), 
                    memory2.get('text', '')
                )
                
                if similarity > 0.7:  # Threshold for grouping
                    current_group.append(memory2)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _summarize_memory_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of a group of similar memories"""
        if not group:
            return {}
        
        # Simple approach: combine texts and take most recent timestamp
        combined_text = ' | '.join([mem.get('text', '') for mem in group])
        latest_timestamp = max([mem.get('created_at', '') for mem in group])
        
        # Create consolidated memory
        consolidated = {
            'id': f"consolidated_{hash(combined_text)}",
            'text': f"Consolidated memory: {combined_text}",
            'created_at': latest_timestamp,
            'type': 'consolidated',
            'source_count': len(group)
        }
        
        return consolidated
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, value in self.context_cache.items():
                if (current_time - value['timestamp']).total_seconds() > 1800:  # 30 minutes
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.context_cache[key]
                
        except Exception as e:
            logging.error(f"Error cleaning up cache: {e}")

class EpisodicMemoryStore:
    """Store for episodic memories (personal experiences and events)"""
    
    async def get_relevant_memories(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get relevant episodic memories based on query"""
        try:
            # Use existing ChromaDB functionality
            results = chroma_collection.query(
                query_texts=[query],
                n_results=top_n,
                include=["documents", "metadatas"]
            )
            
            memories = []
            for i, doc in enumerate(results["documents"][0]):
                memory = {
                    "id": results["ids"][0][i],
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": results["distances"][0][i] if results["distances"] else 0
                }
                memories.append(memory)
            
            return memories
        except Exception as e:
            logging.error(f"Error retrieving episodic memories: {e}")
            return []
    
    async def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all episodic memories"""
        try:
            # Get all memories from ChromaDB
            results = chroma_collection.get(
                include=["documents", "metadatas"]
            )
            
            memories = []
            for i, doc in enumerate(results["documents"]):
                memory = {
                    "id": results["ids"][i],
                    "text": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                }
                memories.append(memory)
            
            return memories
        except Exception as e:
            logging.error(f"Error retrieving all episodic memories: {e}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        try:
            results = chroma_collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }
            
            return None
        except Exception as e:
            logging.error(f"Error retrieving memory by ID: {e}")
            return None

class SemanticMemoryStore:
    """Store for semantic memories (facts, concepts, general knowledge)"""
    
    def __init__(self):
        # In a full implementation, this would connect to a knowledge base
        self.knowledge_base = {}
    
    async def get_relevant_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant semantic knowledge based on query"""
        # Placeholder implementation
        return []

class ProceduralMemoryStore:
    """Store for procedural memories (skills, routines, procedures)"""
    
    def __init__(self):
        # In a full implementation, this would store action sequences and procedures
        self.procedures = {}
    
    async def get_relevant_procedures(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant procedures based on query"""
        # Placeholder implementation
        return []

class MetaMemoryStore:
    """Store for meta-memories (memories about memories, learning strategies)"""
    
    def __init__(self:
        # In a full implementation, this would store memory management strategies
        self.meta_knowledge = {}
    
    async def get_memory_strategies(self, context: str) -> List[Dict[str, Any]]:
        """Get memory management strategies based on context"""
        # Placeholder implementation
        return []
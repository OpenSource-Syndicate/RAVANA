"""
Consciousness Integration Module for RAVANA AGI System
Integrates consciousness features into the main AGI system
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core.consciousness_core import ConsciousnessCore
from core.state import SharedState

logger = logging.getLogger(__name__)


class ConsciousnessIntegration:
    """
    Module for integrating consciousness features into the main AGI system
    """
    
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.consciousness_core = ConsciousnessCore(agi_system)
        self.enabled = True
        
    async def process_with_consciousness(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process cognitive inputs through the consciousness architecture.
        
        Args:
            inputs: Dictionary containing various cognitive inputs
            
        Returns:
            Processed inputs with consciousness-based modifications
        """
        if not self.enabled:
            # If consciousness is disabled, return inputs unchanged
            return inputs
        
        try:
            # Process inputs through consciousness architecture
            conscious_outputs = await self.consciousness_core.process_cognitive_inputs(inputs)
            
            # Update AGI system's shared state with consciousness information
            await self._update_shared_state()
            
            # Perform introspection on the processing
            introspection_result = await self.consciousness_core.introspect()
            
            # Log consciousness-related information
            logger.info(f"Consciousness processing completed. Introspection: {introspection_result}")
            
            return conscious_outputs
            
        except Exception as e:
            logger.error(f"Error in consciousness processing: {e}")
            # Return original inputs if consciousness processing fails
            return inputs
    
    async def reflect_on_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Perform consciousness-aware reflection on an experience."""
        if not self.enabled:
            return {}
        
        try:
            reflection = await self.consciousness_core.reflect_on_experience(experience)
            logger.info(f"Consciousness-aware reflection completed: {reflection.get('learning_extracted', [])[:2]}")
            return reflection
        except Exception as e:
            logger.error(f"Error in consciousness-aware reflection: {e}")
            return {}
    
    async def get_consciousness_state(self) -> Dict[str, Any]:
        """Get the current consciousness state."""
        if not self.enabled:
            return {}
        
        state = self.consciousness_core.get_consciousness_state()
        return {
            'attention_focus': state.attention_focus,
            'self_model': state.self_model,
            'experience_stream_length': len(state.experience_stream),
            'metacognitive_state': state.metacognitive_state,
            'timestamp': state.timestamp.isoformat()
        }
    
    async def _update_shared_state(self):
        """Update the AGI system's shared state with consciousness information."""
        consciousness_state = self.consciousness_core.get_consciousness_state()
        
        # Update shared state with attention information
        if hasattr(self.agi_system, 'shared_state'):
            shared_state: SharedState = self.agi_system.shared_state
            shared_state.attention_focus = consciousness_state.attention_focus
            shared_state.consciousness_metrics = {
                'awareness_level': len(consciousness_state.attention_focus),
                'self_knowledge_update_count': len(consciousness_state.self_model.get('history', [])),
                'experience_coherence': len(consciousness_state.experience_stream)
            }
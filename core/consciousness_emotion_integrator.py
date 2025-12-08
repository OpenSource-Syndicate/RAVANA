"""
Integration module for connecting enhanced consciousness and emotional intelligence systems,
enabling bidirectional influence and mutual enhancement.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from core.enhanced_consciousness_core import EnhancedConsciousnessCore
from modules.emotional_intellegence.enhanced_emotional_intellegence import EnhancedEmotionalIntelligence


logger = logging.getLogger(__name__)


class ConsciousnessEmotionIntegrator:
    """
    Integrates enhanced consciousness and emotional intelligence systems 
    with bidirectional influence and feedback mechanisms.
    """
    def __init__(self, consciousness_core: EnhancedConsciousnessCore, 
                 emotional_intelligence: EnhancedEmotionalIntelligence):
        self.consciousness_core = consciousness_core
        self.emotional_intelligence = emotional_intelligence
        
        # Integration parameters
        self.emotional_modulation_strength = 0.5  # How much emotions modulate consciousness
        self.consciousness_control_strength = 0.3  # How much consciousness controls emotions
        self.bidirectional_coupling = 0.7  # Strength of bidirectional influence
        
        # Feedback loops
        self.emotion_to_consciousness_feedback = {}
        self.consciousness_to_emotion_feedback = {}
        
        # Integration history
        self.integration_history: List[Dict[str, Any]] = []
        self.feedback_loops: List[Dict[str, Any]] = []
        
        # Dynamic adjustment parameters
        self.adjustment_sensitivity = 0.1
        self.balance_goal = 0.5  # Desired balance between consciousness and emotion
        
        # Integration state
        self.integration_coherence = 0.6  # How well systems work together
        self.integration_stability = 0.7  # Stability of integration
        self.feedback_delay = 0.05  # Delay in feedback propagation (in seconds)

    def integrate_processing_with_emotions(self, inputs: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process inputs with integrated consciousness and emotional intelligence
        """
        context = context or {}
        
        # Update context with emotional state information
        emotional_context = self.emotional_intelligence.get_enhanced_emotional_context()
        context.update(emotional_context)
        
        # Process through consciousness system
        consciousness_output = self.consciousness_core.process_with_dynamic_consciousness(
            inputs, context
        )
        
        # Process through emotional system
        emotional_output = self.emotional_intelligence.process_action_result_enhanced(
            consciousness_output, context
        )
        
        # Integrate both outputs with bidirectional influence
        integrated_output = self._bidirectional_integration(
            consciousness_output, emotional_output, context
        )
        
        # Apply feedback mechanisms
        self._apply_feedback_mechanisms(consciousness_output, emotional_output)
        
        # Record integration event
        self._record_integration_event(consciousness_output, emotional_output, integrated_output, context)
        
        # Update integration parameters dynamically
        self._update_integration_parameters(consciousness_output, emotional_output)
        
        return {
            'consciousness_output': consciousness_output,
            'emotional_output': emotional_output,
            'integrated_output': integrated_output,
            'integration_metrics': self._calculate_integration_metrics()
        }

    def _bidirectional_integration(self, consciousness_output: Dict[str, Any], 
                                 emotional_output: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform bidirectional integration with consciousness-emotion coupling
        """
        # Extract key components
        con_attended = consciousness_output.get('attended_inputs', {})
        con_processed = consciousness_output.get('processed_content', {})
        emo_impact = emotional_output.get('emotional_impact', {})
        emo_regulated = emotional_output.get('regulated_impact', {})
        
        # Apply emotional modulation to consciousness processing
        modulated_processed = self._apply_emotional_modulation(con_processed, emo_regulated)
        
        # Apply consciousness regulation to emotional processing
        regulated_emotional = self._apply_consciousness_regulation(emo_impact, con_processed)
        
        # Integrate attention with emotional salience
        integrated_attention = self._integrate_attention_emotion(
            con_attended, emo_impact, context
        )
        
        # Calculate emotional influence on working memory
        emotional_WM_influence = self._calculate_emotional_WM_influence(emo_regulated)
        
        # Update working memory with emotional influence
        self._update_working_memory_with_emotion(emotional_WM_influence)
        
        # Integrate global workspace with emotional weighting
        integrated_workspace = self._integrate_workspace_emotion(
            consciousness_output.get('integrated_content', {}), 
            emotional_output.get('mood_changes', {})
        )
        
        return {
            'attended_inputs': integrated_attention,
            'processed_content': modulated_processed,
            'emotional_impact': regulated_emotional,
            'integrated_workspace': integrated_workspace,
            'metacognitive_output': consciousness_output.get('metacognitive_output', {}),
            'emotional_state': emotional_output.get('emotional_state', {}),
            'integration_strength': self.bidirectional_coupling
        }

    def _apply_emotional_modulation(self, processed_content: Dict[str, Any], 
                                  emotional_impact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply emotional state modulation to consciousness processing
        """
        modulated_content = {}
        
        for key, value in processed_content.items():
            # Apply emotional weighting based on emotional impact
            emotional_weight = 0.1  # Base weight
            
            # Calculate emotional relevance to content
            for emotion, impact in emotional_impact.items():
                if emotion.lower() in str(key).lower() or emotion.lower() in str(value).lower():
                    emotional_weight += abs(impact) * 0.2
            
            # Apply modulation based on integration strength
            modulation_factor = 1.0 + (emotional_weight * self.emotional_modulation_strength)
            
            # Apply modulation to content
            if isinstance(value, dict):
                modulated_value = value.copy()
                modulated_value['activation'] = value.get('activation', 0.5) * modulation_factor
                modulated_content[key] = modulated_value
            else:
                modulated_content[key] = {
                    'content': value,
                    'activation': 0.5 * modulation_factor,
                    'emotional_modulation': emotional_weight
                }
        
        return modulated_content

    def _apply_consciousness_regulation(self, emotional_impact: Dict[str, Any], 
                                      processed_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply consciousness-based regulation to emotional responses
        """
        regulated_impact = {}
        
        for emotion, impact in emotional_impact.items():
            # Apply consciousness regulation based on metacognitive assessment
            regulation_strength = 0.1  # Base regulation
            
            # Check for metacognitive insights about this emotion
            if 'metacognitive_output' in processed_content:
                if 'suggestions' in processed_content['metacognitive_output']:
                    # If this emotion is flagged in suggestions, increase regulation
                    suggestions = processed_content['metacognitive_output']['suggestions']
                    for suggestion in suggestions:
                        if emotion.lower() in suggestion.lower():
                            regulation_strength += 0.2
            
            # Apply regulation factor
            regulation_factor = 1 - (regulation_strength * self.consciousness_control_strength)
            
            # Apply to emotional impact
            regulated_impact[emotion] = impact * regulation_factor
        
        return regulated_impact

    def _integrate_attention_emotion(self, attended_inputs: Dict[str, Any], 
                                   emotional_impact: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate attention allocation with emotional salience
        """
        integrated_inputs = {}
        
        # Determine emotional salience for each attended input
        emotional_salience = {}
        for key, value in attended_inputs.items():
            salience = 0.1  # Base salience
            
            # Check for emotional relevance
            for emotion, impact in emotional_impact.items():
                if emotion.lower() in str(key).lower() or emotion.lower() in str(value).lower():
                    salience += abs(impact) * 0.3
            
            emotional_salience[key] = salience
        
        # Adjust attention weights based on emotional salience
        for key, value in attended_inputs.items():
            emotional_weight = emotional_salience[key]
            
            # Apply emotional weighting to activation levels
            if isinstance(value, dict):
                adjusted_value = value.copy()
                current_activation = value.get('activation', 0.5)
                
                # Boost activation based on emotional salience
                adjusted_activation = current_activation * (1 + emotional_weight)
                adjusted_value['activation'] = min(1.0, adjusted_activation)
                
                # Add emotional salience information
                adjusted_value['emotional_salience'] = emotional_weight
                integrated_inputs[key] = adjusted_value
            else:
                integrated_inputs[key] = {
                    'content': value,
                    'activation': 0.5 * (1 + emotional_weight),
                    'emotional_salience': emotional_weight
                }
        
        return integrated_inputs

    def _calculate_emotional_WM_influence(self, emotional_impact: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate how emotions influence working memory operations
        """
        wm_influence = {}
        
        for emotion, impact in emotional_impact.items():
            # Different emotions have different effects on working memory
            if emotion in ['Anxious', 'Stressed']:
                # Anxiety typically reduces working memory capacity
                wm_influence['capacity_modulation'] = -abs(impact) * 0.3
            elif emotion in ['Happy', 'Excited']:
                # Positive emotions can enhance working memory
                wm_influence['capacity_modulation'] = abs(impact) * 0.1
            elif emotion in ['Focused', 'Concentrated']:
                # Focused emotions improve working memory efficiency
                wm_influence['efficiency_modulation'] = abs(impact) * 0.2
            elif emotion in ['Tired', 'Fatigued']:
                # Fatigue reduces working memory performance
                wm_influence['decay_acceleration'] = abs(impact) * 0.2
        
        return wm_influence

    def _update_working_memory_with_emotion(self, emotional_influence: Dict[str, float]):
        """
        Update working memory parameters based on emotional influence
        """
        if not emotional_influence:
            return
        
        # Update working memory dynamics with emotional influence
        if 'capacity_modulation' in emotional_influence:
            # Adjust working memory capacity based on emotion
            current_capacity = self.consciousness_core.working_memory_dynamics.state.capacity
            capacity_change = emotional_influence['capacity_modulation']
            new_capacity = max(3, current_capacity + int(capacity_change * 2))
            self.consciousness_core.working_memory_dynamics.state.capacity = new_capacity
        
        if 'decay_acceleration' in emotional_influence:
            # Adjust decay rate based on emotion
            current_decay = self.consciousness_core.working_memory_dynamics.state.decay_rate
            decay_change = emotional_influence['decay_acceleration']
            new_decay = current_decay * (1 + decay_change)
            self.consciousness_core.working_memory_dynamics.state.decay_rate = min(0.8, new_decay)
        
        if 'efficiency_modulation' in emotional_influence:
            # Adjust rehearsal efficiency based on emotion
            current_efficiency = self.consciousness_core.working_memory_dynamics.state.rehearsal_efficiency
            efficiency_change = emotional_influence['efficiency_modulation']
            new_efficiency = current_efficiency * (1 + efficiency_change)
            self.consciousness_core.working_memory_dynamics.state.rehearsal_efficiency = min(1.0, new_efficiency)

    def _integrate_workspace_emotion(self, workspace_content: Dict[str, Any], 
                                   mood_changes: Dict[str, float]) -> Dict[str, Any]:
        """
        Integrate global workspace content with emotional weighting
        """
        integrated_content = {}
        
        for key, value in workspace_content.items():
            # Apply mood congruence effects
            mood_congruence = 0.0
            
            # Check if content is congruent with current mood
            for mood, change in mood_changes.items():
                if mood.lower() in str(key).lower() or mood.lower() in str(value).lower():
                    mood_congruence += abs(change)
            
            # Apply mood congruence weighting to workspace access
            if isinstance(value, dict):
                weighted_value = value.copy()
                
                # Adjust priority based on mood congruence
                current_priority = value.get('priority', 0.5)
                mood_weight = min(0.3, mood_congruence * 0.5)  # Limit influence
                new_priority = max(0.1, min(1.0, current_priority + mood_weight))
                
                weighted_value['priority'] = new_priority
                weighted_value['mood_congruence'] = mood_congruence
                integrated_content[key] = weighted_value
            else:
                integrated_content[key] = {
                    'content': value,
                    'priority': 0.5 + min(0.3, mood_congruence * 0.5),
                    'mood_congruence': mood_congruence
                }
        
        return integrated_content

    def _apply_feedback_mechanisms(self, consciousness_output: Dict[str, Any], 
                                 emotional_output: Dict[str, Any]):
        """
        Apply feedback mechanisms between consciousness and emotional systems
        """
        # Extract relevant metrics
        con_confidence = consciousness_output.get('metacognitive_output', {}).get('confidence', 0.5)
        emo_valence = emotional_output.get('emotional_state', {}).get('valence', 0.0)
        emo_regulation_cap = emotional_output.get('emotional_state', {}).get('regulation_capacity', 0.6)
        
        # Create feedback loops
        emotion_to_consciousness = {
            'valence_influence': emo_valence,
            'regulation_support': emo_regulation_cap,
            'timestamp': datetime.now()
        }
        
        consciousness_to_emotion = {
            'awareness_clarity': con_confidence,
            'cognitive_control': con_confidence * emo_regulation_cap,
            'timestamp': datetime.now()
        }
        
        # Apply emotion-to-consciousness feedback
        self._apply_emotion_to_consciousness_feedback(emotion_to_consciousness)
        
        # Apply consciousness-to-emotion feedback
        self._apply_consciousness_to_emotion_feedback(consciousness_to_emotion)
        
        # Record feedback event
        feedback_event = {
            'emotion_to_consciousness': emotion_to_consciousness,
            'consciousness_to_emotion': consciousness_to_emotion,
            'timestamp': datetime.now()
        }
        
        self.feedback_loops.append(feedback_event)
        
        # Keep only recent feedback loops
        if len(self.feedback_loops) > 50:
            self.feedback_loops = self.feedback_loops[-50:]

    def _apply_emotion_to_consciousness_feedback(self, feedback: Dict[str, Any]):
        """
        Apply emotional feedback to consciousness parameters
        """
        # Adjust consciousness sensitivity based on emotional valence
        valence_influence = feedback.get('valence_influence', 0.0)
        
        # Positive emotions can increase openness to experience
        if valence_influence > 0.3:
            # Increase attention span temporarily
            self.consciousness_core.attention_dynamics.state.attention_span = min(
                10, 
                self.consciousness_core.attention_dynamics.state.attention_span + 1
            )
        elif valence_influence < -0.3:
            # Reduce attention span with negative emotions
            self.consciousness_core.attention_dynamics.state.attention_span = max(
                3,
                self.consciousness_core.attention_dynamics.state.attention_span - 1
            )
        
        # Apply regulation support to attention
        reg_support = feedback.get('regulation_support', 0.6)
        self.consciousness_core.attention_dynamics.state.distraction_resistance = min(
            0.9,
            self.consciousness_core.attention_dynamics.state.distraction_resistance + (reg_support - 0.5) * 0.1
        )

    def _apply_consciousness_to_emotion_feedback(self, feedback: Dict[str, Any]):
        """
        Apply consciousness feedback to emotional parameters
        """
        # Adjust emotional regulation based on consciousness clarity
        awareness_clarity = feedback.get('awareness_clarity', 0.5)
        cognitive_control = feedback.get('cognitive_control', 0.5)
        
        # Update emotional regulation capacity
        self.emotional_intelligence.emotional_state.regulation_capacity = (
            self.emotional_intelligence.emotional_state.regulation_capacity * 0.7 + 
            awareness_clarity * 0.2 + 
            cognitive_control * 0.1
        )
        
        # Update emotional recovery rate
        self.emotional_intelligence.emotional_state.recovery_rate = (
            self.emotional_intelligence.emotional_state.recovery_rate * 0.8 + 
            cognitive_control * 0.2
        )

    def _record_integration_event(self, consciousness_output: Dict[str, Any], 
                                emotional_output: Dict[str, Any], 
                                integrated_output: Dict[str, Any], 
                                context: Dict[str, Any]):
        """
        Record integration event for historical analysis and learning
        """
        event = {
            'timestamp': datetime.now(),
            'consciousness_metrics': consciousness_output.get('consciousness_metrics', {}),
            'emotional_metrics': emotional_output.get('emotional_state', {}),
            'context': context,
            'integration_strength': self.bidirectional_coupling
        }
        
        self.integration_history.append(event)
        
        # Keep only recent history
        if len(self.integration_history) > 100:
            self.integration_history = self.integration_history[-100:]

    def _update_integration_parameters(self, consciousness_output: Dict[str, Any], 
                                     emotional_output: Dict[str, Any]):
        """
        Update integration parameters dynamically based on system outputs
        """
        # Calculate balance between systems
        con_confidence = consciousness_output.get('metacognitive_output', {}).get('confidence', 0.5)
        emo_regulation = emotional_output.get('emotional_state', {}).get('regulation_capacity', 0.6)
        
        # Calculate current balance
        current_balance = (con_confidence + emo_regulation) / 2
        
        # Adjust coupling strength based on balance
        if abs(current_balance - self.balance_goal) > 0.2:
            # If systems are unbalanced, reduce coupling to allow stabilization
            self.bidirectional_coupling = max(0.3, self.bidirectional_coupling - 0.05)
        else:
            # If well balanced, gradually increase coupling
            self.bidirectional_coupling = min(0.9, self.bidirectional_coupling + 0.01)
        
        # Update emotional modulation strength based on consciousness clarity
        self.emotional_modulation_strength = 0.4 + (con_confidence * 0.2)
        
        # Update consciousness control strength based on emotional regulation capacity
        self.consciousness_control_strength = 0.2 + (emo_regulation * 0.2)
        
        # Update integration coherence based on consistency of outputs
        self.integration_coherence = self._calculate_integration_coherence(
            consciousness_output, emotional_output
        )

    def _calculate_integration_coherence(self, consciousness_output: Dict[str, Any], 
                                       emotional_output: Dict[str, Any]) -> float:
        """
        Calculate how well consciousness and emotional outputs align
        """
        # This is a simplified calculation; in a real implementation, 
        # this would use more sophisticated alignment measures
        
        metacog_output = consciousness_output.get('metacognitive_output', {})
        emo_state = emotional_output.get('emotional_state', {})
        
        # Check alignment between confidence and emotional regulation
        con_conf = metacog_output.get('confidence', 0.5)
        emo_reg = emo_state.get('regulation_capacity', 0.5)
        
        # Calculate alignment score based on how compatible these are
        alignment_diff = abs(con_conf - emo_reg)
        coherence_score = max(0.1, 1.0 - alignment_diff)
        
        return coherence_score

    def _calculate_integration_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics related to consciousness-emotion integration
        """
        if not self.integration_history:
            return {
                'integration_coherence': self.integration_coherence,
                'coupling_strength': self.bidirectional_coupling,
                'emotional_modulation_strength': self.emotional_modulation_strength,
                'consciousness_control_strength': self.consciousness_control_strength,
                'balance_score': 0.5
            }
        
        # Calculate recent averages
        recent_events = self.integration_history[-10:] if len(self.integration_history) > 10 else self.integration_history
        avg_balance = sum(
            (event['consciousness_metrics'].get('metacognitive_accuracy', 0.5) + 
             event['emotional_metrics'].get('regulation_capacity', 0.5)) / 2
            for event in recent_events
        ) / len(recent_events) if recent_events else 0.5
        
        return {
            'integration_coherence': self.integration_coherence,
            'coupling_strength': self.bidirectional_coupling,
            'emotional_modulation_strength': self.emotional_modulation_strength,
            'consciousness_control_strength': self.consciousness_control_strength,
            'balance_score': avg_balance,
            'integration_stability': self.integration_stability,
            'recent_integration_events': len(recent_events)
        }

    def activate_emotional_awareness(self, depth: float = 0.7) -> Dict[str, Any]:
        """
        Activate emotional awareness through consciousness system
        """
        # Get current emotional state
        emotional_state = self.emotional_intelligence.get_enhanced_emotional_context()
        
        # Enhance consciousness to pay attention to emotional state
        attention_context = {
            'focus_on_emotions': True,
            'emotional_relevance': depth,
            'self_awareness_mode': True
        }
        
        # Process emotional state through consciousness to create awareness
        awareness_input = {
            'emotional_state': emotional_state,
            'emotional_changes': self.emotional_intelligence.mood_vector,
            'recent_emotional_events': len(self.emotional_intelligence.emotional_events)
        }
        
        awareness_output = self.consciousness_core.process_with_dynamic_consciousness(
            awareness_input, 
            attention_context
        )
        
        # Apply metacognitive reflection on emotional state
        reflection_result = self.consciousness_core.self_model_dynamics.reflect_on_self(depth=depth)
        
        return {
            'emotional_awareness': awareness_output,
            'emotional_insight': reflection_result,
            'consciousness_emotion_alignment': self._calculate_alignment_score(awareness_output, emotional_state)
        }

    def _calculate_alignment_score(self, consciousness_output: Dict[str, Any], 
                                 emotional_state: Dict[str, Any]) -> float:
        """
        Calculate how well consciousness and emotional states are aligned
        """
        # Simplified alignment measure
        # In a real implementation, this would use more sophisticated cross-modal measures
        con_attention = consciousness_output.get('consciousness_metrics', {}).get('attention_focus', 0.5)
        emo_valence = emotional_state.get('valence', 0.0)
        emo_regulation = emotional_state.get('regulation_capacity', 0.5)
        
        # Alignment is higher when consciousness attention aligns with emotional regulation
        attention_reg_align = 1 - abs(con_attention - emo_regulation)
        valence_reg_align = 1 - abs(emo_valence) * (1 - emo_regulation)  # Low valence with high regulation is good alignment
        
        alignment_score = (attention_reg_align * 0.6 + valence_reg_align * 0.4)
        
        return max(0.0, min(1.0, alignment_score))

    def regulate_emotional_response(self, target_emotion: str, target_intensity: float,
                                  consciousness_guidance: bool = True) -> Dict[str, Any]:
        """
        Regulate emotional response with consciousness guidance
        """
        if consciousness_guidance:
            # First, engage consciousness to identify optimal regulation strategy
            metacognitive_insight = self.consciousness_core.metacognitive_system.monitor_process(
                {'target_emotion': target_emotion, 'target_intensity': target_intensity},
                {'regulation_task': True}
            )
            
            # Select appropriate emotional regulation strategy based on insight
            strategy = self._select_emotional_strategy_from_insight(metacognitive_insight)
        else:
            strategy = None  # Let emotion system choose default strategy
        
        # Apply regulation through emotional intelligence system
        regulation_result = self.emotional_intelligence.regulate_emotion(
            target_emotion, target_intensity, strategy
        )
        
        # Update consciousness about the regulation process
        self.consciousness_core.adapt_to_cognitive_load(
            1 - self.emotional_intelligence.emotional_state.regulation_capacity
        )
        
        return {
            'regulation_result': regulation_result,
            'metacognitive_insight': metacognitive_insight if consciousness_guidance else None,
            'strategy_used': strategy,
            'post_regulation_emotional_state': self.emotional_intelligence.emotional_state.__dict__.copy()
        }

    def _select_emotional_strategy_from_insight(self, insight: Dict[str, Any]):
        """
        Select optimal emotional regulation strategy based on metacognitive insight
        """
        from modules.emotional_intellegence.enhanced_emotional_intellegence import EmotionalRegulationStrategy
        
        # Analyze insight for appropriate strategy selection
        if insight and 'suggestions' in insight:
            suggestions = insight['suggestions']
            
            # Look for cues in suggestions about what strategy to use
            if any('cognitive' in s.lower() for s in suggestions):
                return EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL
            elif any('attention' in s.lower() for s in suggestions):
                return EmotionalRegulationStrategy.ATTENTION_DEPLOYMENT
            elif any('accept' in s.lower() for s in suggestions):
                return EmotionalRegulationStrategy.ACCEPTANCE
            elif any('situation' in s.lower() for s in suggestions):
                return EmotionalRegulationStrategy.SITUATION_SELECTION
        
        # Default to cognitive reappraisal for high-level systems
        return EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL

    def get_dynamic_balance_recommendations(self) -> List[str]:
        """
        Get recommendations for maintaining optimal consciousness-emotion balance
        """
        recommendations = []
        
        # Check current balance metrics
        metrics = self._calculate_integration_metrics()
        
        if metrics['balance_score'] < 0.3:
            recommendations.append(
                "Systems are imbalanced. Consider engaging consciousness to regulate emotional responses."
            )
        elif metrics['balance_score'] > 0.7:
            recommendations.append(
                "Over-reliance on rational processing detected. Consider allowing more emotional input."
            )
        
        if metrics['integration_coherence'] < 0.4:
            recommendations.append(
                "Poor integration between systems. Reduce coupling strength temporarily."
            )
        
        if self.emotional_modulation_strength > 0.7 and self.consciousness_control_strength < 0.3:
            recommendations.append(
                "Emotional influence is too strong. Engage metacognitive awareness."
            )
        
        if self.consciousness_control_strength > 0.6 and self.emotional_modulation_strength < 0.3:
            recommendations.append(
                "Cognitive control is suppressing emotional processing. Allow more emotional awareness."
            )
        
        return recommendations
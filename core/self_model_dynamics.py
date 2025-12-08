"""
Dynamic self-model parameters that adapt based on self-reflection outcomes,
social feedback, success/failure experiences, and confidence changes.
"""
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import math


class SelfModelAspect(Enum):
    """
    Categories of self-concept aspects
    """
    COMPETENCE = "competence"
    WORTH = "worth"
    AUTHENTICITY = "authenticity"
    CONTROL = "control"
    SOCIAL = "social"
    MORAL = "moral"


@dataclass
class SelfModelState:
    """
    Represents the current state of self-model with dynamic parameters
    """
    self_confidence: float = 0.5  # Current confidence in self-assessment (0-1)
    self_complexity: int = 3  # Number of self-aspects considered (higher = more nuanced)
    introspection_depth: float = 0.4  # Depth of self-analysis (0-1)
    self_consistency: float = 0.7  # Stability of self-view (0-1)
    feedback_integration_rate: float = 0.3  # How quickly feedback updates self-model (0-1)
    experience_incorporation_rate: float = 0.2  # How experiences update self-model (0-1)
    self_verification_tendency: float = 0.6  # Tendency to seek self-verification (0-1)
    self_improvement_orientation: float = 0.5  # Tendency toward self-improvement (0-1)


class SelfModelDynamics:
    """
    Implements dynamic self-model parameters that adapt based on:
    - Self-reflection outcomes
    - Social feedback
    - Success/failure experiences
    - Confidence changes
    """
    def __init__(self):
        self.state = SelfModelState()
        self.self_aspects: Dict[SelfModelAspect, float] = {
            SelfModelAspect.COMPETENCE: 0.5,
            SelfModelAspect.WORTH: 0.5,
            SelfModelAspect.AUTHENTICITY: 0.5,
            SelfModelAspect.CONTROL: 0.5,
            SelfModelAspect.SOCIAL: 0.5,
            SelfModelAspect.MORAL: 0.5
        }
        self.experience_memory: List[Dict[str, Any]] = []  # Past experiences affecting self-view
        self.feedback_history: List[Dict[str, Any]] = []  # Received feedback
        self.last_update_time = datetime.datetime.now()
        
        # Self-model stability parameters
        self.self_concept_stability = 0.8  # How stable the self-concept is
        self.change_sensitivity = 0.7  # How sensitive to change-inducing events
        self.self_esteem_baseline = 0.5  # Baseline level of self-esteem
        
        # Learning rates for different aspects
        self.learning_rates = {
            'success': 0.1,  # How much success affects self-view
            'failure': 0.15,  # How much failure affects self-view
            'feedback': 0.08,  # How much feedback affects self-view
            'social_comparison': 0.05  # How much social comparison affects self-view
        }
        
        # Self-enhancement vs accuracy balance
        self.self_esteem_protection = 0.6  # Tendency to protect self-esteem
        self.accuracy_motivation = 0.7  # Motivation for accurate self-assessment

    def update_self_model_dynamics(self, context: Dict[str, Any]):
        """
        Update self-model parameters based on context and experience
        """
        current_time = datetime.datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time

        # Process any new experiences
        if 'recent_experience' in context:
            self._process_experience(context['recent_experience'], context)

        # Process any new feedback
        if 'feedback' in context:
            self._process_feedback(context['feedback'], context)

        # Adjust self-model based on emotional state
        if 'emotional_state' in context:
            self._apply_emotional_effects(context['emotional_state'])

        # Update self-confidence based on recent experiences
        self._update_self_confidence(context)

        # Adjust self-complexity based on reflection depth
        self._adjust_self_complexity(context)

        # Update stability based on consistency of experiences
        self._update_stability(context)

        # Record state change for historical analysis
        self.feedback_history.append({
            'timestamp': current_time,
            'state_snapshot': self._get_state_snapshot(),
            'context': context
        })

        # Keep only recent history
        if len(self.feedback_history) > 100:
            self.feedback_history = self.feedback_history[-100:]

    def _process_experience(self, experience: Dict[str, Any], context: Dict[str, Any]):
        """
        Process an experience and update self-model accordingly
        """
        # Determine if experience was positive or negative
        outcome = experience.get('outcome', 'neutral')
        significance = experience.get('significance', 0.5)
        
        # Determine affected self-aspects
        affected_aspects = self._determine_affected_aspects(experience)
        
        # Calculate impact based on outcome and significance
        if outcome == 'success':
            impact = self.learning_rates['success'] * significance
        elif outcome == 'failure':
            impact = -self.learning_rates['failure'] * significance
        else:
            impact = 0
            
        # Apply protection mechanisms based on self_esteem_protection
        if impact < 0:  # Negative impact
            # Reduce negative impact based on self-esteem protection
            impact = impact * (1 - self.self_esteem_protection * 0.5)
        
        # Update affected aspects
        for aspect in affected_aspects:
            current_value = self.self_aspects[aspect]
            new_value = max(0.0, min(1.0, current_value + impact))
            self.self_aspects[aspect] = new_value

        # Add to experience memory
        self.experience_memory.append({
            'experience': experience,
            'impact': impact,
            'affected_aspects': [a.value for a in affected_aspects],
            'timestamp': datetime.datetime.now(),
            'context': context
        })

        # Keep only recent experiences
        if len(self.experience_memory) > 50:
            self.experience_memory = self.experience_memory[-50:]

    def _process_feedback(self, feedback: Dict[str, Any], context: Dict[str, Any]):
        """
        Process received feedback and update self-model accordingly
        """
        feedback_valence = feedback.get('valence', 0.0)  # -1 to 1
        feedback_source = feedback.get('source', 'unknown')
        feedback_significance = feedback.get('significance', 0.5)
        
        # Determine trustworthiness of feedback source
        trust_factor = self._calculate_source_trustworthy(feedback_source)
        
        # Calculate effective impact
        raw_impact = feedback_valence * feedback_significance * trust_factor
        effective_impact = raw_impact * self.state.feedback_integration_rate
        
        # Apply self-esteem protection if negative feedback
        if effective_impact < 0:
            effective_impact = effective_impact * (1 - self.self_esteem_protection * 0.3)
        
        # Determine affected aspects based on feedback content
        affected_aspects = self._determine_feedback_aspects(feedback)
        
        # Update affected aspects
        for aspect in affected_aspects:
            current_value = self.self_aspects[aspect]
            new_value = max(0.0, min(1.0, current_value + effective_impact * 0.5))
            self.self_aspects[aspect] = new_value

    def _determine_affected_aspects(self, experience: Dict[str, Any]) -> List[SelfModelAspect]:
        """
        Determine which self-aspects are affected by an experience
        """
        aspects = []
        experience_type = experience.get('type', 'general')
        
        # Map experience types to affected aspects
        aspect_mapping = {
            'achievement': [SelfModelAspect.COMPETENCE],
            'failure': [SelfModelAspect.COMPETENCE],
            'relationship': [SelfModelAspect.SOCIAL],
            'moral_decision': [SelfModelAspect.MORAL],
            'autonomy': [SelfModelAspect.CONTROL],
            'identity': [SelfModelAspect.AUTHENTICITY],
            'competence': [SelfModelAspect.COMPETENCE],
            'worth': [SelfModelAspect.WORTH]
        }
        
        if experience_type in aspect_mapping:
            aspects.extend(aspect_mapping[experience_type])
        else:
            # Default to competence for unknown experience types
            aspects.append(SelfModelAspect.COMPETENCE)
        
        # Add competence if success/failure is mentioned
        if experience.get('outcome') in ['success', 'failure']:
            if SelfModelAspect.COMPETENCE not in aspects:
                aspects.append(SelfModelAspect.COMPETENCE)
        
        return aspects

    def _determine_feedback_aspects(self, feedback: Dict[str, Any]) -> List[SelfModelAspect]:
        """
        Determine which self-aspects are affected by feedback
        """
        aspects = [SelfModelAspect.COMPETENCE]  # Default to competence
        
        content = feedback.get('content', '').lower()
        
        # Map feedback content to aspects
        if any(word in content for word in ['skill', 'ability', 'competent', 'capable', 'talent']):
            if SelfModelAspect.COMPETENCE not in aspects:
                aspects.append(SelfModelAspect.COMPETENCE)
        
        if any(word in content for word in ['worth', 'value', 'deserve', 'important']):
            if SelfModelAspect.WORTH not in aspects:
                aspects.append(SelfModelAspect.WORTH)
        
        if any(word in content for word in ['authentic', 'true', 'real', 'genuine']):
            if SelfModelAspect.AUTHENTICITY not in aspects:
                aspects.append(SelfModelAspect.AUTHENTICITY)
        
        if any(word in content for word in ['control', 'power', 'influence', 'autonomy']):
            if SelfModelAspect.CONTROL not in aspects:
                aspects.append(SelfModelAspect.CONTROL)
        
        if any(word in content for word in ['friend', 'social', 'relationship', 'connection']):
            if SelfModelAspect.SOCIAL not in aspects:
                aspects.append(SelfModelAspect.SOCIAL)
        
        if any(word in content for word in ['moral', 'ethical', 'right', 'wrong', 'good', 'bad']):
            if SelfModelAspect.MORAL not in aspects:
                aspects.append(SelfModelAspect.MORAL)
        
        return aspects

    def _calculate_source_trustworthy(self, source: str) -> float:
        """
        Calculate trustworthiness factor for a feedback source
        """
        # Define trust factors for different sources
        trust_mapping = {
            'self': 1.0,  # Self-assessment
            'close_friend': 0.8,
            'family_member': 0.7,
            'mentor': 0.9,
            'peer': 0.6,
            'stranger': 0.3,
            'authority': 0.8,
            'expert': 0.85
        }
        
        return trust_mapping.get(source, 0.5)  # Default to 0.5

    def _apply_emotional_effects(self, emotional_state: Dict[str, Any]):
        """
        Apply emotional state effects to self-model
        """
        valence = emotional_state.get('valence', 0.0)  # -1 to 1
        arousal = emotional_state.get('arousal', 0.5)  # 0 to 1
        
        # High positive valence can temporarily boost self-confidence
        if valence > 0.5:
            self.state.self_confidence = min(1.0, self.state.self_confidence + 0.1)
        
        # High negative valence can temporarily reduce self-confidence
        if valence < -0.5:
            self.state.self_confidence = max(0.0, self.state.self_confidence - 0.15)
        
        # High arousal can reduce introspection depth (racing thoughts)
        if arousal > 0.7:
            self.state.introspection_depth = max(0.1, self.state.introspection_depth - 0.1)

    def _update_self_confidence(self, context: Dict[str, Any]):
        """
        Update self-confidence based on recent experiences and context
        """
        # Get recent experiences
        recent_experiences = [exp for exp in self.experience_memory 
                              if (datetime.datetime.now() - exp['timestamp']).total_seconds() < 3600]  # Last hour
        
        if not recent_experiences:
            # Gradually return to baseline if no recent experiences
            baseline_pull = 0.01  # Slow return to baseline
            if self.state.self_confidence > self.self_esteem_baseline:
                self.state.self_confidence -= baseline_pull
            elif self.state.self_confidence < self.self_esteem_baseline:
                self.state.self_confidence += baseline_pull
        else:
            # Calculate average success rate in recent experiences
            successes = [exp for exp in recent_experiences if exp['experience'].get('outcome') == 'success']
            success_rate = len(successes) / len(recent_experiences) if recent_experiences else 0
            
            # Update confidence based on success rate
            confidence_delta = (success_rate - 0.5) * 0.2  # Scale by 0.2
            self.state.self_confidence = max(0.1, min(0.9, self.state.self_confidence + confidence_delta))

    def _adjust_self_complexity(self, context: Dict[str, Any]):
        """
        Adjust self-complexity based on reflection depth and cognitive capacity
        """
        # Increase complexity during deep reflection
        if context.get('reflection_depth', 0) > 0.7:
            self.state.self_complexity = min(8, self.state.self_complexity + 0.1)
        # Decrease complexity when cognitive load is high
        elif context.get('cognitive_load', 0) > 0.8:
            self.state.self_complexity = max(2, self.state.self_complexity - 0.1)

    def _update_stability(self, context: Dict[str, Any]):
        """
        Update self-concept stability based on consistency of experiences
        """
        # Look at recent experiences for consistency
        recent_experiences = [exp for exp in self.experience_memory 
                              if (datetime.datetime.now() - exp['timestamp']).total_seconds() < 7200]  # Last 2 hours
        
        if len(recent_experiences) > 5:
            # Calculate consistency of impacts
            impacts = [exp['impact'] for exp in recent_experiences]
            avg_impact = sum(impacts) / len(impacts)
            variance = sum((imp - avg_impact) ** 2 for imp in impacts) / len(impacts) if impacts else 0
            
            # Higher variance means less stability
            stability_factor = max(0.3, 1.0 - variance)
            self.self_concept_stability = stability_factor

    def get_self_aspect(self, aspect: SelfModelAspect) -> float:
        """
        Get the current value for a specific self-aspect
        """
        return self.self_aspects.get(aspect, 0.5)

    def get_self_overview(self) -> Dict[str, float]:
        """
        Get an overview of all self-aspects
        """
        return {aspect.value: value for aspect, value in self.self_aspects.items()}

    def reflect_on_self(self, depth: float = 1.0) -> Dict[str, Any]:
        """
        Perform self-reflection to examine self-model
        """
        # Update dynamics with current state
        self.update_self_model_dynamics({})
        
        # Prepare reflection results
        reflection = {
            'self_confidence': self.state.self_confidence,
            'self_complexity': self.state.self_complexity,
            'introspection_depth': self.state.introspection_depth,
            'self_consistency': self.state.self_consistency,
            'self_aspects': self.get_self_overview(),
            'recent_experiences_count': len(self.experience_memory),
            'feedback_integration_rate': self.state.feedback_integration_rate,
            'self_verification_tendency': self.state.self_verification_tendency,
            'self_improvement_orientation': self.state.self_improvement_orientation
        }
        
        # If requested deep reflection, include more details
        if depth > 0.7:
            reflection['experience_analysis'] = self._analyze_experiences()
            reflection['feedback_analysis'] = self._analyze_feedback()
            reflection['consistency_analysis'] = self._analyze_consistency()
            reflection['change_trends'] = self._identify_change_trends()
        
        return reflection

    def _analyze_experiences(self) -> Dict[str, Any]:
        """
        Analyze recent experiences affecting self-view
        """
        if not self.experience_memory:
            return {'total_experiences': 0}
        
        # Analyze recent experiences (last 24 hours)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        recent_experiences = [exp for exp in self.experience_memory 
                              if exp['timestamp'] > cutoff_time]
        
        if not recent_experiences:
            return {'total_experiences': len(self.experience_memory)}
        
        # Calculate statistics
        avg_impact = sum(exp['impact'] for exp in recent_experiences) / len(recent_experiences)
        positive_count = sum(1 for exp in recent_experiences if exp['impact'] > 0)
        negative_count = sum(1 for exp in recent_experiences if exp['impact'] < 0)
        
        # Determine most affected aspects
        aspect_counts = {}
        for exp in recent_experiences:
            for aspect in exp['affected_aspects']:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        
        return {
            'total_experiences': len(recent_experiences),
            'average_impact': avg_impact,
            'positive_experiences': positive_count,
            'negative_experiences': negative_count,
            'most_affected_aspects': sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def _analyze_feedback(self) -> Dict[str, Any]:
        """
        Analyze received feedback
        """
        if not self.feedback_history:
            return {'total_feedback': 0}
        
        # Analyze recent feedback (last 24 hours)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        recent_feedback = [fb for fb in self.feedback_history 
                          if fb['timestamp'] > cutoff_time and 'state_snapshot' in fb]
        
        return {
            'total_recent_feedback': len(recent_feedback),
            'total_feedback': len(self.feedback_history),
            'feedback_integration_rate': self.state.feedback_integration_rate
        }

    def _analyze_consistency(self) -> Dict[str, Any]:
        """
        Analyze consistency of self-view over time
        """
        # Calculate variance in self-aspect ratings over time
        if len(self.feedback_history) < 10:
            return {'consistency_score': 0.5, 'data_points': len(self.feedback_history)}
        
        # Look at the last 10 data points
        recent_snapshots = self.feedback_history[-10:]
        
        # Calculate variance for each aspect
        aspects_var = {}
        for aspect in SelfModelAspect:
            values = [snap['state_snapshot']['self_aspects'].get(aspect.value, 0.5) 
                     for snap in recent_snapshots if 'self_aspects' in snap.get('state_snapshot', {})]
            
            if len(values) > 1:
                avg = sum(values) / len(values)
                variance = sum((v - avg) ** 2 for v in values) / len(values)
                aspects_var[aspect.value] = variance
        
        avg_variance = sum(aspects_var.values()) / len(aspects_var) if aspects_var else 0.1
        consistency_score = max(0.0, 1.0 - avg_variance * 5)  # Convert variance to consistency
        
        return {
            'consistency_score': consistency_score,
            'aspect_variances': aspects_var,
            'data_points': len(recent_snapshots)
        }

    def _identify_change_trends(self) -> Dict[str, Any]:
        """
        Identify trends in self-model changes
        """
        if len(self.experience_memory) < 5:
            return {'trend_strength': 0.0, 'primary_trend': 'stable'}
        
        # Look at the last 10 experiences
        recent_experiences = self.experience_memory[-10:]
        
        # Calculate trend in impacts
        impacts = [exp['impact'] for exp in recent_experiences]
        if len(impacts) > 1:
            # Simple linear trend estimation
            n = len(impacts)
            x = list(range(n))
            y = impacts
            
            # Calculate slope (trend)
            avg_x = sum(x) / n
            avg_y = sum(y) / n
            numerator = sum((x[i] - avg_x) * (y[i] - avg_y) for i in range(n))
            denominator = sum((x[i] - avg_x) ** 2 for i in range(n))
            
            if denominator != 0:
                slope = numerator / denominator
                trend_strength = abs(slope)
                
                if slope > 0.01:
                    primary_trend = 'improving'
                elif slope < -0.01:
                    primary_trend = 'declining'
                else:
                    primary_trend = 'stable'
            else:
                trend_strength = 0.0
                primary_trend = 'stable'
        else:
            trend_strength = 0.0
            primary_trend = 'stable'
        
        return {
            'trend_strength': trend_strength,
            'primary_trend': primary_trend,
            'recent_impact_avg': sum(impacts) / len(impacts) if impacts else 0
        }

    def _get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current state for history
        """
        return {
            'self_confidence': self.state.self_confidence,
            'self_complexity': self.state.self_complexity,
            'introspection_depth': self.state.introspection_depth,
            'self_consistency': self.state.self_consistency,
            'self_aspects': self.get_self_overview(),
            'self_concept_stability': self.self_concept_stability,
            'change_sensitivity': self.change_sensitivity
        }

    def get_self_model_metrics(self) -> Dict[str, Any]:
        """
        Get current self-model metrics for monitoring and adjustment
        """
        return {
            'self_confidence': self.state.self_confidence,
            'self_complexity': self.state.self_complexity,
            'introspection_depth': self.state.introspection_depth,
            'self_consistency': self.state.self_consistency,
            'feedback_integration_rate': self.state.feedback_integration_rate,
            'self_verification_tendency': self.state.self_verification_tendency,
            'self_improvement_orientation': self.state.self_improvement_orientation,
            'self_concept_stability': self.self_concept_stability,
            'total_experiences': len(self.experience_memory),
            'total_feedback_interactions': len(self.feedback_history),
            'self_aspects_balance': self._calculate_aspects_balance()
        }

    def _calculate_aspects_balance(self) -> float:
        """
        Calculate how balanced the different self-aspects are (0 = unbalanced, 1 = balanced)
        """
        values = list(self.self_aspects.values())
        if not values:
            return 1.0
            
        # Calculate variance in aspect values
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        
        # Convert variance to balance score (lower variance = higher balance)
        balance_score = max(0.0, 1.0 - variance)
        return balance_score
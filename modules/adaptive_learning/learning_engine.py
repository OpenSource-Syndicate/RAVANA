"""
Adaptive Learning Engine for AGI System
Analyzes past decisions and outcomes to improve future performance.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from sqlmodel import Session, select
from database.models import ActionLog, DecisionLog, MoodLog

logger = logging.getLogger(__name__)

class AdaptiveLearningEngine:
    """Engine for learning from past decisions and adapting strategies."""
    
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.engine = agi_system.engine
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.decision_history = deque(maxlen=1000)  # Keep last 1000 decisions
        self.learning_insights = []
        self.adaptation_strategies = {}
        
    async def analyze_decision_patterns(self, days_back: int = 7) -> Dict[str, Any]:
        """Analyze patterns in recent decisions and their outcomes."""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
            
            with Session(self.engine) as session:
                # Get recent action logs
                action_stmt = select(ActionLog).where(
                    ActionLog.timestamp >= cutoff_date
                ).order_by(ActionLog.timestamp.desc())
                
                actions = session.exec(action_stmt).all()
                
                # Get recent decision logs
                decision_stmt = select(DecisionLog).where(
                    DecisionLog.timestamp >= cutoff_date
                ).order_by(DecisionLog.timestamp.desc())
                
                decisions = session.exec(decision_stmt).all()
            
            # Analyze success/failure patterns
            success_count = 0
            failure_count = 0
            action_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
            
            for action in actions:
                action_success_rates[action.action_name]['total'] += 1
                if action.status == 'success':
                    success_count += 1
                    action_success_rates[action.action_name]['success'] += 1
                else:
                    failure_count += 1
            
            # Calculate success rates
            overall_success_rate = success_count / (success_count + failure_count) if (success_count + failure_count) > 0 else 0
            
            action_rates = {}
            for action_name, stats in action_success_rates.items():
                action_rates[action_name] = {
                    'success_rate': stats['success'] / stats['total'] if stats['total'] > 0 else 0,
                    'total_attempts': stats['total'],
                    'successes': stats['success']
                }
            
            analysis = {
                'period_days': days_back,
                'total_actions': len(actions),
                'total_decisions': len(decisions),
                'overall_success_rate': overall_success_rate,
                'action_success_rates': action_rates,
                'top_performing_actions': sorted(
                    action_rates.items(), 
                    key=lambda x: x[1]['success_rate'], 
                    reverse=True
                )[:5],
                'underperforming_actions': sorted(
                    action_rates.items(), 
                    key=lambda x: x[1]['success_rate']
                )[:5]
            }
            
            logger.info(f"Analyzed {len(actions)} actions over {days_back} days. Success rate: {overall_success_rate:.2%}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze decision patterns: {e}")
            return {'error': str(e)}
    
    async def identify_success_factors(self) -> List[Dict[str, Any]]:
        """Identify factors that contribute to successful outcomes."""
        try:
            analysis = await self.analyze_decision_patterns()
            if 'error' in analysis:
                return []
            
            success_factors = []
            
            # Factor 1: High-performing actions
            top_actions = analysis.get('top_performing_actions', [])
            if top_actions:
                success_factors.append({
                    'factor': 'high_performing_actions',
                    'description': 'Actions with consistently high success rates',
                    'actions': [action[0] for action in top_actions if action[1]['success_rate'] > 0.8],
                    'recommendation': 'Prioritize these actions when possible'
                })
            
            # Factor 2: Overall success rate trends
            if analysis.get('overall_success_rate', 0) > 0.7:
                success_factors.append({
                    'factor': 'high_overall_performance',
                    'description': f"Overall success rate is {analysis['overall_success_rate']:.1%}",
                    'recommendation': 'Current strategy is working well, maintain approach'
                })
            elif analysis.get('overall_success_rate', 0) < 0.5:
                success_factors.append({
                    'factor': 'low_overall_performance',
                    'description': f"Overall success rate is {analysis['overall_success_rate']:.1%}",
                    'recommendation': 'Strategy needs adjustment, consider alternative approaches'
                })
            
            # Factor 3: Action diversity
            action_count = len(analysis.get('action_success_rates', {}))
            if action_count > 10:
                success_factors.append({
                    'factor': 'high_action_diversity',
                    'description': f'Using {action_count} different action types',
                    'recommendation': 'Good action diversity, continue exploring different approaches'
                })
            
            self.learning_insights.extend(success_factors)
            return success_factors
            
        except Exception as e:
            logger.error(f"Failed to identify success factors: {e}")
            return []
    
    async def generate_adaptation_strategies(self) -> Dict[str, Any]:
        """Generate strategies to improve future performance."""
        try:
            success_factors = await self.identify_success_factors()
            analysis = await self.analyze_decision_patterns()
            
            strategies = {}
            
            # Strategy 1: Action prioritization
            if 'action_success_rates' in analysis:
                high_success_actions = [
                    action for action, stats in analysis['action_success_rates'].items()
                    if stats['success_rate'] > 0.8 and stats['total_attempts'] >= 3
                ]
                
                low_success_actions = [
                    action for action, stats in analysis['action_success_rates'].items()
                    if stats['success_rate'] < 0.3 and stats['total_attempts'] >= 3
                ]
                
                strategies['action_prioritization'] = {
                    'prefer_actions': high_success_actions,
                    'avoid_actions': low_success_actions,
                    'description': 'Prioritize high-success actions, be cautious with low-success ones'
                }
            
            # Strategy 2: Confidence adjustment
            overall_rate = analysis.get('overall_success_rate', 0.5)
            if overall_rate > 0.8:
                strategies['confidence_adjustment'] = {
                    'confidence_modifier': 1.1,
                    'description': 'High success rate, increase confidence in decisions'
                }
            elif overall_rate < 0.4:
                strategies['confidence_adjustment'] = {
                    'confidence_modifier': 0.8,
                    'description': 'Low success rate, be more cautious in decisions'
                }
            
            # Strategy 3: Exploration vs exploitation
            action_diversity = len(analysis.get('action_success_rates', {}))
            if action_diversity < 5:
                strategies['exploration_strategy'] = {
                    'exploration_bonus': 0.2,
                    'description': 'Low action diversity, encourage exploration of new actions'
                }
            elif action_diversity > 15:
                strategies['exploitation_strategy'] = {
                    'exploitation_bonus': 0.1,
                    'description': 'High action diversity, focus on exploiting known good actions'
                }
            
            # Strategy 4: Context-aware adaptations
            strategies['context_awareness'] = {
                'mood_sensitivity': 0.1,
                'memory_weight': 0.15,
                'description': 'Adjust decisions based on mood and memory context'
            }
            
            self.adaptation_strategies.update(strategies)
            logger.info(f"Generated {len(strategies)} adaptation strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"Failed to generate adaptation strategies: {e}")
            return {}
    
    async def apply_learning_to_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned adaptations to influence a decision."""
        try:
            if not self.adaptation_strategies:
                await self.generate_adaptation_strategies()
            
            adaptations = {}
            
            # Apply action prioritization
            if 'action_prioritization' in self.adaptation_strategies:
                strategy = self.adaptation_strategies['action_prioritization']
                adaptations['preferred_actions'] = strategy.get('prefer_actions', [])
                adaptations['avoided_actions'] = strategy.get('avoid_actions', [])
            
            # Apply confidence adjustment
            if 'confidence_adjustment' in self.adaptation_strategies:
                strategy = self.adaptation_strategies['confidence_adjustment']
                adaptations['confidence_modifier'] = strategy.get('confidence_modifier', 1.0)
            
            # Apply exploration/exploitation strategy
            if 'exploration_strategy' in self.adaptation_strategies:
                adaptations['exploration_bonus'] = self.adaptation_strategies['exploration_strategy'].get('exploration_bonus', 0)
            elif 'exploitation_strategy' in self.adaptation_strategies:
                adaptations['exploitation_bonus'] = self.adaptation_strategies['exploitation_strategy'].get('exploitation_bonus', 0)
            
            # Context-aware adaptations
            if 'context_awareness' in self.adaptation_strategies:
                strategy = self.adaptation_strategies['context_awareness']
                mood = decision_context.get('mood', {})
                if mood:
                    # Adjust based on mood
                    mood_score = sum(mood.get(m, 0) for m in ['happy', 'confident', 'curious']) - sum(mood.get(m, 0) for m in ['anxious', 'frustrated'])
                    adaptations['mood_adjustment'] = mood_score * strategy.get('mood_sensitivity', 0.1)
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Failed to apply learning to decision: {e}")
            return {}
    
    async def record_decision_outcome(self, decision: Dict[str, Any], outcome: Any, success: bool):
        """Record the outcome of a decision for future learning."""
        try:
            decision_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': decision.get('action', 'unknown'),
                'params': decision.get('params', {}),
                'confidence': decision.get('confidence', 0.5),
                'outcome': str(outcome)[:500],  # Limit outcome length
                'success': success,
                'mood_context': decision.get('mood_context', {}),
                'memory_context': decision.get('memory_context', [])
            }
            
            self.decision_history.append(decision_record)
            
            # Update success/failure patterns
            action_name = decision.get('action', 'unknown')
            if success:
                self.success_patterns[action_name].append(decision_record)
            else:
                self.failure_patterns[action_name].append(decision_record)
            
            # Limit pattern history
            for patterns in [self.success_patterns, self.failure_patterns]:
                for action_patterns in patterns.values():
                    if len(action_patterns) > 50:  # Keep last 50 patterns per action
                        action_patterns[:] = action_patterns[-50:]
            
            logger.debug(f"Recorded decision outcome: {action_name} -> {'success' if success else 'failure'}")
            
        except Exception as e:
            logger.error(f"Failed to record decision outcome: {e}")
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of current learning state."""
        try:
            analysis = await self.analyze_decision_patterns()
            success_factors = await self.identify_success_factors()
            
            summary = {
                'total_decisions_tracked': len(self.decision_history),
                'success_patterns_count': sum(len(patterns) for patterns in self.success_patterns.values()),
                'failure_patterns_count': sum(len(patterns) for patterns in self.failure_patterns.values()),
                'recent_performance': analysis,
                'success_factors': success_factors,
                'active_strategies': list(self.adaptation_strategies.keys()),
                'learning_insights_count': len(self.learning_insights)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {'error': str(e)}
    
    async def reset_learning_data(self, keep_recent_days: int = 30):
        """Reset learning data, optionally keeping recent data."""
        try:
            if keep_recent_days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=keep_recent_days)
                
                # Filter decision history
                recent_decisions = [
                    d for d in self.decision_history 
                    if datetime.fromisoformat(d['timestamp']) > cutoff_date
                ]
                self.decision_history.clear()
                self.decision_history.extend(recent_decisions)
                
                # Filter patterns
                for action_name in list(self.success_patterns.keys()):
                    recent_patterns = [
                        p for p in self.success_patterns[action_name]
                        if datetime.fromisoformat(p['timestamp']) > cutoff_date
                    ]
                    if recent_patterns:
                        self.success_patterns[action_name] = recent_patterns
                    else:
                        del self.success_patterns[action_name]
                
                for action_name in list(self.failure_patterns.keys()):
                    recent_patterns = [
                        p for p in self.failure_patterns[action_name]
                        if datetime.fromisoformat(p['timestamp']) > cutoff_date
                    ]
                    if recent_patterns:
                        self.failure_patterns[action_name] = recent_patterns
                    else:
                        del self.failure_patterns[action_name]
            else:
                # Complete reset
                self.decision_history.clear()
                self.success_patterns.clear()
                self.failure_patterns.clear()
            
            self.learning_insights.clear()
            self.adaptation_strategies.clear()
            
            logger.info(f"Reset learning data, kept {keep_recent_days} days of recent data")
            
        except Exception as e:
            logger.error(f"Failed to reset learning data: {e}")

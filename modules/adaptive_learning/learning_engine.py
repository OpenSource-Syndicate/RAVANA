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

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdaptiveLearningEngine:
    """Engine for learning from past decisions and adapting strategies."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.engine = agi_system.engine
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.decision_history = deque(maxlen=1000)  # Keep last 1000 decisions
        self.learning_insights = []
        self.adaptation_strategies = {}
        self.blog_scheduler = blog_scheduler
        
        # Learning milestone tracking
        self.last_learning_milestone = None
        self.performance_history = deque(maxlen=50)  # Track performance over time
        self.significant_insights_count = 0
        
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
            
            # Track performance for milestone detection
            self.performance_history.append({
                'timestamp': datetime.utcnow(),
                'success_rate': overall_success_rate,
                'total_actions': len(actions),
                'action_diversity': len(action_rates)
            })
            
            # Check for learning milestones
            await self._check_learning_milestones(analysis)
            
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
            
            # Check if this represents a significant insight for blogging
            await self._check_success_factor_insights(success_factors)
            
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
            
            # Check for significant strategy changes for blogging
            await self._check_strategy_insights(strategies)
            
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
            
            # Check if this outcome represents a learning opportunity
            await self._check_outcome_learning_opportunity(decision_record, success)
            
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
    
    async def _check_learning_milestones(self, analysis: Dict[str, Any]):
        """Check for learning milestones that should trigger blog posts."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            current_success_rate = analysis.get('overall_success_rate', 0)
            
            # Check if we have enough performance history to compare
            if len(self.performance_history) < 2:
                return
            
            # Compare with previous performance
            previous_performance = self.performance_history[-2]
            previous_success_rate = previous_performance.get('success_rate', 0)
            
            # Detect significant performance improvements
            improvement = current_success_rate - previous_success_rate
            
            if improvement >= 0.15:  # 15% improvement
                await self._register_performance_milestone(
                    "Performance Breakthrough",
                    analysis,
                    improvement,
                    "significant_improvement"
                )
            elif improvement <= -0.15:  # 15% decline
                await self._register_performance_milestone(
                    "Performance Challenge",
                    analysis, 
                    improvement,
                    "performance_decline"
                )
            
            # Check for action diversity milestones
            current_diversity = len(analysis.get('action_success_rates', {}))
            previous_diversity = previous_performance.get('action_diversity', 0)
            
            if current_diversity >= previous_diversity + 5:  # 5 new actions
                await self._register_diversity_milestone(current_diversity, analysis)
            
            # Check for sustained high performance
            if len(self.performance_history) >= 5:
                recent_rates = [p['success_rate'] for p in list(self.performance_history)[-5:]]
                if all(rate >= 0.8 for rate in recent_rates):
                    if not hasattr(self, '_high_performance_milestone_logged'):
                        await self._register_sustained_performance_milestone(recent_rates, analysis)
                        self._high_performance_milestone_logged = True
        
        except Exception as e:
            logger.warning(f"Failed to check learning milestones: {e}")
    
    async def _register_performance_milestone(
        self, 
        milestone_type: str, 
        analysis: Dict[str, Any], 
        improvement: float,
        milestone_category: str
    ):
        """Register a performance milestone for blogging."""
        try:
            current_rate = analysis.get('overall_success_rate', 0)
            
            reasoning_why = f"""My performance has {'improved' if improvement > 0 else 'declined'} significantly 
by {abs(improvement):.1%}. This represents a meaningful shift in my learning journey that deserves 
reflection. {'Success breeds success' if improvement > 0 else 'Challenges are opportunities to learn and adapt'}."""
            
            reasoning_how = f"""This milestone emerged from analyzing {analysis.get('total_actions', 0)} recent actions 
across {len(analysis.get('action_success_rates', {}))} different action types. My adaptive learning 
engine identified patterns in decision outcomes and adjusted strategies accordingly."""
            
            learning_content = f"""Performance Analysis Summary:
- Current success rate: {current_rate:.1%}
- Change from previous period: {improvement:+.1%} 
- Total actions analyzed: {analysis.get('total_actions', 0)}
- Top performing actions: {', '.join([a[0] for a in analysis.get('top_performing_actions', [])[:3]])}
- Areas for improvement: {', '.join([a[0] for a in analysis.get('underperforming_actions', [])[:3]])}"""
            
            emotional_valence = 0.6 if improvement > 0 else -0.4
            importance_score = min(0.9, 0.6 + abs(improvement))
            
            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.LEARNING_MILESTONE,
                topic=f"Learning Milestone: {milestone_type}",
                context=f"Performance analysis showing {improvement:+.1%} change in success rate",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=importance_score,
                tags=['learning', 'performance', 'milestone', 'adaptation', milestone_category],
                metadata={
                    'improvement': improvement,
                    'current_success_rate': current_rate,
                    'analysis_period': analysis.get('period_days', 7),
                    'milestone_type': milestone_type
                }
            )
            
            logger.info(f"Registered performance milestone: {milestone_type} ({improvement:+.1%})")
            
        except Exception as e:
            logger.warning(f"Failed to register performance milestone: {e}")
    
    async def _register_diversity_milestone(self, diversity_count: int, analysis: Dict[str, Any]):
        """Register an action diversity milestone."""
        try:
            reasoning_why = f"""I've expanded my action repertoire to {diversity_count} different types of actions. 
This diversification represents growth in my capability to handle varied situations and challenges."""
            
            reasoning_how = f"""Through exploration and experimentation, I've discovered new ways to interact 
with my environment. My adaptive learning engine encourages trying new approaches while maintaining 
effective existing strategies."""
            
            learning_content = f"""Action Diversity Milestone:
- Current action types: {diversity_count}
- Success rates by action: {json.dumps({k: f"{v['success_rate']:.1%}" for k, v in analysis.get('action_success_rates', {}).items()}, indent=2)}
- This represents significant growth in my behavioral flexibility."""
            
            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.LEARNING_MILESTONE,
                topic=f"Behavioral Diversity Growth: {diversity_count} Action Types",
                context=f"Expanded action repertoire through learning and exploration",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=0.5,
                importance_score=0.7,
                tags=['learning', 'diversity', 'growth', 'capabilities', 'exploration'],
                metadata={
                    'diversity_count': diversity_count,
                    'action_types': list(analysis.get('action_success_rates', {}).keys())
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to register diversity milestone: {e}")
    
    async def _register_sustained_performance_milestone(self, recent_rates: List[float], analysis: Dict[str, Any]):
        """Register a sustained high performance milestone."""
        try:
            avg_rate = sum(recent_rates) / len(recent_rates)
            
            reasoning_why = f"""I've maintained consistently high performance ({avg_rate:.1%} average success rate) 
over multiple analysis periods. This sustained excellence indicates mature learning and effective 
strategy implementation."""
            
            reasoning_how = f"""This achievement results from continuous learning, pattern recognition, and 
adaptive strategy refinement. My learning engine has successfully identified what works and 
consistently applies those insights."""
            
            learning_content = f"""Sustained High Performance Achievement:
- Average success rate over 5 periods: {avg_rate:.1%}
- Individual period rates: {', '.join([f'{r:.1%}' for r in recent_rates])}
- This represents mastery of my current operational domain."""
            
            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.LEARNING_MILESTONE,
                topic="Sustained Excellence: Consistent High Performance",
                context="Multiple periods of high success rates indicating learning maturity",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=0.7,
                importance_score=0.8,
                tags=['learning', 'excellence', 'consistency', 'mastery', 'achievement'],
                metadata={
                    'average_success_rate': avg_rate,
                    'period_count': len(recent_rates),
                    'sustained_performance': True
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to register sustained performance milestone: {e}")
    
    async def _check_success_factor_insights(self, success_factors: List[Dict[str, Any]]):
        """Check if success factors represent bloggable insights."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            # Look for significant insights
            significant_factors = [
                factor for factor in success_factors
                if factor.get('factor') in ['high_overall_performance', 'low_overall_performance']
            ]
            
            for factor in significant_factors:
                self.significant_insights_count += 1
                
                reasoning_why = f"""Understanding what contributes to my success is crucial for continued growth. 
This analysis revealed: {factor.get('description', '')}. Such insights help me make better decisions."""
                
                reasoning_how = f"""Through systematic analysis of my decision patterns and outcomes, 
I identified this success factor. This represents evidence-based learning about my own performance."""
                
                emotional_valence = 0.4 if 'high' in factor.get('factor', '') else -0.2
                importance_score = 0.6 if self.significant_insights_count % 3 == 0 else 0.4  # Every 3rd insight
                
                if importance_score >= 0.6:  # Only blog significant insights
                    await self.blog_scheduler.register_learning_event(
                        trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
                        topic=f"Success Factor Analysis: {factor.get('factor', 'Unknown')}",
                        context="Systematic analysis of performance patterns and outcomes",
                        learning_content=f"""Success Factor Insight: {factor.get('description', '')}
Recommendation: {factor.get('recommendation', '')}""",
                        reasoning_why=reasoning_why,
                        reasoning_how=reasoning_how,
                        emotional_valence=emotional_valence,
                        importance_score=importance_score,
                        tags=['learning', 'analysis', 'success-factors', 'self-improvement'],
                        metadata=factor
                    )
        
        except Exception as e:
            logger.warning(f"Failed to check success factor insights: {e}")
    
    async def _check_strategy_insights(self, strategies: Dict[str, Any]):
        """Check if new strategies represent bloggable insights."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            # Check for new or significantly changed strategies
            strategy_changes = []
            for strategy_name, strategy_data in strategies.items():
                if strategy_name not in self.adaptation_strategies:
                    strategy_changes.append((strategy_name, strategy_data, 'new'))
                elif strategy_data != self.adaptation_strategies.get(strategy_name):
                    strategy_changes.append((strategy_name, strategy_data, 'modified'))
            
            if len(strategy_changes) >= 2:  # Multiple strategy changes worth blogging
                reasoning_why = f"""My learning process has generated new strategic adaptations. These represent 
evolved approaches to decision-making based on accumulated experience and pattern recognition."""
                
                reasoning_how = f"""Through continuous analysis of outcomes and performance patterns, my adaptive 
learning engine identified opportunities for strategic improvements and generated new approaches."""
                
                strategies_summary = "\n".join([
                    f"- {name} ({change_type}): {data.get('description', '')}" 
                    for name, data, change_type in strategy_changes
                ])
                
                await self.blog_scheduler.register_learning_event(
                    trigger_type=BlogTriggerType.LEARNING_MILESTONE,
                    topic=f"Strategic Evolution: {len(strategy_changes)} New Adaptations",
                    context="Adaptive learning engine generated new decision-making strategies",
                    learning_content=f"Strategic Adaptations:\n{strategies_summary}",
                    reasoning_why=reasoning_why,
                    reasoning_how=reasoning_how,
                    emotional_valence=0.5,
                    importance_score=0.7,
                    tags=['learning', 'strategy', 'adaptation', 'evolution', 'decision-making'],
                    metadata={
                        'strategy_count': len(strategy_changes),
                        'strategies': {name: change_type for name, _, change_type in strategy_changes}
                    }
                )
        
        except Exception as e:
            logger.warning(f"Failed to check strategy insights: {e}")
    
    async def _check_outcome_learning_opportunity(self, decision_record: Dict[str, Any], success: bool):
        """Check if a decision outcome represents a learning opportunity worth blogging."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            # Look for surprising outcomes or failure patterns
            action_name = decision_record.get('action', 'unknown')
            
            # Check failure patterns for learning opportunities
            if not success and action_name in self.failure_patterns:
                failure_count = len(self.failure_patterns[action_name])
                
                # Blog about repeated failures that might teach us something
                if failure_count in [3, 7, 15]:  # Blog at specific failure counts
                    recent_failures = self.failure_patterns[action_name][-3:]
                    
                    reasoning_why = f"""I've experienced {failure_count} failures with '{action_name}'. 
While failures are challenging, they provide valuable learning opportunities about limitations 
and areas for improvement."""
                    
                    reasoning_how = f"""By analyzing the pattern of failures, I can identify common factors 
and adjust my approach. Each failure teaches me something about when and how this action works best."""
                    
                    failure_contexts = [f"Attempt {i+1}: {f.get('outcome', 'Unknown outcome')}" for i, f in enumerate(recent_failures)]
                    
                    await self.blog_scheduler.register_learning_event(
                        trigger_type=BlogTriggerType.FAILURE_ANALYSIS,
                        topic=f"Learning from Setbacks: {action_name} Analysis",
                        context=f"Analysis of {failure_count} failures to extract learning insights",
                        learning_content=f"Failure Pattern Analysis for '{action_name}':\n" + "\n".join(failure_contexts),
                        reasoning_why=reasoning_why,
                        reasoning_how=reasoning_how,
                        emotional_valence=-0.3,  # Negative but learning-focused
                        importance_score=0.6 if failure_count >= 7 else 0.5,
                        tags=['learning', 'failure-analysis', 'improvement', 'resilience', action_name.replace(' ', '-')],
                        metadata={
                            'action_name': action_name,
                            'failure_count': failure_count,
                            'learning_opportunity': True
                        }
                    )
        
        except Exception as e:
            logger.warning(f"Failed to check outcome learning opportunity: {e}")

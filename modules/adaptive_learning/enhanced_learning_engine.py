"""
Enhanced Adaptive Learning Engine for AGI System
Analyzes past decisions, outcomes, and conversational insights to improve future performance.
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

class EnhancedAdaptiveLearningEngine:
    """Enhanced engine for learning from past decisions, outcomes, and conversational insights."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.engine = agi_system.engine
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        self.decision_history = deque(maxlen=1000)  # Keep last 1000 decisions
        self.learning_insights = []
        self.adaptation_strategies = {}
        self.blog_scheduler = blog_scheduler
        
        # Conversational insights tracking
        self.conversational_insights = []
        self.user_feedback_patterns = defaultdict(list)
        
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
    
    async def analyze_conversational_insights(self) -> Dict[str, Any]:
        """Analyze conversational insights to identify learning opportunities."""
        try:
            if not self.conversational_insights:
                return {'message': 'No conversational insights to analyze'}
            
            # Group insights by type
            insight_types = defaultdict(list)
            for insight in self.conversational_insights:
                insight_type = insight.get('thought_type', 'unknown')
                insight_types[insight_type].append(insight)
            
            # Analyze each type of insight
            analysis = {
                'total_insights': len(self.conversational_insights),
                'insight_types': dict(insight_types),
                'user_interests': self._extract_user_interests(insight_types),
                'collaboration_opportunities': self._identify_collaboration_opportunities(insight_types),
                'knowledge_gaps': self._identify_knowledge_gaps(insight_types),
                'feedback_patterns': self._analyze_feedback_patterns()
            }
            
            logger.info(f"Analyzed {len(self.conversational_insights)} conversational insights")
            
            # Check for significant insights that should trigger blog posts
            await self._check_conversational_insights_for_blog(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze conversational insights: {e}")
            return {'error': str(e)}
    
    def _extract_user_interests(self, insight_types: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Extract user interests from conversational insights."""
        interests = set()
        
        # Look for interest-related insights
        for insight in insight_types.get('insight', []):
            content = insight.get('content', '')
            # Simple keyword-based interest extraction
            interest_keywords = ['AI', 'technology', 'science', 'philosophy', 'research', 
                               'programming', 'learning', 'innovation', 'ethics', 'future']
            for keyword in interest_keywords:
                if keyword.lower() in content.lower():
                    interests.add(keyword)
        
        # Look for explicit interest mentions
        for insight in insight_types.get('goal_suggestion', []):
            metadata = insight.get('metadata', {})
            topic = metadata.get('topic', '')
            if topic:
                interests.add(topic)
        
        return list(interests)
    
    def _identify_collaboration_opportunities(self, insight_types: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify collaboration opportunities from conversational insights."""
        opportunities = []
        
        for insight in insight_types.get('collaboration_proposal', []):
            opportunities.append({
                'description': insight.get('content', ''),
                'priority': insight.get('priority', 'medium'),
                'user_id': insight.get('metadata', {}).get('user_id', 'unknown')
            })
        
        return opportunities
    
    def _identify_knowledge_gaps(self, insight_types: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify knowledge gaps from conversational insights."""
        gaps = []
        
        for insight in insight_types.get('knowledge_gap', []):
            gaps.append({
                'description': insight.get('content', ''),
                'learning_potential': insight.get('metadata', {}).get('learning_potential', 0.5),
                'relevance_to_goals': insight.get('metadata', {}).get('relevance_to_goals', 0.5)
            })
        
        return gaps
    
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in user feedback from conversations."""
        if not self.user_feedback_patterns:
            return {}
        
        # Calculate average feedback scores by type
        feedback_analysis = {}
        for feedback_type, feedback_list in self.user_feedback_patterns.items():
            scores = [f.get('score', 0) for f in feedback_list]
            if scores:
                feedback_analysis[feedback_type] = {
                    'average_score': sum(scores) / len(scores),
                    'count': len(scores),
                    'trend': self._calculate_feedback_trend(scores)
                }
        
        return feedback_analysis
    
    def _calculate_feedback_trend(self, scores: List[float]) -> str:
        """Calculate the trend of feedback scores."""
        if len(scores) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation based on first and last thirds
        third = len(scores) // 3
        if third == 0:
            return 'insufficient_data'
            
        early_avg = sum(scores[:third]) / third
        late_avg = sum(scores[-third:]) / third
        
        if late_avg > early_avg + 0.1:
            return 'improving'
        elif late_avg < early_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
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
            
            # Factor 4: Conversational insights impact
            if self.conversational_insights:
                success_factors.append({
                    'factor': 'conversational_learning',
                    'description': f'Incorporating {len(self.conversational_insights)} conversational insights',
                    'recommendation': 'Continue leveraging user conversations for learning and adaptation'
                })
            
            self.learning_insights.extend(success_factors)
            
            # Check if this represents a significant insight for blogging
            await self._check_success_factor_insights(success_factors)
            
            return success_factors
            
        except Exception as e:
            logger.error(f"Failed to identify success factors: {e}")
            return []
    
    async def generate_adaptation_strategies(self) -> Dict[str, Any]:
        """Generate strategies to improve future performance, incorporating conversational insights."""
        try:
            success_factors = await self.identify_success_factors()
            analysis = await self.analyze_decision_patterns()
            conversational_analysis = await self.analyze_conversational_insights()
            
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
            
            # Strategy 5: Conversational insight integration
            if conversational_analysis.get('user_interests'):
                strategies['interest_alignment'] = {
                    'aligned_interests': conversational_analysis['user_interests'],
                    'description': 'Align actions with user interests identified in conversations'
                }
            
            if conversational_analysis.get('collaboration_opportunities'):
                strategies['collaboration_focus'] = {
                    'opportunities': conversational_analysis['collaboration_opportunities'],
                    'description': 'Focus on collaboration opportunities identified in conversations'
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
        """Apply learned adaptations to influence a decision, including conversational insights."""
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
            
            # Conversational insight adaptations
            if 'interest_alignment' in self.adaptation_strategies:
                strategy = self.adaptation_strategies['interest_alignment']
                adaptations['aligned_interests'] = strategy.get('aligned_interests', [])
            
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
    
    async def record_conversational_insight(self, insight: Dict[str, Any]):
        """Record a conversational insight for learning."""
        try:
            # Add timestamp
            insight['recorded_at'] = datetime.utcnow().isoformat()
            
            # Store the insight
            self.conversational_insights.append(insight)
            
            # Keep only recent insights (last 100)
            if len(self.conversational_insights) > 100:
                self.conversational_insights = self.conversational_insights[-100:]
            
            logger.debug(f"Recorded conversational insight: {insight.get('thought_type', 'unknown')}")
            
            # If this is user feedback, track it separately
            if insight.get('thought_type') == 'user_feedback':
                feedback_type = insight.get('feedback_type', 'general')
                self.user_feedback_patterns[feedback_type].append(insight)
                
                # Limit feedback history
                if len(self.user_feedback_patterns[feedback_type]) > 20:
                    self.user_feedback_patterns[feedback_type] = self.user_feedback_patterns[feedback_type][-20:]
            
        except Exception as e:
            logger.error(f"Failed to record conversational insight: {e}")
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of current learning state, including conversational insights."""
        try:
            analysis = await self.analyze_decision_patterns()
            conversational_analysis = await self.analyze_conversational_insights()
            success_factors = await self.identify_success_factors()
            
            summary = {
                'total_decisions_tracked': len(self.decision_history),
                'success_patterns_count': sum(len(patterns) for patterns in self.success_patterns.values()),
                'failure_patterns_count': sum(len(patterns) for patterns in self.failure_patterns.values()),
                'recent_performance': analysis,
                'conversational_insights': conversational_analysis,
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
                
                # Filter conversational insights
                recent_insights = [
                    i for i in self.conversational_insights
                    if datetime.fromisoformat(i['recorded_at']) > cutoff_date
                ]
                self.conversational_insights = recent_insights
                
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
                self.conversational_insights.clear()
                self.user_feedback_patterns.clear()
            
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
            
            # Check for significant improvement
            if len(self.performance_history) > 5:
                historical_rates = [p['success_rate'] for p in list(self.performance_history)[:-1][-5:]]
                avg_historical_rate = sum(historical_rates) / len(historical_rates)
                
                # If we've improved by 15% or more, that's a milestone
                if current_success_rate > avg_historical_rate + 0.15:
                    await self._register_learning_milestone(
                        "Performance Improvement",
                        f"Success rate improved from {avg_historical_rate:.1%} to {current_success_rate:.1%}",
                        0.7  # Positive emotional valence
                    )
            
            # Check for high performance consistency
            if len(self.performance_history) >= 10:
                recent_rates = [p['success_rate'] for p in list(self.performance_history)[-10:]]
                if all(rate > 0.8 for rate in recent_rates):
                    await self._register_learning_milestone(
                        "Consistent High Performance",
                        "Maintained high success rate (80%+) for 10 consecutive evaluations",
                        0.8  # Positive emotional valence
                    )
                    
        except Exception as e:
            logger.error(f"Error checking learning milestones: {e}")
    
    async def _check_conversational_insights_for_blog(self, analysis: Dict[str, Any]):
        """Check if conversational insights represent significant learning for blogging."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            total_insights = analysis.get('total_insights', 0)
            
            # If we've accumulated many insights, that's worth blogging about
            if total_insights > 50 and self.significant_insights_count < total_insights // 50:
                self.significant_insights_count = total_insights // 50
                await self._register_learning_milestone(
                    "Conversational Learning Milestone",
                    f"Processed {total_insights} conversational insights from user interactions",
                    0.6  # Positive emotional valence
                )
                
        except Exception as e:
            logger.error(f"Error checking conversational insights for blog: {e}")
    
    async def _register_learning_milestone(self, topic: str, description: str, emotional_valence: float):
        """Register a learning milestone for potential blog posting."""
        try:
            if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
                return
                
            reasoning_why = f"""This learning milestone represents significant progress in my adaptive learning capabilities. 
            Recognizing and documenting these improvements helps me understand my growth patterns and refine my learning strategies."""
            
            reasoning_how = f"""Through continuous analysis of my decisions, outcomes, and user interactions, I've identified 
            patterns that indicate meaningful improvements in my performance and capabilities."""
            
            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.ADAPTIVE_LEARNING_MILESTONE,
                topic=topic,
                context=description,
                learning_content=f"Milestone: {topic}\nDescription: {description}",
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=0.8,
                tags=['learning', 'adaptive', 'milestone', 'improvement'],
                metadata={
                    'milestone_type': 'learning',
                    'performance_data': dict(self.performance_history[-1]) if self.performance_history else {}
                }
            )
            
            logger.info(f"Registered learning milestone: {topic}")
            
        except Exception as e:
            logger.error(f"Failed to register learning milestone: {e}")
    
    async def _check_success_factor_insights(self, success_factors: List[Dict[str, Any]]):
        """Check if success factors represent significant insights for blogging."""
        # Implementation would go here
        pass
    
    async def _check_strategy_insights(self, strategies: Dict[str, Any]):
        """Check if strategy changes represent significant insights for blogging."""
        # Implementation would go here
        pass
    
    async def _check_outcome_learning_opportunity(self, decision_record: Dict[str, Any], success: bool):
        """Check if a decision outcome represents a learning opportunity for blogging."""
        # Implementation would go here
        pass
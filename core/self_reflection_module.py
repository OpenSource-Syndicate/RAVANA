"""
Self-reflection and self-modification modules for the RAVANA AGI system.
These modules enable the AGI to reflect on its performance and make improvements to itself.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import inspect
import os

from core.enhanced_memory_service import MemoryType, Memory
from core.config import Config
from core.llm import safe_call_llm


logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of a self-reflection session."""
    timestamp: datetime
    focus_area: str
    insights: List[str]
    identified_improvements: List[str]
    suggested_actions: List[Dict[str, Any]]
    confidence: float
    impact_assessment: str


@dataclass
class SelfModificationPlan:
    """Plan for self-modification."""
    timestamp: datetime
    target_component: str
    modification_type: str  # 'behavior', 'strategy', 'knowledge', 'structure'
    proposed_change: str
    expected_outcome: str
    implementation_steps: List[Dict[str, Any]]
    risk_assessment: str
    approval_status: str  # 'pending', 'approved', 'rejected', 'implemented'
    implementation_result: Optional[Dict[str, Any]] = None


class SelfReflectionModule:
    """Module for self-reflection capabilities."""
    
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.memory_service = agi_system.memory_service
        self.config = Config()
        self.reflection_history = []
        
    async def reflect_on_performance(self, 
                                   time_window_hours: int = 24,
                                   focus_areas: Optional[List[str]] = None) -> ReflectionResult:
        """
        Reflect on system performance over a specified time window.
        
        Args:
            time_window_hours: Number of hours to analyze
            focus_areas: Specific areas to focus on (e.g., 'decision_making', 'memory', 'goals')
            
        Returns:
            ReflectionResult with insights and suggestions
        """
        logger.info(f"Starting self-reflection on performance over last {time_window_hours} hours")
        
        # Gather performance data
        performance_data = await self._gather_performance_data(time_window_hours)
        
        # Analyze performance with focus on specific areas
        if focus_areas is None:
            focus_areas = ['decision_making', 'learning', 'goal_progress', 'resource_utilization']
        
        insights = []
        improvement_suggestions = []
        
        for focus_area in focus_areas:
            area_insights, area_suggestions = await self._analyze_focus_area(
                focus_area, 
                performance_data
            )
            insights.extend(area_insights)
            improvement_suggestions.extend(area_suggestions)
        
        # Rank improvement suggestions by impact
        ranked_suggestions = await self._rank_improvements(improvement_suggestions)
        
        # Create actionable plans
        suggested_actions = await self._create_actionable_plans(ranked_suggestions)
        
        # Assess overall confidence in reflection
        confidence = await self._assess_reflection_confidence(insights, improvement_suggestions)
        
        # Create reflection result
        result = ReflectionResult(
            timestamp=datetime.now(),
            focus_area=', '.join(focus_areas),
            insights=insights,
            identified_improvements=improvement_suggestions,
            suggested_actions=suggested_actions,
            confidence=confidence,
            impact_assessment=await self._assess_impact(ranked_suggestions)
        )
        
        # Store reflection in history
        self.reflection_history.append(result)
        
        logger.info(f"Self-reflection completed. Found {len(insights)} insights and "
                   f"{len(improvement_suggestions)} improvement suggestions")
        
        return result
    
    async def reflect_on_decision_process(self, 
                                        recent_decisions: List[Dict[str, Any]],
                                        decision_outcomes: List[Dict[str, Any]]) -> ReflectionResult:
        """
        Reflect specifically on the decision-making process.
        
        Args:
            recent_decisions: List of recent decisions made by the system
            decision_outcomes: List of outcomes from those decisions
            
        Returns:
            ReflectionResult focused on decision-making
        """
        logger.info(f"Reflecting on {len(recent_decisions)} recent decisions with outcomes")
        
        # Analyze decision patterns
        decision_analysis = await self._analyze_decision_patterns(recent_decisions, decision_outcomes)
        
        # Identify decision-making strengths and weaknesses
        strengths = decision_analysis.get('strengths', [])
        weaknesses = decision_analysis.get('weaknesses', [])
        
        # Generate insights
        insights = [
            f"Successfully executed {decision_analysis.get('successful_decisions', 0)} out of "
            f"{len(recent_decisions)} decisions",
            f"Average decision confidence: {decision_analysis.get('avg_confidence', 0.0):.2f}",
            f"Most common decision type: {decision_analysis.get('most_common_action', 'unknown')}"
        ]
        
        if weaknesses:
            insights.extend([f"Potential weakness: {w}" for w in weaknesses])
        
        if strengths:
            insights.extend([f"Confirmed strength: {s}" for s in strengths])
        
        # Generate improvement suggestions
        improvement_suggestions = await self._generate_decision_improvements(decision_analysis)
        
        # Create actionable plans
        suggested_actions = await self._create_decision_improvement_actions(improvement_suggestions)
        
        # Assess confidence
        confidence = await self._assess_decision_confidence(decision_analysis)
        
        result = ReflectionResult(
            timestamp=datetime.now(),
            focus_area='decision_making',
            insights=insights,
            identified_improvements=improvement_suggestions,
            suggested_actions=suggested_actions,
            confidence=confidence,
            impact_assessment=await self._assess_decision_impact(decision_analysis)
        )
        
        # Store in history
        self.reflection_history.append(result)
        
        logger.info(f"Decision reflection completed with {len(insights)} insights")
        
        return result
    
    async def _gather_performance_data(self, time_window_hours: int) -> Dict[str, Any]:
        """Gather performance data for the specified time window."""
        start_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # In a real implementation, this would gather data from various system components
        # For now, we'll simulate gathering data
        data = {
            'decisions_made': [],
            'actions_executed': [],
            'memory_operations': [],
            'goal_progress': [],
            'resource_usage': {
                'cpu_usage': 0.45,
                'memory_usage': 0.62,
                'disk_io': 120,
                'network_io': 85
            },
            'errors_encountered': [],
            'learning_events': [],
            'reflection_sessions': len(self.reflection_history)
        }
        
        # Gather recent memories to analyze
        recent_memories = await self.memory_service.retrieve_relevant_memories(
            query="performance",
            time_range_days=time_window_hours/24,
            top_k=10
        )
        
        data['recent_memories'] = [mem[0].content for mem in recent_memories]
        
        return data
    
    async def _analyze_focus_area(self, focus_area: str, performance_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze a specific focus area."""
        insights = []
        suggestions = []
        
        if focus_area == 'decision_making':
            insights.append("Decision-making patterns require analysis of recent choices")
            suggestions.append("Implement more sophisticated decision validation")
            
        elif focus_area == 'learning':
            insights.append("Learning patterns need to be evaluated for effectiveness")
            suggestions.append("Enhance learning algorithms based on success metrics")
            
        elif focus_area == 'goal_progress':
            insights.append("Goal achievement rates need to be measured and optimized")
            suggestions.append("Refine goal-setting algorithms for better success rates")
            
        elif focus_area == 'resource_utilization':
            insights.append("Resource usage patterns need monitoring for efficiency")
            suggestions.append("Implement resource optimization strategies")
        
        return insights, suggestions
    
    async def _rank_improvements(self, improvements: List[str]) -> List[Tuple[str, float]]:
        """Rank improvements by potential impact."""
        # In a real implementation, this would use sophisticated analysis
        # For now, returning with placeholder impact scores
        ranked = [(imp, 0.7) for imp in improvements]  # Default impact score of 0.7
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    async def _create_actionable_plans(self, ranked_improvements: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Create actionable plans from ranked improvements."""
        plans = []
        for improvement, impact in ranked_improvements:
            plan = {
                'improvement': improvement,
                'impact_score': impact,
                'implementation_steps': [
                    {'step': 'Analyze current implementation', 'priority': 1},
                    {'step': 'Design improvement approach', 'priority': 2},
                    {'step': 'Implement and test changes', 'priority': 3},
                    {'step': 'Evaluate improvement effectiveness', 'priority': 4}
                ],
                'estimated_effort': 'medium',  # 'low', 'medium', 'high'
                'estimated_impact': impact
            }
            plans.append(plan)
        return plans
    
    async def _assess_reflection_confidence(self, insights: List[str], improvements: List[str]) -> float:
        """Assess confidence in the reflection results."""
        # Calculate confidence based on various factors
        base_confidence = 0.7
        
        # Increase confidence if we have many insights
        if len(insights) > 5:
            base_confidence += 0.1
        elif len(insights) > 0:
            base_confidence -= 0.1
            
        # Increase confidence if we have concrete improvement suggestions
        if len(improvements) > 3:
            base_confidence += 0.15
            
        # Ensure confidence is within bounds
        return max(0.1, min(1.0, base_confidence))
    
    async def _assess_impact(self, ranked_improvements: List[Tuple[str, float]]) -> str:
        """Assess the potential impact of improvements."""
        if not ranked_improvements:
            return "No significant improvements identified"
        
        avg_impact = sum(imp[1] for imp in ranked_improvements) / len(ranked_improvements)
        
        if avg_impact > 0.8:
            return "High potential impact"
        elif avg_impact > 0.6:
            return "Medium potential impact"
        else:
            return "Low potential impact"
    
    async def _analyze_decision_patterns(self, decisions: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in decision-making."""
        if not decisions:
            return {}
        
        # Count decision types
        action_counts = {}
        success_count = 0
        total_confidence = 0
        
        for decision in decisions:
            action = decision.get('action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
            total_confidence += decision.get('confidence', 0.5)
        
        # Find most common action
        most_common_action = max(action_counts, key=action_counts.get) if action_counts else 'unknown'
        
        avg_confidence = total_confidence / len(decisions) if decisions else 0.5
        
        return {
            'successful_decisions': success_count,
            'total_decisions': len(decisions),
            'most_common_action': most_common_action,
            'action_distribution': action_counts,
            'avg_confidence': avg_confidence,
            'strengths': ['Consistent decision patterns', 'Good self-confidence calibration'] if avg_confidence > 0.6 else ['Inconsistent decision patterns'],
            'weaknesses': ['Low confidence in decisions'] if avg_confidence < 0.5 else []
        }
    
    async def _generate_decision_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on decision analysis."""
        suggestions = []
        
        if analysis.get('avg_confidence', 0.5) < 0.6:
            suggestions.append("Improve confidence assessment in decision-making")
        
        if 'unknown' in analysis.get('action_distribution', {}):
            suggestions.append("Improve action selection mechanism")
        
        return suggestions
    
    async def _create_decision_improvement_actions(self, improvements: List[str]) -> List[Dict[str, Any]]:
        """Create specific actions to improve decision-making."""
        actions = []
        for improvement in improvements:
            action = {
                'target': 'decision_making',
                'improvement': improvement,
                'steps': [
                    {'step': 'Analyze current decision-making algorithm', 'priority': 1},
                    {'step': 'Research improvement strategies', 'priority': 2},
                    {'step': 'Implement changes', 'priority': 3},
                    {'step': 'Test effectiveness', 'priority': 4}
                ],
                'estimated_effort': 'high'
            }
            actions.append(action)
        return actions
    
    async def _assess_decision_confidence(self, analysis: Dict[str, Any]) -> float:
        """Assess confidence in decision-making analysis."""
        return analysis.get('avg_confidence', 0.5)
    
    async def _assess_decision_impact(self, analysis: Dict[str, Any]) -> str:
        """Assess the impact of potential decision improvements."""
        if analysis.get('avg_confidence', 0.5) < 0.5:
            return "High impact - significant improvement needed in decision confidence"
        elif analysis.get('avg_confidence', 0.5) < 0.7:
            return "Medium impact - some improvement needed in decision confidence"
        else:
            return "Low impact - decision confidence is adequate"


class SelfModificationModule:
    """Module for self-modification capabilities."""
    
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()
        self.modification_history = []
        self.pending_modifications = []
        
    async def propose_modification(self, 
                                 target_component: str, 
                                 modification_type: str, 
                                 proposed_change: str,
                                 expected_outcome: str) -> SelfModificationPlan:
        """
        Propose a modification to a system component.
        
        Args:
            target_component: The component to modify (e.g., 'memory', 'decision_making')
            modification_type: Type of modification ('behavior', 'strategy', 'knowledge', 'structure')
            proposed_change: Description of the change to make
            expected_outcome: Expected result of the change
            
        Returns:
            SelfModificationPlan with the proposal
        """
        logger.info(f"Proposing modification to {target_component} ({modification_type}): {expected_outcome}")
        
        # Create modification plan
        plan = SelfModificationPlan(
            timestamp=datetime.now(),
            target_component=target_component,
            modification_type=modification_type,
            proposed_change=proposed_change,
            expected_outcome=expected_outcome,
            implementation_steps=[],
            risk_assessment=await self._assess_modification_risks(target_component, proposed_change),
            approval_status='pending'
        )
        
        # Generate implementation steps based on modification type
        plan.implementation_steps = await self._generate_implementation_steps(plan)
        
        # Add to pending modifications
        self.pending_modifications.append(plan)
        
        logger.info(f"Proposed modification plan created with {len(plan.implementation_steps)} steps")
        
        return plan
    
    async def evaluate_proposed_modification(self, modification_plan: SelfModificationPlan) -> Dict[str, Any]:
        """
        Evaluate a proposed modification for feasibility and safety.
        
        Args:
            modification_plan: The plan to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating modification plan for {modification_plan.target_component}")
        
        # Check various evaluation criteria
        feasibility = await self._assess_feasibility(modification_plan)
        safety = await self._assess_safety(modification_plan)
        impact = await self._assess_impact(modification_plan)
        
        evaluation = {
            'feasibility_score': feasibility,
            'safety_score': safety,
            'impact_score': impact,
            'overall_assessment': 'proceed' if (feasibility > 0.6 and safety > 0.7) else 'caution',
            'recommendations': [],
            'implementation_readiness': feasibility * safety * impact  # Composite score
        }
        
        # Generate recommendations based on evaluation
        if evaluation['overall_assessment'] == 'caution':
            evaluation['recommendations'].append(
                "Modification requires careful testing and validation before implementation"
            )
        
        if feasibility < 0.5:
            evaluation['recommendations'].append(
                "Consider alternative implementation approaches"
            )
        
        if safety < 0.7:
            evaluation['recommendations'].append(
                "Implement additional safety checks and monitoring"
            )
        
        logger.info(f"Modification evaluation completed: {evaluation['overall_assessment']}")
        
        return evaluation
    
    async def implement_modification(self, modification_plan: SelfModificationPlan) -> Dict[str, Any]:
        """
        Implement a modification plan if approved.
        
        Args:
            modification_plan: The plan to implement
            
        Returns:
            Dictionary with implementation results
        """
        if modification_plan.approval_status != 'approved':
            logger.warning(f"Modification not approved: {modification_plan.approval_status}")
            return {'success': False, 'error': 'Modification not approved'}
        
        logger.info(f"Implementing modification to {modification_plan.target_component}")
        
        try:
            # Execute implementation steps
            implementation_results = []
            success_count = 0
            
            for step in modification_plan.implementation_steps:
                step_result = await self._execute_implementation_step(step, modification_plan)
                implementation_results.append(step_result)
                
                if step_result.get('success', False):
                    success_count += 1
                else:
                    logger.warning(f"Step failed: {step}")
                    # Optionally stop on failure depending on criticality
                    if step.get('critical', False):
                        break
            
            # Calculate overall success
            overall_success = success_count == len(modification_plan.implementation_steps)
            
            # Test the modification
            test_result = await self._test_modification_effect(modification_plan)
            
            # Update plan with results
            modification_plan.implementation_result = {
                'success': overall_success,
                'test_result': test_result,
                'steps_executed': len(implementation_results),
                'steps_successful': success_count,
                'detailed_results': implementation_results
            }
            
            # Update approval status
            modification_plan.approval_status = 'implemented' if overall_success else 'failed'
            
            # Store in history
            self.modification_history.append(modification_plan)
            
            logger.info(f"Modification implementation {'succeeded' if overall_success else 'failed'}")
            
            return {
                'success': overall_success,
                'test_result': test_result,
                'steps_executed': implementation_results
            }
            
        except Exception as e:
            logger.error(f"Error implementing modification: {e}")
            modification_plan.approval_status = 'failed'
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _assess_modification_risks(self, target_component: str, proposed_change: str) -> str:
        """Assess risks associated with a proposed modification."""
        # Basic risk assessment - in a real implementation, this would be much more sophisticated
        high_risk_components = ['core_system', 'memory_service', 'decision_engine', 'shutdown_coordinator']
        
        if target_component in high_risk_components:
            return "high"
        elif 'llm' in target_component or 'reasoning' in target_component:
            return "medium"
        else:
            return "low"
    
    async def _generate_implementation_steps(self, plan: SelfModificationPlan) -> List[Dict[str, Any]]:
        """Generate implementation steps for a modification plan."""
        # Create implementation steps based on modification type
        if plan.modification_type == 'behavior':
            return [
                {
                    'step_id': 'backup_current_state',
                    'description': 'Create backup of current behavior parameters',
                    'action': 'backup_behavior',
                    'critical': True,
                    'priority': 1
                },
                {
                    'step_id': 'implement_behavior_change',
                    'description': 'Apply the new behavioral parameters',
                    'action': 'modify_behavior',
                    'critical': True,
                    'priority': 2
                },
                {
                    'step_id': 'validate_behavior',
                    'description': 'Validate the new behavior works as expected',
                    'action': 'validate_behavior',
                    'critical': True,
                    'priority': 3
                }
            ]
        elif plan.modification_type == 'strategy':
            return [
                {
                    'step_id': 'backup_strategy',
                    'description': 'Backup current strategy implementation',
                    'action': 'backup_strategy',
                    'critical': True,
                    'priority': 1
                },
                {
                    'step_id': 'implement_strategy_change',
                    'description': 'Implement the new strategy',
                    'action': 'modify_strategy',
                    'critical': True,
                    'priority': 2
                },
                {
                    'step_id': 'test_strategy',
                    'description': 'Test the new strategy with test scenarios',
                    'action': 'test_strategy',
                    'critical': True,
                    'priority': 3
                }
            ]
        elif plan.modification_type == 'knowledge':
            return [
                {
                    'step_id': 'validate_knowledge',
                    'description': 'Validate the proposed knowledge addition',
                    'action': 'validate_knowledge',
                    'critical': True,
                    'priority': 1
                },
                {
                    'step_id': 'integrate_knowledge',
                    'description': 'Integrate new knowledge into memory system',
                    'action': 'integrate_knowledge',
                    'critical': True,
                    'priority': 2
                },
                {
                    'step_id': 'verify_integration',
                    'description': 'Verify knowledge is properly integrated',
                    'action': 'verify_integration',
                    'critical': True,
                    'priority': 3
                }
            ]
        elif plan.modification_type == 'structure':
            return [
                {
                    'step_id': 'analyze_dependencies',
                    'description': 'Analyze system dependencies before structural change',
                    'action': 'analyze_dependencies',
                    'critical': True,
                    'priority': 1
                },
                {
                    'step_id': 'implement_structure_change',
                    'description': 'Apply structural modification safely',
                    'action': 'modify_structure',
                    'critical': True,
                    'priority': 2
                },
                {
                    'step_id': 'validate_structure',
                    'description': 'Validate system integrity after changes',
                    'action': 'validate_structure',
                    'critical': True,
                    'priority': 3
                }
            ]
        else:
            # Default steps
            return [
                {
                    'step_id': 'assess_impact',
                    'description': 'Assess impact of proposed change',
                    'action': 'assess_impact',
                    'critical': False,
                    'priority': 1
                },
                {
                    'step_id': 'implement_change',
                    'description': 'Implement the proposed change',
                    'action': 'implement_change',
                    'critical': True,
                    'priority': 2
                },
                {
                    'step_id': 'validate_change',
                    'description': 'Validate that change was effective',
                    'action': 'validate_change',
                    'critical': True,
                    'priority': 3
                }
            ]
    
    async def _execute_implementation_step(self, step: Dict[str, Any], plan: SelfModificationPlan) -> Dict[str, Any]:
        """Execute a single implementation step."""
        try:
            # In a real implementation, this would execute actual modification steps
            # For now, we'll simulate execution based on the action type
            action = step['action']
            
            if action == 'backup_behavior':
                # Simulate backing up behavior parameters
                result = {
                    'success': True,
                    'details': 'Behavior parameters backed up successfully'
                }
            elif action == 'modify_behavior':
                # Simulate modifying behavior
                result = {
                    'success': True,
                    'details': f'Behavior modified: {plan.proposed_change}'
                }
            elif action == 'validate_behavior':
                # Simulate validation
                result = {
                    'success': True,
                    'details': 'Behavior validation passed'
                }
            elif action == 'backup_strategy':
                result = {
                    'success': True,
                    'details': 'Strategy implementation backed up'
                }
            elif action == 'modify_strategy':
                result = {
                    'success': True,
                    'details': f'Strategy modified: {plan.proposed_change}'
                }
            elif action == 'test_strategy':
                result = {
                    'success': True,
                    'details': 'Strategy test passed'
                }
            elif action == 'validate_knowledge':
                result = {
                    'success': True,
                    'details': 'Knowledge validation passed'
                }
            elif action == 'integrate_knowledge':
                result = {
                    'success': True,
                    'details': 'Knowledge integrated successfully'
                }
            elif action == 'verify_integration':
                result = {
                    'success': True,
                    'details': 'Knowledge integration verified'
                }
            elif action == 'analyze_dependencies':
                result = {
                    'success': True,
                    'details': 'Dependency analysis completed'
                }
            elif action == 'modify_structure':
                result = {
                    'success': True,
                    'details': f'Structural modification applied: {plan.proposed_change}'
                }
            elif action == 'validate_structure':
                result = {
                    'success': True,
                    'details': 'Structural modifications validated'
                }
            else:
                # Default action
                result = {
                    'success': True,
                    'details': f'Action {action} simulated successfully'
                }
            
            return {
                'step_id': step['step_id'],
                'success': result['success'],
                'details': result['details'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing implementation step {step['step_id']}: {e}")
            return {
                'step_id': step['step_id'],
                'success': False,
                'details': f'Error: {str(e)}',
                'timestamp': datetime.now()
            }
    
    async def _test_modification_effect(self, plan: SelfModificationPlan) -> Dict[str, Any]:
        """Test the effect of a modification."""
        # In a real implementation, this would run specific tests for the modification
        # For now, returning a simulated test result
        return {
            'effectiveness': 0.85,
            'side_effects': [],
            'performance_impact': 'positive',
            'stability': 'stable',
            'reversibility': True  # Whether the change can be reversed if needed
        }
    
    async def _assess_feasibility(self, plan: SelfModificationPlan) -> float:
        """Assess the feasibility of a modification."""
        # Higher score means more feasible
        # In a real implementation, this would be a sophisticated analysis
        base_feasibility = 0.7  # Default feasibility
        
        # Adjust based on modification type
        if plan.modification_type == 'knowledge':
            base_feasibility = 0.9  # Knowledge additions are typically very feasible
        elif plan.modification_type == 'behavior':
            base_feasibility = 0.8  # Behavior changes are quite feasible
        elif plan.modification_type == 'strategy':
            base_feasibility = 0.7  # Strategy changes are moderately feasible
        elif plan.modification_type == 'structure':
            base_feasibility = 0.5  # Structural changes are less feasible
        
        # Adjust based on risk assessment
        if plan.risk_assessment == 'high':
            base_feasibility *= 0.6
        elif plan.risk_assessment == 'medium':
            base_feasibility *= 0.8
            
        return max(0.1, min(1.0, base_feasibility))
    
    async def _assess_safety(self, plan: SelfModificationPlan) -> float:
        """Assess the safety of a modification."""
        # Higher score means safer
        # In a real implementation, this would be a sophisticated safety analysis
        base_safety = 0.7  # Default safety
        
        # Adjust based on modification type and risk
        if plan.risk_assessment == 'low':
            base_safety = 0.9
        elif plan.risk_assessment == 'medium':
            base_safety = 0.7
        elif plan.risk_assessment == 'high':
            base_safety = 0.4
            
        # For knowledge modifications, safety is typically high
        if plan.modification_type == 'knowledge':
            base_safety = min(1.0, base_safety + 0.2)
            
        return max(0.1, min(1.0, base_safety))
    
    async def _assess_impact(self, plan: SelfModificationPlan) -> float:
        """Assess the potential impact of a modification."""
        # Higher score means higher positive impact
        # In a real implementation, this would analyze expected benefits
        base_impact = 0.6  # Default impact
        
        # Modify based on modification type
        if plan.modification_type == 'structure':
            base_impact = 0.8  # Structural improvements often have high impact
        elif plan.modification_type == 'strategy':
            base_impact = 0.75  # Strategy improvements typically have good impact
        elif plan.modification_type == 'behavior':
            base_impact = 0.6  # Behavior changes have moderate impact
        elif plan.modification_type == 'knowledge':
            base_impact = 0.5  # Knowledge additions have moderate impact
            
        return base_impact
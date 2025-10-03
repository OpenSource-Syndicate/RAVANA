"""
Failure Learning and Prototype Development System for RAVANA AGI

This module implements an enhanced system for learning from failures,
analyzing why experiments failed, and creating prototypes based on lessons learned.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json

from core.llm import async_safe_call_llm
from core.config import Config

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of failures for better analysis."""
    TECHNICAL_ERROR = "technical_error"
    RESOURCE_LIMITATION = "resource_limitation"
    CONCEPTUAL_FLAW = "conceptual_flaw"
    IMPLEMENTATION_BUG = "implementation_bug"
    THEORETICAL_IMPOSSIBILITY = "theoretical_impossibility"
    INCOMPLETE_KNOWLEDGE = "incomplete_knowledge"
    WRONG_ASSUMPTION = "wrong_assumption"


class FailureLearningSystem:
    """System for learning from failures and creating prototypes."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.config = Config()
        self.failure_memory = []  # Stores analyzed failures for learning
        self.prototype_history = []  # Stores created prototypes
        self.failure_analysis_cache = {}  # Cache for failure analysis
        
    async def analyze_failure(self, 
                             failure_context: str, 
                             experiment_result: Dict[str, Any],
                             failure_details: str = None) -> Dict[str, Any]:
        """
        Perform deep analysis of a failure to understand root causes and lessons learned.
        
        Args:
            failure_context: Context of what was being attempted
            experiment_result: Result from the failed experiment
            failure_details: Specific details about what went wrong
            
        Returns:
            Dictionary with analysis results and lessons learned
        """
        logger.info(f"Analyzing failure in context: {failure_context[:100]}...")
        
        analysis_prompt = f"""
        Perform a comprehensive analysis of this failed experiment:

        Context: {failure_context}
        
        Experiment Result: {json.dumps(experiment_result, indent=2)}
        
        Failure Details: {failure_details or 'No specific details provided'}
        
        Analyze this failure by:
        1. Categorizing the type of failure using these categories:
           - TECHNICAL_ERROR: Implementation or technical issues
           - RESOURCE_LIMITATION: Insufficient resources (time, memory, etc.)
           - CONCEPTUAL_FLAW: Flaw in the underlying concept or approach
           - IMPLEMENTATION_BUG: Coding or implementation errors
           - THEORETICAL_IMPOSSIBILITY: Fundamentally impossible approach
           - INCOMPLETE_KNOWLEDGE: Missing required knowledge
           - WRONG_ASSUMPTION: Incorrect assumptions made
        
        2. Identifying the root cause
        3. Extracting specific lessons learned
        4. Identifying what could be repurposed or salvaged
        5. Suggesting alternative approaches that avoid the failure
        
        Return your analysis as JSON with these keys:
        - failure_category: The most appropriate category from above
        - root_causes: List of root causes
        - lessons_learned: List of specific lessons
        - salvageable_components: List of parts that could be reused
        - alternative_approaches: List of different approaches to try
        - critical_insights: Key insights that could prevent similar failures
        """
        
        try:
            response = await async_safe_call_llm(analysis_prompt)
            
            # Parse the response
            try:
                analysis_data = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the data more robustly
                # Look for the JSON block within the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    analysis_data = json.loads(json_str)
                else:
                    # Fallback: create a basic analysis
                    analysis_data = {
                        'failure_category': 'unknown',
                        'root_causes': ['Could not parse analysis'],
                        'lessons_learned': ['Improve analysis parsing'],
                        'salvageable_components': [],
                        'alternative_approaches': ['Retry with different parameters'],
                        'critical_insights': ['Analysis needs improvement']
                    }
            
            # Calculate confidence in the analysis
            analysis_data['confidence'] = 0.8  # Default confidence
            if 'root_causes' in analysis_data and len(analysis_data['root_causes']) > 0:
                analysis_data['confidence'] = min(1.0, analysis_data['confidence'] + 0.1)
            
            # Store analysis in memory for future learning
            failure_record = {
                'id': f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'context': failure_context,
                'result': experiment_result,
                'failure_details': failure_details,
                'analysis': analysis_data,
                'timestamp': datetime.now()
            }
            
            self.failure_memory.append(failure_record)
            
            # Keep only the most recent 50 failures to prevent memory issues
            if len(self.failure_memory) > 50:
                self.failure_memory = self.failure_memory[-50:]
            
            logger.info(f"Failure analysis completed, category: {analysis_data.get('failure_category', 'unknown')}")
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            # Return a default analysis on error
            return {
                'failure_category': 'unknown',
                'root_causes': [f'Analysis failed with error: {e}'],
                'lessons_learned': ['Error occurred during analysis'],
                'salvageable_components': [],
                'alternative_approaches': ['Retry with different parameters'],
                'critical_insights': ['Need to improve analysis process'],
                'confidence': 0.3
            }
    
    async def create_prototype_from_failure(self, 
                                          failure_analysis: Dict[str, Any], 
                                          context: str) -> Dict[str, Any]:
        """
        Create a prototype system based on lessons learned from a failure.
        
        Args:
            failure_analysis: Analysis of the failure to learn from
            context: Context for the new prototype
            
        Returns:
            Dictionary with prototype information
        """
        logger.info(f"Creating prototype from failure analysis in context: {context[:100]}...")
        
        prototype_prompt = f"""
        Based on this failure analysis, create a new prototype that addresses the identified issues:

        Failure Analysis:
        {json.dumps(failure_analysis, indent=2)}
        
        Context: {context}
        
        Create a prototype that:
        1. Addresses the root causes identified in the failure
        2. Incorporates the lessons learned
        3. Uses salvageable components where possible
        4. Implements one of the suggested alternative approaches
        5. Includes improvements to prevent similar failures
        
        The prototype should be described with:
        - A name for the prototype
        - A brief description of what it does
        - How it addresses the failure points
        - What makes it different from the failed approach
        - Expected benefits or improvements
        - Potential risks or challenges
        
        Return your prototype description as JSON with these keys:
        - name: Name of the prototype
        - description: Brief description of the prototype
        - failure_addresses: How it addresses the identified failures
        - improvements_over_original: What makes it better than the original
        - expected_benefits: Expected benefits from this approach
        - potential_risks: Potential risks with this new approach
        - implementation_complexity: How complex the implementation would be (low, medium, high)
        """
        
        try:
            response = await async_safe_call_llm(prototype_prompt)
            
            try:
                prototype_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    prototype_data = json.loads(json_str)
                else:
                    # Fallback: create a basic prototype
                    prototype_data = {
                        'name': 'Basic Prototype from Failure',
                        'description': 'A basic prototype created from failure analysis',
                        'failure_addresses': 'Addresses failure through improved approach',
                        'improvements_over_original': 'Uses lessons learned from failure',
                        'expected_benefits': 'Should avoid the issues that caused the original failure',
                        'potential_risks': 'May introduce new unknown issues',
                        'implementation_complexity': 'medium'
                    }
            
            # Add metadata to the prototype
            prototype_data['id'] = f"prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            prototype_data['created_from_failure_analysis'] = failure_analysis
            prototype_data['timestamp'] = datetime.now().isoformat()
            prototype_data['status'] = 'created'  # Initial status
            
            # Store the prototype history
            self.prototype_history.append(prototype_data)
            
            # Keep only the most recent 50 prototypes
            if len(self.prototype_history) > 50:
                self.prototype_history = self.prototype_history[-50:]
            
            logger.info(f"Prototype created: {prototype_data['name']}")
            return prototype_data
            
        except Exception as e:
            logger.error(f"Error creating prototype from failure: {e}")
            # Return a basic prototype on error
            return {
                'name': 'Error Recovery Prototype',
                'description': f'Prototype created after error in failure analysis: {e}',
                'failure_addresses': 'Attempts to address failure through alternative approach',
                'improvements_over_original': 'Different implementation to avoid original errors',
                'expected_benefits': 'Should handle errors more gracefully',
                'potential_risks': 'Unproven approach',
                'implementation_complexity': 'medium',
                'id': f"prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'status': 'error_recovery'
            }
    
    async def identify_cross_domain_lessons(self) -> List[Dict[str, Any]]:
        """
        Identify lessons that can be applied across different domains.
        
        Returns:
            List of cross-domain insights and their applications
        """
        if len(self.failure_memory) < 2:
            return []  # Need at least 2 failures to identify patterns
        
        # Extract failure contexts and analyses for pattern recognition
        failure_data = []
        for record in self.failure_memory[-10:]:  # Look at recent failures
            failure_data.append({
                'context': record['context'],
                'category': record['analysis'].get('failure_category', 'unknown'),
                'lessons': record['analysis'].get('lessons_learned', []),
                'root_causes': record['analysis'].get('root_causes', [])
            })
        
        pattern_prompt = f"""
        Analyze these recent failures to identify cross-domain patterns and lessons:

        Failures:
        {json.dumps(failure_data, indent=2)}
        
        Identify:
        1. Common patterns across different domains
        2. General lessons that apply beyond specific contexts
        3. Systematic issues in approach or methodology
        4. Successful patterns from the prototypes created
        
        Return your findings as JSON with these keys:
        - cross_domain_patterns: Common patterns across domains
        - general_lessons: Lessons applicable across contexts
        - systematic_issues: Issues in overall approach
        - successful_patterns: What worked well in prototypes
        - recommendations: Recommendations for future experiments
        """
        
        try:
            response = await async_safe_call_llm(pattern_prompt)
            
            try:
                patterns_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    patterns_data = json.loads(json_str)
                else:
                    patterns_data = {
                        'cross_domain_patterns': [],
                        'general_lessons': [],
                        'systematic_issues': [],
                        'successful_patterns': [],
                        'recommendations': []
                    }
            
            return patterns_data
            
        except Exception as e:
            logger.error(f"Error identifying cross-domain lessons: {e}")
            return []
    
    async def get_failure_insights(self) -> Dict[str, Any]:
        """
        Get insights from the failure memory.
        
        Returns:
            Dictionary with insights and statistics about failures
        """
        if not self.failure_memory:
            return {
                'total_failures': 0,
                'most_common_categories': [],
                'lessons_learned': [],
                'improvement_trends': []
            }
        
        # Count failure categories
        category_counts = {}
        all_lessons = []
        for record in self.failure_memory:
            category = record['analysis'].get('failure_category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            lessons = record['analysis'].get('lessons_learned', [])
            all_lessons.extend(lessons)
        
        # Get most common categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_failures': len(self.failure_memory),
            'most_common_categories': sorted_categories[:5],  # Top 5
            'lessons_learned': list(set(all_lessons))[:10],  # Unique lessons, top 10
            'improvement_trends': await self._get_improvement_trends()
        }
    
    async def _get_improvement_trends(self) -> List[Dict[str, Any]]:
        """
        Identify improvement trends from failure-prototype cycles.
        """
        if len(self.prototype_history) < 2:
            return []
        
        # Look for patterns where prototypes improved on previous failures
        trend_prompt = f"""
        Analyze these prototypes created from failures to identify improvement trends:

        Prototypes:
        {json.dumps(self.prototype_history[-10:], indent=2)}  # Look at recent prototypes
        
        Identify:
        1. Trends in how failures are being addressed
        2. Improvements in approach over time
        3. Patterns in successful vs unsuccessful prototype creation
        4. Evolution in the system's ability to learn from failures
        
        Return your analysis as JSON with these keys:
        - improvement_trends: How approaches are improving
        - learning_patterns: Patterns in how the system learns
        - evolution_insights: How the learning system has evolved
        - future_directions: Where the learning system should go next
        """
        
        try:
            response = await async_safe_call_llm(trend_prompt)
            
            try:
                trends_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    trends_data = json.loads(json_str)
                else:
                    trends_data = {
                        'improvement_trends': [],
                        'learning_patterns': [],
                        'evolution_insights': [],
                        'future_directions': []
                    }
            
            return trends_data
            
        except Exception as e:
            logger.error(f"Error getting improvement trends: {e}")
            return []
    
    async def apply_lessons_to_task(self, task_description: str) -> Dict[str, Any]:
        """
        Apply lessons learned from past failures to a new task.
        
        Args:
            task_description: Description of the new task
            
        Returns:
            Recommendations for avoiding past failures
        """
        if not self.failure_memory:
            return {
                'task': task_description,
                'avoidance_recommendations': ['No past failures to learn from'],
                'improvement_suggestions': [],
                'risk_alerts': []
            }
        
        # Get relevant past failures based on task similarity
        recent_failures = self.failure_memory[-8:]  # Use recent failures
        
        application_prompt = f"""
        Based on this new task and past failures, provide recommendations:

        New Task: {task_description}
        
        Recent Failures with Analysis:
        {json.dumps(recent_failures, default=str, indent=2)}
        
        Provide recommendations for:
        1. How to approach this task to avoid similar failures
        2. Potential improvements based on lessons learned
        3. Risk areas to be particularly careful about
        4. Successful patterns from prototypes that could be applied
        
        Return your recommendations as JSON with these keys:
        - task: The original task description
        - avoidance_recommendations: How to avoid past failure patterns
        - improvement_suggestions: Specific improvements to implement
        - risk_alerts: Risk areas to monitor carefully
        - successful_patterns_to_apply: Patterns from successful prototypes
        """
        
        try:
            response = await async_safe_call_llm(application_prompt)
            
            try:
                application_data = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    application_data = json.loads(json_str)
                else:
                    application_data = {
                        'task': task_description,
                        'avoidance_recommendations': ['Apply general best practices'],
                        'improvement_suggestions': ['Use validated approaches'],
                        'risk_alerts': ['Monitor for implementation errors'],
                        'successful_patterns_to_apply': []
                    }
            
            return application_data
            
        except Exception as e:
            logger.error(f"Error applying lessons to task: {e}")
            return {
                'task': task_description,
                'avoidance_recommendations': [f'Error in lesson application: {e}'],
                'improvement_suggestions': [],
                'risk_alerts': ['Could not apply lessons due to error'],
                'successful_patterns_to_apply': []
            }
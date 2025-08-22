"""
Failure Analysis Engine

Comprehensive system for extracting maximum value from building failures through
sophisticated analysis techniques, pattern recognition, and learning extraction.
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import hashlib

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of build failures"""
    DESIGN_FLAW = "design_flaw"
    IMPLEMENTATION_ERROR = "implementation_error"
    RESOURCE_CONSTRAINT = "resource_constraint"
    KNOWLEDGE_GAP = "knowledge_gap"
    ENVIRONMENTAL_FACTOR = "environmental_factor"
    COMPLEXITY_FACTOR = "complexity_factor"
    ASSUMPTION_VIOLATION = "assumption_violation"
    INTEGRATION_FAILURE = "integration_failure"
    TIMING_ISSUE = "timing_issue"
    COMMUNICATION_BREAKDOWN = "communication_breakdown"


class FailureSeverity(Enum):
    """Severity levels of failures"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class AnalysisMethod(Enum):
    """Methods for failure analysis"""
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    CROSS_STRATEGY_ANALYSIS = "cross_strategy_analysis"
    PATTERN_ANALYSIS = "pattern_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CAUSAL_CHAIN_ANALYSIS = "causal_chain_analysis"


@dataclass
class FailurePoint:
    """Specific point of failure in execution"""
    location: str
    description: str
    category: FailureCategory
    severity: FailureSeverity
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    immediate_cause: str = ""
    root_cause: str = ""


@dataclass
class FailurePattern:
    """Identified pattern in failures"""
    pattern_id: str
    description: str
    frequency: int
    contexts: List[str]
    common_factors: List[str]
    mitigation_strategies: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CausalChain:
    """Chain of causation leading to failure"""
    initiating_event: str
    intermediate_causes: List[str]
    proximate_cause: str
    failure_outcome: str
    intervention_points: List[str] = field(default_factory=list)
    probability_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class FailureAnalysisResult:
    """Complete failure analysis result"""
    failure_id: str
    build_context: Dict[str, Any]
    failure_points: List[FailurePoint]
    root_causes: List[str]
    contributing_factors: List[str]
    failure_patterns: List[FailurePattern]
    causal_chains: List[CausalChain]
    similar_failures: List[str]
    lessons_learned: List[str]
    improvement_recommendations: List[str]
    prevention_strategies: List[str]
    confidence_score: float
    analysis_methods_used: List[AnalysisMethod]


class FailureAnalysisEngine:
    """
    Comprehensive system for analyzing and learning from build failures
    """
    
    def __init__(self, agi_system, knowledge_service=None):
        self.agi_system = agi_system
        self.knowledge_service = knowledge_service or agi_system.knowledge_service
        
        # Failure knowledge base
        self.failure_history: List[FailureAnalysisResult] = []
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.root_cause_frequency: Counter = Counter()
        self.mitigation_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
        # Analysis configuration
        self.pattern_detection_threshold = 3  # Minimum occurrences to form pattern
        self.similarity_threshold = 0.7  # Threshold for similar failure detection
        self.confidence_threshold = 0.6  # Minimum confidence for recommendations
        
        # Domain-specific failure knowledge
        self.domain_failure_patterns = self._initialize_domain_patterns()
        self.common_failure_modes = self._initialize_common_failure_modes()
        self.diagnostic_questions = self._initialize_diagnostic_questions()
        
        logger.info("Failure Analysis Engine initialized")
    
    async def analyze_failure(
        self,
        failure_details: Dict[str, Any],
        build_context: Any,
        research_similar_failures: bool = True
    ) -> FailureAnalysisResult:
        """
        Comprehensive analysis of build failures for learning extraction
        """
        logger.info(f"Starting failure analysis for build: {getattr(build_context, 'build_id', 'unknown')}")
        
        # Generate unique failure ID
        failure_id = self._generate_failure_id(failure_details, build_context)
        
        # Extract failure points from details
        failure_points = await self._extract_failure_points(failure_details, build_context)
        
        # Perform root cause analysis
        root_causes = await self._perform_root_cause_analysis(failure_points, build_context)
        
        # Identify contributing factors
        contributing_factors = await self._identify_contributing_factors(failure_points, build_context)
        
        # Detect failure patterns
        failure_patterns = await self._detect_failure_patterns(failure_points, root_causes)
        
        # Build causal chains
        causal_chains = await self._build_causal_chains(failure_points, root_causes)
        
        # Find similar failures
        similar_failures = []
        if research_similar_failures:
            similar_failures = await self._find_similar_failures(failure_details, build_context)
        
        # Extract lessons learned
        lessons_learned = await self._extract_lessons_learned(
            failure_points, root_causes, contributing_factors, similar_failures
        )
        
        # Generate recommendations
        improvement_recommendations = await self._generate_improvement_recommendations(
            root_causes, contributing_factors, failure_patterns
        )
        
        # Generate prevention strategies
        prevention_strategies = await self._generate_prevention_strategies(
            causal_chains, failure_patterns, lessons_learned
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(
            failure_points, root_causes, similar_failures
        )
        
        # Record analysis methods used
        analysis_methods_used = [
            AnalysisMethod.ROOT_CAUSE_ANALYSIS,
            AnalysisMethod.PATTERN_ANALYSIS,
            AnalysisMethod.CAUSAL_CHAIN_ANALYSIS
        ]
        
        if similar_failures:
            analysis_methods_used.append(AnalysisMethod.COMPARATIVE_ANALYSIS)
        
        # Create analysis result
        analysis_result = FailureAnalysisResult(
            failure_id=failure_id,
            build_context=self._extract_build_context_dict(build_context),
            failure_points=failure_points,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            failure_patterns=failure_patterns,
            causal_chains=causal_chains,
            similar_failures=similar_failures,
            lessons_learned=lessons_learned,
            improvement_recommendations=improvement_recommendations,
            prevention_strategies=prevention_strategies,
            confidence_score=confidence_score,
            analysis_methods_used=analysis_methods_used
        )
        
        # Store analysis for future learning
        await self._store_failure_analysis(analysis_result)
        
        logger.info(f"Failure analysis complete. Confidence: {confidence_score:.2f}")
        return analysis_result
    
    async def identify_failure_trends(
        self,
        time_window_hours: int = 168  # 1 week
    ) -> Dict[str, Any]:
        """Identify trends in failure patterns over time"""
        
        current_time = time.time()
        cutoff_time = current_time - (time_window_hours * 3600)
        
        # Filter recent failures
        recent_failures = [
            f for f in self.failure_history
            if any(fp.timestamp >= cutoff_time for fp in f.failure_points)
        ]
        
        if not recent_failures:
            return {"message": "No recent failures to analyze"}
        
        # Analyze trends
        category_trends = self._analyze_category_trends(recent_failures)
        severity_trends = self._analyze_severity_trends(recent_failures)
        temporal_patterns = self._analyze_temporal_patterns(recent_failures)
        recurring_issues = self._identify_recurring_issues(recent_failures)
        
        return {
            "analysis_period_hours": time_window_hours,
            "total_failures_analyzed": len(recent_failures),
            "category_trends": category_trends,
            "severity_trends": severity_trends,
            "temporal_patterns": temporal_patterns,
            "recurring_issues": recurring_issues,
            "recommendations": self._generate_trend_based_recommendations(
                category_trends, severity_trends, recurring_issues
            )
        }
    
    async def suggest_failure_prevention_measures(
        self,
        build_description: str,
        strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest prevention measures based on historical failures"""
        
        prevention_measures = []
        
        # Analyze build description for risk factors
        risk_factors = self._identify_build_risk_factors(build_description)
        
        # Check each strategy for potential failure modes
        for strategy in strategies:
            strategy_risks = await self._assess_strategy_failure_risks(strategy, risk_factors)
            
            strategy_prevention = {
                "strategy_id": strategy.get("id", "unknown"),
                "strategy_name": strategy.get("name", "Unknown Strategy"),
                "identified_risks": strategy_risks,
                "prevention_measures": [],
                "monitoring_points": [],
                "fallback_options": []
            }
            
            # Generate prevention measures for each risk
            for risk in strategy_risks:
                measures = await self._generate_risk_specific_prevention(risk, strategy)
                strategy_prevention["prevention_measures"].extend(measures)
            
            # Add monitoring points
            monitoring_points = self._identify_monitoring_points(strategy, strategy_risks)
            strategy_prevention["monitoring_points"] = monitoring_points
            
            # Suggest fallback options
            fallback_options = await self._suggest_fallback_options(strategy, strategy_risks)
            strategy_prevention["fallback_options"] = fallback_options
            
            prevention_measures.append(strategy_prevention)
        
        return prevention_measures
    
    async def generate_failure_recovery_plan(
        self,
        failure_analysis: FailureAnalysisResult,
        available_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive recovery plan from failure"""
        
        recovery_plan = {
            "immediate_actions": [],
            "short_term_fixes": [],
            "long_term_improvements": [],
            "alternative_approaches": [],
            "resource_requirements": {},
            "timeline_estimate": {},
            "success_probability": 0.0
        }
        
        # Immediate actions based on failure category
        for failure_point in failure_analysis.failure_points:
            immediate_actions = self._generate_immediate_actions(failure_point)
            recovery_plan["immediate_actions"].extend(immediate_actions)
        
        # Short-term fixes for root causes
        for root_cause in failure_analysis.root_causes:
            short_term_fixes = await self._generate_short_term_fixes(root_cause, available_resources)
            recovery_plan["short_term_fixes"].extend(short_term_fixes)
        
        # Long-term improvements
        for recommendation in failure_analysis.improvement_recommendations:
            long_term_improvement = await self._convert_recommendation_to_action(
                recommendation, available_resources
            )
            if long_term_improvement:
                recovery_plan["long_term_improvements"].append(long_term_improvement)
        
        # Generate alternative approaches
        alternative_approaches = await self._generate_alternative_approaches(
            failure_analysis, available_resources
        )
        recovery_plan["alternative_approaches"] = alternative_approaches
        
        # Estimate resource requirements
        recovery_plan["resource_requirements"] = self._estimate_recovery_resources(recovery_plan)
        
        # Estimate timeline
        recovery_plan["timeline_estimate"] = self._estimate_recovery_timeline(recovery_plan)
        
        # Calculate success probability
        recovery_plan["success_probability"] = self._calculate_recovery_success_probability(
            failure_analysis, recovery_plan
        )
        
        return recovery_plan
    
    # Core analysis methods
    
    async def _extract_failure_points(
        self,
        failure_details: Dict[str, Any],
        build_context: Any
    ) -> List[FailurePoint]:
        """Extract specific failure points from failure details"""
        
        failure_points = []
        
        # Extract from error messages
        error_message = failure_details.get('error_message', '')
        if error_message:
            failure_point = self._analyze_error_message(error_message, build_context)
            if failure_point:
                failure_points.append(failure_point)
        
        # Extract from execution results
        execution_results = failure_details.get('execution_results', [])
        for result in execution_results:
            if hasattr(result, 'error_message') and result.error_message:
                failure_point = self._analyze_execution_error(result, build_context)
                if failure_point:
                    failure_points.append(failure_point)
        
        # Extract from strategy failures
        strategy_failures = failure_details.get('strategy_failures', [])
        for strategy_failure in strategy_failures:
            failure_point = self._analyze_strategy_failure(strategy_failure, build_context)
            if failure_point:
                failure_points.append(failure_point)
        
        # Extract from timeout events
        if failure_details.get('timeout', False):
            failure_point = FailurePoint(
                location="execution_timeout",
                description="Strategy execution exceeded time limit",
                category=FailureCategory.TIMING_ISSUE,
                severity=FailureSeverity.HIGH,
                timestamp=time.time(),
                context={"timeout_duration": failure_details.get('timeout_duration', 'unknown')},
                immediate_cause="Execution timeout",
                root_cause="Insufficient time allocation or complex problem"
            )
            failure_points.append(failure_point)
        
        return failure_points
    
    async def _perform_root_cause_analysis(
        self,
        failure_points: List[FailurePoint],
        build_context: Any
    ) -> List[str]:
        """Perform root cause analysis using multiple techniques"""
        
        root_causes = []
        
        # Apply 5-Why analysis
        for failure_point in failure_points:
            why_analysis = await self._apply_five_why_analysis(failure_point, build_context)
            if why_analysis:
                root_causes.append(why_analysis)
        
        # Apply cause-and-effect analysis
        cause_effect_analysis = await self._apply_cause_effect_analysis(failure_points, build_context)
        root_causes.extend(cause_effect_analysis)
        
        # Apply failure mode analysis
        failure_mode_analysis = await self._apply_failure_mode_analysis(failure_points, build_context)
        root_causes.extend(failure_mode_analysis)
        
        # Remove duplicates and rank by probability
        unique_root_causes = list(set(root_causes))
        ranked_root_causes = self._rank_root_causes_by_probability(unique_root_causes, failure_points)
        
        return ranked_root_causes[:5]  # Return top 5 root causes
    
    async def _detect_failure_patterns(
        self,
        failure_points: List[FailurePoint],
        root_causes: List[str]
    ) -> List[FailurePattern]:
        """Detect patterns in failures across different contexts"""
        
        detected_patterns = []
        
        # Check against known patterns
        for pattern_id, pattern in self.failure_patterns.items():
            if self._pattern_matches_current_failure(pattern, failure_points, root_causes):
                pattern.frequency += 1
                detected_patterns.append(pattern)
        
        # Generate new patterns
        new_patterns = await self._generate_new_failure_patterns(failure_points, root_causes)
        detected_patterns.extend(new_patterns)
        
        # Update pattern database
        for pattern in new_patterns:
            self.failure_patterns[pattern.pattern_id] = pattern
        
        return detected_patterns
    
    async def _build_causal_chains(
        self,
        failure_points: List[FailurePoint],
        root_causes: List[str]
    ) -> List[CausalChain]:
        """Build causal chains from initiating events to final failures"""
        
        causal_chains = []
        
        for root_cause in root_causes:
            # Find the chain of events leading from root cause to failure
            chain = await self._trace_causal_chain(root_cause, failure_points)
            if chain:
                causal_chains.append(chain)
        
        return causal_chains
    
    async def _find_similar_failures(
        self,
        failure_details: Dict[str, Any],
        build_context: Any
    ) -> List[str]:
        """Find similar failures in the knowledge base"""
        
        similar_failures = []
        
        # Create failure signature
        failure_signature = self._create_failure_signature(failure_details, build_context)
        
        # Search knowledge base for similar failures
        if self.knowledge_service:
            try:
                query = f"failure analysis similar to: {failure_signature}"
                knowledge_results = await self.knowledge_service.search_knowledge(
                    query=query,
                    category="failure_analysis",
                    limit=5
                )
                
                for result in knowledge_results:
                    if self._calculate_failure_similarity(failure_signature, result) > self.similarity_threshold:
                        similar_failures.append(result.get('id', 'unknown'))
                        
            except Exception as e:
                logger.warning(f"Failed to search for similar failures: {e}")
        
        # Search internal failure history
        for historical_failure in self.failure_history:
            similarity = self._calculate_failure_similarity(
                failure_signature,
                self._create_failure_signature_from_analysis(historical_failure)
            )
            
            if similarity > self.similarity_threshold:
                similar_failures.append(historical_failure.failure_id)
        
        return similar_failures
    
    async def _extract_lessons_learned(
        self,
        failure_points: List[FailurePoint],
        root_causes: List[str],
        contributing_factors: List[str],
        similar_failures: List[str]
    ) -> List[str]:
        """Extract actionable lessons from failure analysis"""
        
        lessons = []
        
        # Generate lessons from root causes
        for root_cause in root_causes:
            lesson = self._generate_lesson_from_root_cause(root_cause)
            if lesson:
                lessons.append(lesson)
        
        # Generate lessons from failure points
        for failure_point in failure_points:
            lesson = self._generate_lesson_from_failure_point(failure_point)
            if lesson:
                lessons.append(lesson)
        
        # Extract lessons from similar failures
        for similar_failure_id in similar_failures:
            similar_lessons = await self._extract_lessons_from_similar_failure(similar_failure_id)
            lessons.extend(similar_lessons)
        
        # Generate systemic lessons
        systemic_lessons = self._generate_systemic_lessons(
            failure_points, root_causes, contributing_factors
        )
        lessons.extend(systemic_lessons)
        
        # Remove duplicates and prioritize
        unique_lessons = list(set(lessons))
        prioritized_lessons = self._prioritize_lessons(unique_lessons, failure_points)
        
        return prioritized_lessons[:10]  # Return top 10 lessons
    
    # Utility and helper methods
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain-specific failure patterns"""
        return {
            "physics_simulation": [
                "Numerical instability in differential equations",
                "Boundary condition violations",
                "Conservation law violations",
                "Scale separation issues"
            ],
            "computational": [
                "Memory allocation failures",
                "Algorithmic complexity explosion",
                "Floating point precision errors",
                "Deadlock and race conditions"
            ],
            "experimental": [
                "Measurement precision limitations",
                "Environmental interference",
                "Equipment calibration drift",
                "Statistical significance issues"
            ],
            "integration": [
                "Interface compatibility problems",
                "Data format mismatches",
                "Version compatibility issues",
                "Communication protocol failures"
            ]
        }
    
    def _initialize_common_failure_modes(self) -> Dict[FailureCategory, List[str]]:
        """Initialize common failure modes by category"""
        return {
            FailureCategory.DESIGN_FLAW: [
                "Insufficient requirements analysis",
                "Inadequate architecture design",
                "Missing edge case handling",
                "Scalability limitations not considered"
            ],
            FailureCategory.IMPLEMENTATION_ERROR: [
                "Logic errors in code",
                "Incorrect algorithm implementation",
                "Missing error handling",
                "Resource leaks"
            ],
            FailureCategory.RESOURCE_CONSTRAINT: [
                "Insufficient memory allocation",
                "CPU time limitations",
                "Network bandwidth constraints",
                "Storage capacity limits"
            ],
            FailureCategory.KNOWLEDGE_GAP: [
                "Missing domain expertise",
                "Incomplete understanding of problem",
                "Lack of relevant experience",
                "Insufficient research"
            ]
        }
    
    def _initialize_diagnostic_questions(self) -> Dict[FailureCategory, List[str]]:
        """Initialize diagnostic questions for each failure category"""
        return {
            FailureCategory.DESIGN_FLAW: [
                "Were all requirements properly identified and analyzed?",
                "Was the architecture design reviewed by experts?",
                "Were edge cases and failure modes considered?",
                "Was the design validated against use cases?"
            ],
            FailureCategory.IMPLEMENTATION_ERROR: [
                "Was the code reviewed by peers?",
                "Were unit tests comprehensive?",
                "Was the implementation tested under stress conditions?",
                "Were all error conditions handled properly?"
            ],
            FailureCategory.RESOURCE_CONSTRAINT: [
                "Were resource requirements properly estimated?",
                "Was resource usage monitored during execution?",
                "Were resource limits tested before deployment?",
                "Were alternatives for resource optimization considered?"
            ],
            FailureCategory.KNOWLEDGE_GAP: [
                "Was sufficient research conducted before implementation?",
                "Were domain experts consulted?",
                "Were alternative approaches investigated?",
                "Was the team's expertise adequate for the challenge?"
            ]
        }
    
    def _generate_failure_id(self, failure_details: Dict[str, Any], build_context: Any) -> str:
        """Generate unique ID for failure"""
        
        # Create a hash based on key failure characteristics
        failure_data = {
            'error_message': failure_details.get('error_message', ''),
            'build_description': getattr(build_context, 'description', ''),
            'timestamp': int(time.time() // 3600)  # Hour-level granularity
        }
        
        failure_string = str(sorted(failure_data.items()))
        return hashlib.md5(failure_string.encode()).hexdigest()[:12]
    
    async def _store_failure_analysis(self, analysis_result: FailureAnalysisResult):
        """Store failure analysis for future learning"""
        
        # Add to internal history
        self.failure_history.append(analysis_result)
        
        # Update root cause frequency
        for root_cause in analysis_result.root_causes:
            self.root_cause_frequency[root_cause] += 1
        
        # Keep history bounded
        if len(self.failure_history) > 1000:
            self.failure_history = self.failure_history[-1000:]
        
        # Store in knowledge base if available
        if self.knowledge_service:
            try:
                await self.knowledge_service.add_knowledge(
                    content=self._serialize_failure_analysis(analysis_result),
                    source="failure_analysis_engine",
                    category="failure_analysis"
                )
            except Exception as e:
                logger.warning(f"Failed to store failure analysis in knowledge base: {e}")
    
    def _serialize_failure_analysis(self, analysis: FailureAnalysisResult) -> str:
        """Serialize failure analysis for storage"""
        
        summary = f"""
        Failure Analysis Report - ID: {analysis.failure_id}
        
        Root Causes:
        {chr(10).join(f"- {cause}" for cause in analysis.root_causes)}
        
        Contributing Factors:
        {chr(10).join(f"- {factor}" for factor in analysis.contributing_factors)}
        
        Lessons Learned:
        {chr(10).join(f"- {lesson}" for lesson in analysis.lessons_learned)}
        
        Improvement Recommendations:
        {chr(10).join(f"- {rec}" for rec in analysis.improvement_recommendations)}
        
        Prevention Strategies:
        {chr(10).join(f"- {strategy}" for strategy in analysis.prevention_strategies)}
        
        Confidence Score: {analysis.confidence_score:.2f}
        """
        
        return summary.strip()
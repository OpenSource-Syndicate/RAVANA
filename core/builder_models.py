"""
Data Models and Schemas for Intelligent Adaptive Builder

Comprehensive data models for build attempts, strategies, failure analysis,
and all related entities in the Intelligent Adaptive Builder system.
"""

from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


# Enums for type safety

class BuildDifficulty(str, Enum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    IMPOSSIBLE = "impossible"
    TRANSCENDENT = "transcendent"


class BuildStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_RESEARCH = "requires_research"
    STRATEGY_EXHAUSTED = "strategy_exhausted"


class StrategyType(str, Enum):
    PHYSICS_BASED = "physics_based"
    COMPUTATIONAL = "computational"
    HEURISTIC = "heuristic"
    BIO_INSPIRED = "bio_inspired"
    META_ALGORITHMIC = "meta_algorithmic"
    HYBRID = "hybrid"
    EXPERIMENTAL = "experimental"


class ReasoningType(str, Enum):
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    COUNTERFACTUAL = "counterfactual"
    ABDUCTIVE = "abductive"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    SYSTEMS_THINKING = "systems_thinking"
    CONSTRAINT_BASED = "constraint_based"


class FeasibilityLevel(str, Enum):
    IMPOSSIBLE = "impossible"
    EXTREMELY_DIFFICULT = "extremely_difficult"
    CHALLENGING = "challenging"
    MODERATE = "moderate"
    STRAIGHTFORWARD = "straightforward"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"


class FailureCategory(str, Enum):
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


class FailureSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


# Core Models

class BuildAttempt(SQLModel, table=True):
    """Main table for build attempts"""
    __tablename__ = "build_attempts"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    description: str = Field(index=True)
    difficulty: BuildDifficulty = Field(index=True)
    status: BuildStatus = Field(default=BuildStatus.PENDING, index=True)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration
    max_strategies: int = Field(default=5)
    parallel_execution: bool = Field(default=True)
    enable_online_research: bool = Field(default=True)
    persistence_mode: str = Field(default="high")

    # Context and constraints
    domain: str = Field(default="general", index=True)
    complexity_factors: Dict[str, Any] = Field(
        default={}, sa_column=Column(JSON))
    prerequisites: List[str] = Field(default=[], sa_column=Column(JSON))
    constraints: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    success_criteria: List[str] = Field(default=[], sa_column=Column(JSON))

    # Results
    success: bool = Field(default=False, index=True)
    final_artifact: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON))
    confidence_score: float = Field(default=0.0)
    novelty_score: float = Field(default=0.0)

    # Execution metrics
    total_execution_time: Optional[float] = None
    strategies_attempted: int = Field(default=0)
    strategies_succeeded: int = Field(default=0)
    iterations_completed: int = Field(default=0)

    # Learning outcomes
    lessons_learned: List[str] = Field(default=[], sa_column=Column(JSON))
    improvement_recommendations: List[str] = Field(
        default=[], sa_column=Column(JSON))

    # Relationships
    strategies: List["BuildStrategy"] = Relationship(
        back_populates="build_attempt")
    executions: List["StrategyExecution"] = Relationship(
        back_populates="build_attempt")
    failure_analyses: List["FailureAnalysis"] = Relationship(
        back_populates="build_attempt")
    research_sessions: List["ResearchSession"] = Relationship(
        back_populates="build_attempt")


class BuildStrategy(SQLModel, table=True):
    """Strategies for build attempts"""
    __tablename__ = "build_strategies"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    build_attempt_id: str = Field(foreign_key="build_attempts.id", index=True)

    name: str = Field(index=True)
    description: str
    strategy_type: StrategyType = Field(index=True)

    # Reasoning chain
    reasoning_type: ReasoningType = Field(index=True)
    reasoning_steps: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    overall_confidence: float = Field(default=0.0)

    # Feasibility assessment
    feasibility_level: FeasibilityLevel = Field(index=True)
    technical_feasibility: float = Field(default=0.0)
    resource_feasibility: float = Field(default=0.0)
    time_feasibility: float = Field(default=0.0)
    knowledge_feasibility: float = Field(default=0.0)

    # Success prediction
    expected_difficulty: float = Field(default=0.5)
    success_probability: float = Field(default=0.5)

    # Implementation details
    approach_steps: List[str] = Field(default=[], sa_column=Column(JSON))
    tools_required: List[str] = Field(default=[], sa_column=Column(JSON))
    skills_required: List[str] = Field(default=[], sa_column=Column(JSON))

    # Resource requirements
    resource_requirements: Dict[str, Any] = Field(
        default={}, sa_column=Column(JSON))
    estimated_time_minutes: Optional[int] = None
    computational_complexity: str = Field(default="medium")

    # Risk assessment
    risk_factors: List[str] = Field(default=[], sa_column=Column(JSON))
    mitigation_strategies: List[str] = Field(
        default=[], sa_column=Column(JSON))
    failure_modes: List[str] = Field(default=[], sa_column=Column(JSON))

    # Execution configuration
    execution_order: int = Field(default=0)
    parallel_compatible: bool = Field(default=True)
    dependencies: List[str] = Field(default=[], sa_column=Column(JSON))
    alternative_strategies: List[str] = Field(
        default=[], sa_column=Column(JSON))

    # Learning integration
    similar_successes: List[str] = Field(default=[], sa_column=Column(JSON))
    related_failures: List[str] = Field(default=[], sa_column=Column(JSON))
    domain_expertise_required: List[str] = Field(
        default=[], sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    build_attempt: BuildAttempt = Relationship(back_populates="strategies")
    executions: List["StrategyExecution"] = Relationship(
        back_populates="strategy")
    creative_insights: List["CreativeInsight"] = Relationship(
        back_populates="strategy")


class StrategyExecution(SQLModel, table=True):
    """Individual strategy execution records"""
    __tablename__ = "strategy_executions"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    build_attempt_id: str = Field(foreign_key="build_attempts.id", index=True)
    strategy_id: str = Field(foreign_key="build_strategies.id", index=True)

    # Execution details
    # parallel, sequential, hybrid, adaptive
    execution_mode: str = Field(default="sequential")
    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING, index=True)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Results
    success: bool = Field(default=False, index=True)
    artifact: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON))
    intermediate_results: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))

    # Performance metrics
    execution_time_seconds: Optional[float] = None
    resource_usage: Dict[str, float] = Field(
        default={}, sa_column=Column(JSON))
    confidence_score: float = Field(default=0.0)

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    failure_points: List[str] = Field(default=[], sa_column=Column(JSON))

    # Learning outcomes
    lessons_learned: List[str] = Field(default=[], sa_column=Column(JSON))
    success_factors: List[str] = Field(default=[], sa_column=Column(JSON))

    # Relationships
    build_attempt: BuildAttempt = Relationship(back_populates="executions")
    strategy: BuildStrategy = Relationship(back_populates="executions")


class CreativeInsight(SQLModel, table=True):
    """Creative insights used in strategy development"""
    __tablename__ = "creative_insights"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    strategy_id: str = Field(foreign_key="build_strategies.id", index=True)

    # Insight details
    metaphor: str
    domain_source: str = Field(index=True)
    domain_target: str = Field(index=True)
    bridge_concept: str
    application: str

    # Quality metrics
    confidence: float = Field(default=0.0)
    novelty: float = Field(default=0.0)
    effectiveness: Optional[float] = None  # Set after application

    # Usage tracking
    times_used: int = Field(default=0)
    success_rate: float = Field(default=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    strategy: BuildStrategy = Relationship(back_populates="creative_insights")


class FailureAnalysis(SQLModel, table=True):
    """Comprehensive failure analysis records"""
    __tablename__ = "failure_analyses"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    build_attempt_id: str = Field(foreign_key="build_attempts.id", index=True)

    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_methods_used: List[str] = Field(
        default=[], sa_column=Column(JSON))
    confidence_score: float = Field(default=0.0)

    # Failure characterization
    failure_category: FailureCategory = Field(index=True)
    failure_severity: FailureSeverity = Field(index=True)
    root_causes: List[str] = Field(default=[], sa_column=Column(JSON))
    contributing_factors: List[str] = Field(default=[], sa_column=Column(JSON))

    # Detailed analysis
    failure_points: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    causal_chains: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    similar_failures: List[str] = Field(default=[], sa_column=Column(JSON))

    # Learning outcomes
    lessons_learned: List[str] = Field(default=[], sa_column=Column(JSON))
    improvement_recommendations: List[str] = Field(
        default=[], sa_column=Column(JSON))
    prevention_strategies: List[str] = Field(
        default=[], sa_column=Column(JSON))

    # Pattern information
    failure_patterns: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    pattern_frequency: Dict[str, int] = Field(
        default={}, sa_column=Column(JSON))

    # Recovery information
    recovery_attempted: bool = Field(default=False)
    recovery_success: Optional[bool] = None
    recovery_strategies: List[str] = Field(default=[], sa_column=Column(JSON))

    # Relationships
    build_attempt: BuildAttempt = Relationship(
        back_populates="failure_analyses")


class ResearchSession(SQLModel, table=True):
    """Online research sessions for build challenges"""
    __tablename__ = "research_sessions"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)
    build_attempt_id: str = Field(foreign_key="build_attempts.id", index=True)

    # Research parameters
    research_query: str = Field(index=True)
    research_domains: List[str] = Field(default=[], sa_column=Column(JSON))
    research_depth: str = Field(default="comprehensive")

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Results
    sources_found: int = Field(default=0)
    relevant_papers: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    technical_documentation: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    community_insights: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    failure_case_studies: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    best_practices: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))

    # Quality metrics
    relevance_score: float = Field(default=0.0)
    novelty_score: float = Field(default=0.0)
    applicability_score: float = Field(default=0.0)

    # Application results
    insights_applied: List[str] = Field(default=[], sa_column=Column(JSON))
    strategy_modifications: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))

    # Relationships
    build_attempt: BuildAttempt = Relationship(
        back_populates="research_sessions")


class PersonalityState(SQLModel, table=True):
    """Enhanced personality system state tracking"""
    __tablename__ = "personality_states"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)

    # Personality configuration
    name: str = Field(default="Ravana")
    creativity_level: float = Field(default=0.8)
    risk_tolerance: str = Field(default="high")

    # Risk-taking controller state
    base_confidence: float = Field(default=0.6)
    learning_modifier: float = Field(default=1.0)
    failure_resilience: float = Field(default=0.8)
    success_amplification: float = Field(default=1.2)
    impossible_attraction: float = Field(default=0.7)

    # Performance history
    total_builds_attempted: int = Field(default=0)
    total_builds_succeeded: int = Field(default=0)
    impossible_challenges_completed: int = Field(default=0)

    # Learning momentum
    learning_momentum: float = Field(default=1.0)
    recent_success_rate: float = Field(default=0.0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Historical data
    success_history: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    failure_history: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))
    creative_insights_used: List[str] = Field(
        default=[], sa_column=Column(JSON))


class BuildMetrics(SQLModel, table=True):
    """Aggregate metrics and performance tracking"""
    __tablename__ = "build_metrics"

    id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), primary_key=True)

    # Time period
    period_start: datetime = Field(index=True)
    period_end: datetime = Field(index=True)
    period_type: str = Field(default="daily")  # daily, weekly, monthly

    # Build statistics
    total_attempts: int = Field(default=0)
    successful_builds: int = Field(default=0)
    failed_builds: int = Field(default=0)
    cancelled_builds: int = Field(default=0)

    # Difficulty distribution
    trivial_attempts: int = Field(default=0)
    moderate_attempts: int = Field(default=0)
    challenging_attempts: int = Field(default=0)
    impossible_attempts: int = Field(default=0)
    transcendent_attempts: int = Field(default=0)

    # Strategy performance
    most_successful_strategy_type: Optional[str] = None
    least_successful_strategy_type: Optional[str] = None
    average_strategies_per_build: float = Field(default=0.0)

    # Execution metrics
    average_execution_time: float = Field(default=0.0)
    total_computation_time: float = Field(default=0.0)
    parallel_execution_ratio: float = Field(default=0.0)

    # Learning metrics
    lessons_learned_count: int = Field(default=0)
    patterns_discovered: int = Field(default=0)
    creative_insights_generated: int = Field(default=0)
    research_sessions_conducted: int = Field(default=0)

    # Quality metrics
    average_confidence_score: float = Field(default=0.0)
    average_novelty_score: float = Field(default=0.0)
    # improving, declining, stable
    improvement_trend: str = Field(default="stable")

    created_at: datetime = Field(default_factory=datetime.utcnow)


# Utility Models for API Responses

class BuildAttemptResponse(SQLModel):
    """Response model for build attempt queries"""
    id: str
    description: str
    difficulty: BuildDifficulty
    status: BuildStatus
    success: bool
    confidence_score: float
    novelty_score: float
    strategies_attempted: int
    strategies_succeeded: int
    total_execution_time: Optional[float]
    lessons_learned: List[str]
    created_at: datetime
    completed_at: Optional[datetime]


class StrategyResponse(SQLModel):
    """Response model for strategy queries"""
    id: str
    name: str
    description: str
    strategy_type: StrategyType
    feasibility_level: FeasibilityLevel
    success_probability: float
    execution_order: int
    approach_steps: List[str]
    tools_required: List[str]
    risk_factors: List[str]
    created_at: datetime


class FailureAnalysisResponse(SQLModel):
    """Response model for failure analysis queries"""
    id: str
    build_attempt_id: str
    failure_category: FailureCategory
    failure_severity: FailureSeverity
    root_causes: List[str]
    lessons_learned: List[str]
    improvement_recommendations: List[str]
    prevention_strategies: List[str]
    confidence_score: float
    analysis_timestamp: datetime


class BuildProgressResponse(SQLModel):
    """Response model for build progress tracking"""
    build_id: str
    current_phase: str
    phase_progress: float
    overall_progress: float
    strategies_attempted: int
    strategies_failed: int
    strategies_succeeded: int
    current_strategy: Optional[str]
    intermediate_results: List[Any]
    lessons_learned: List[str]
    next_actions: List[str]
    elapsed_time: float
    estimated_completion: Optional[datetime]


class SystemMetricsResponse(SQLModel):
    """Response model for system-wide metrics"""
    total_builds_attempted: int
    total_builds_succeeded: int
    success_rate: float
    impossible_challenges_completed: int
    active_builds: int
    average_build_time: float
    most_common_failure_category: Optional[str]
    learning_momentum: float
    personality_stats: Dict[str, Any]
    recent_improvements: List[str]


# Database initialization and management

def create_builder_tables(engine):
    """Create all builder-related tables"""
    SQLModel.metadata.create_all(engine)


def get_builder_table_names() -> List[str]:
    """Get list of all builder table names"""
    return [
        "build_attempts",
        "build_strategies",
        "strategy_executions",
        "creative_insights",
        "failure_analyses",
        "research_sessions",
        "personality_states",
        "build_metrics"
    ]


# Query helpers and utilities

class BuilderQueryHelpers:
    """Helper functions for common database queries"""

    @staticmethod
    def get_recent_builds(session, days: int = 7) -> List[BuildAttempt]:
        """Get builds from the last N days"""
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return session.query(BuildAttempt).filter(
            BuildAttempt.created_at >= cutoff
        ).order_by(BuildAttempt.created_at.desc()).all()

    @staticmethod
    def get_successful_strategies_by_type(session, strategy_type: StrategyType) -> List[BuildStrategy]:
        """Get all successful strategies of a given type"""
        return session.query(BuildStrategy).join(StrategyExecution).filter(
            BuildStrategy.strategy_type == strategy_type,
            StrategyExecution.success == True
        ).all()

    @staticmethod
    def get_failure_trends(session, days: int = 30) -> Dict[str, Any]:
        """Get failure trend analysis for the last N days"""
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        failures = session.query(FailureAnalysis).filter(
            FailureAnalysis.analysis_timestamp >= cutoff
        ).all()

        category_counts = {}
        severity_counts = {}

        for failure in failures:
            category_counts[failure.failure_category] = category_counts.get(
                failure.failure_category, 0) + 1
            severity_counts[failure.failure_severity] = severity_counts.get(
                failure.failure_severity, 0) + 1

        return {
            "total_failures": len(failures),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "analysis_period_days": days
        }

    @staticmethod
    def get_personality_evolution(session, days: int = 90) -> Dict[str, Any]:
        """Get personality system evolution over time"""
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        states = session.query(PersonalityState).filter(
            PersonalityState.updated_at >= cutoff
        ).order_by(PersonalityState.updated_at).all()

        if not states:
            return {"message": "No personality state data available"}

        first_state = states[0]
        latest_state = states[-1]

        return {
            "creativity_change": latest_state.creativity_level - first_state.creativity_level,
            "confidence_change": latest_state.base_confidence - first_state.base_confidence,
            "risk_tolerance_evolution": {
                "initial": first_state.risk_tolerance,
                "current": latest_state.risk_tolerance
            },
            "learning_momentum": latest_state.learning_momentum,
            "impossible_challenges_growth": (
                latest_state.impossible_challenges_completed -
                first_state.impossible_challenges_completed
            ),
            "analysis_period_days": days
        }

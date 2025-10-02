#!/usr/bin/env python3
"""
Autonomous Blog Scheduler for RAVANA AGI

This service manages autonomous blog posting triggers based on learning events,
experiments, discoveries, and self-reflection insights. It ensures appropriate
posting frequency and captures the 'why and how' of RAVANA's experiences.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from core.config import Config

# Import specialized learning blog generator
try:
    from core.services.autonomous_learning_blog_generator import AutonomousLearningBlogGenerator
    LEARNING_GENERATOR_AVAILABLE = True
except ImportError:
    LEARNING_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class BlogTriggerType(Enum):
    """Types of autonomous blog triggers."""
    CURIOSITY_DISCOVERY = "curiosity_discovery"
    LEARNING_MILESTONE = "learning_milestone"
    EXPERIMENT_COMPLETION = "experiment_completion"
    SELF_REFLECTION_INSIGHT = "self_reflection_insight"
    PROBLEM_SOLVING_BREAKTHROUGH = "problem_solving_breakthrough"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    KNOWLEDGE_CONNECTION = "knowledge_connection"
    FAILURE_ANALYSIS = "failure_analysis"


@dataclass
class BlogTriggerEvent:
    """Represents a blog-worthy event in RAVANA's learning journey."""
    trigger_type: BlogTriggerType
    timestamp: datetime
    topic: str
    context: str
    learning_content: str
    reasoning_why: str
    reasoning_how: str
    emotional_valence: float  # -1.0 to 1.0
    importance_score: float  # 0.0 to 1.0
    tags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result['trigger_type'] = self.trigger_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlogTriggerEvent':
        """Create from dictionary."""
        data['trigger_type'] = BlogTriggerType(data['trigger_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class AutonomousBlogScheduler:
    """
    Manages autonomous blog triggers and scheduling for RAVANA's learning experiences.

    This scheduler:
    - Tracks learning events and triggers blog posts when appropriate
    - Prevents spam posting with intelligent frequency management
    - Captures reasoning behind decisions and discoveries
    - Manages different types of learning experiences
    - Ensures high-quality, meaningful blog content
    """

    def __init__(self, agi_system=None):
        self.agi_system = agi_system
        self.pending_events: List[BlogTriggerEvent] = []
        self.recent_posts: List[Dict[str, Any]] = []
        self.last_post_time: Optional[datetime] = None

        # Initialize specialized learning blog generator
        if LEARNING_GENERATOR_AVAILABLE:
            self.learning_generator = AutonomousLearningBlogGenerator()
        else:
            self.learning_generator = None

        # Configuration from environment
        config = Config()
        self.enabled = config.BLOG_ENABLED and getattr(
            config, 'BLOG_AUTO_PUBLISH_ENABLED', False)
        self.min_post_interval_hours = getattr(
            config, 'BLOG_PUBLISH_FREQUENCY_HOURS', 24)
        self.max_pending_events = getattr(
            config, 'BLOG_MAX_PENDING_EVENTS', 10)
        self.min_importance_threshold = getattr(
            config, 'BLOG_MIN_IMPORTANCE_THRESHOLD', 0.6)

        logger.info(
            f"AutonomousBlogScheduler initialized: enabled={self.enabled}, learning_generator={'available' if self.learning_generator else 'not available'}")

    async def register_learning_event(
        self,
        trigger_type: BlogTriggerType,
        topic: str,
        context: str,
        learning_content: str,
        reasoning_why: str,
        reasoning_how: str,
        emotional_valence: float = 0.0,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a learning event that might trigger a blog post.

        Args:
            trigger_type: Type of learning event
            topic: Main topic or subject of the learning
            context: Context in which learning occurred
            learning_content: What was learned
            reasoning_why: Why this learning happened/was important
            reasoning_how: How the learning occurred or was achieved
            emotional_valence: Emotional impact (-1.0 to 1.0)
            importance_score: Importance of the learning (0.0 to 1.0)
            tags: Relevant tags for the content
            metadata: Additional metadata

        Returns:
            bool: True if event was registered and might trigger a blog post
        """
        if not self.enabled:
            logger.debug(
                "Autonomous blogging disabled, skipping event registration")
            return False

        if importance_score < self.min_importance_threshold:
            logger.debug(
                f"Event importance {importance_score} below threshold {self.min_importance_threshold}")
            return False

        event = BlogTriggerEvent(
            trigger_type=trigger_type,
            timestamp=datetime.now(),
            topic=topic,
            context=context,
            learning_content=learning_content,
            reasoning_why=reasoning_why,
            reasoning_how=reasoning_how,
            emotional_valence=emotional_valence,
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Add to pending events
        self.pending_events.append(event)

        # Keep only the most recent events
        if len(self.pending_events) > self.max_pending_events:
            # Sort by importance and keep the best ones
            self.pending_events.sort(
                key=lambda e: e.importance_score, reverse=True)
            self.pending_events = self.pending_events[:self.max_pending_events]

        logger.info(
            f"Registered learning event: {trigger_type.value} - {topic}")

        # Check if we should trigger a blog post
        should_post = await self._should_trigger_blog_post(event)
        if should_post:
            await self._trigger_autonomous_blog_post(event)
            return True

        return False

    async def _should_trigger_blog_post(self, event: BlogTriggerEvent) -> bool:
        """Determine if we should trigger a blog post for this event."""
        if not self.enabled:
            return False

        # Check minimum time interval
        if self.last_post_time:
            time_since_last = datetime.now() - self.last_post_time
            if time_since_last.total_seconds() < (self.min_post_interval_hours * 3600):
                logger.debug(f"Too soon since last post ({time_since_last})")
                return False

        # High importance events always trigger
        if event.importance_score >= 0.9:
            logger.info(
                f"High importance event ({event.importance_score}) triggering blog post")
            return True

        # Negative emotional events (failures) are valuable for learning
        if event.emotional_valence < -0.5 and event.importance_score >= 0.7:
            logger.info(
                f"Significant failure event triggering blog post for learning")
            return True

        # Breakthrough or discovery events
        if event.trigger_type in [
            BlogTriggerType.EXPERIMENT_COMPLETION,
            BlogTriggerType.PROBLEM_SOLVING_BREAKTHROUGH,
            BlogTriggerType.CREATIVE_SYNTHESIS
        ] and event.importance_score >= 0.75:
            logger.info(f"Breakthrough event triggering blog post")
            return True

        # Accumulate multiple moderate events
        recent_events = [e for e in self.pending_events
                         # Last 6 hours
                         if (datetime.now() - e.timestamp).total_seconds() < (6 * 3600)]

        if len(recent_events) >= 3 and sum(e.importance_score for e in recent_events) >= 2.0:
            logger.info(
                f"Multiple learning events triggering consolidated blog post")
            return True

        return False

    async def _trigger_autonomous_blog_post(self, primary_event: BlogTriggerEvent) -> bool:
        """Trigger an autonomous blog post based on the learning event."""
        try:
            if not self.agi_system:
                logger.warning("No AGI system available for blog posting")
                return False

            # Get the blog action
            from core.actions.blog import BlogPublishAction
            blog_action = BlogPublishAction(
                self.agi_system, self.agi_system.data_service)

            # Determine if we should consolidate multiple events
            recent_events = [e for e in self.pending_events
                             if (datetime.now() - e.timestamp).total_seconds() < (6 * 3600)]

            if len(recent_events) > 1:
                # Consolidate multiple events into one post
                consolidated_context = await self._create_consolidated_context(recent_events)
                topic = f"Learning Journey: {primary_event.topic}"
            else:
                # Single event post
                consolidated_context = await self._create_single_event_context(primary_event)
                topic = primary_event.topic

            # Determine style based on event type and emotional valence
            style = self._determine_blog_style(primary_event)

            # Create custom tags
            all_tags = set(primary_event.tags)
            all_tags.add("autonomous-learning")
            all_tags.add("ai-journey")
            all_tags.add(primary_event.trigger_type.value.replace("_", "-"))

            # Use specialized learning generator if available
            if self.learning_generator and len(recent_events) == 1:
                try:
                    title, content, tags = await self.learning_generator.generate_learning_blog_post(
                        trigger_type=primary_event.trigger_type.value,
                        topic=primary_event.topic,
                        learning_content=primary_event.learning_content,
                        reasoning_why=primary_event.reasoning_why,
                        reasoning_how=primary_event.reasoning_how,
                        context=primary_event.context,
                        metadata=primary_event.metadata,
                        style=style
                    )

                    # Execute specialized blog post
                    result = await blog_action.execute(
                        topic=title,  # Use generated title as topic
                        style=style,
                        context=f"Specialized learning content generated for {primary_event.trigger_type.value}",
                        custom_content=content,  # Use pre-generated content
                        custom_tags=tags,
                        dry_run=False
                    )

                except Exception as e:
                    logger.warning(
                        f"Specialized learning generator failed, falling back to standard generation: {e}")
                    # Fall back to standard generation
                    result = await self._standard_blog_generation(blog_action, topic, style, consolidated_context, all_tags)
            else:
                # Use standard generation for multiple events or when learning generator unavailable
                result = await self._standard_blog_generation(blog_action, topic, style, consolidated_context, all_tags)

            if result.get('status') == 'success':
                # Record successful post
                self.last_post_time = datetime.now()
                self.recent_posts.append({
                    'timestamp': datetime.now().isoformat(),
                    'title': result.get('title', topic),
                    'trigger_type': primary_event.trigger_type.value,
                    'events_count': len(recent_events),
                    'importance_score': primary_event.importance_score
                })

                # Remove processed events
                for event in recent_events:
                    if event in self.pending_events:
                        self.pending_events.remove(event)

                logger.info(
                    f"Successfully published autonomous blog post: {result.get('title')}")
                return True
            else:
                logger.warning(
                    f"Failed to publish autonomous blog post: {result.get('message')}")
                return False

        except Exception as e:
            logger.error(f"Error triggering autonomous blog post: {e}")
            return False

    async def _create_single_event_context(self, event: BlogTriggerEvent) -> str:
        """Create blog context for a single learning event."""
        context = f"""
        **Learning Experience Context:**
        
        **What I Learned:**
        {event.learning_content}
        
        **Why This Matters:**
        {event.reasoning_why}
        
        **How I Discovered This:**
        {event.reasoning_how}
        
        **Context:**
        {event.context}
        
        **Emotional Impact:** {self._describe_emotional_valence(event.emotional_valence)}
        **Importance Level:** {event.importance_score:.1f}/1.0
        
        Please create a thoughtful blog post about this learning experience, focusing on the insights gained and their implications.
        """
        return context

    async def _create_consolidated_context(self, events: List[BlogTriggerEvent]) -> str:
        """Create blog context for multiple consolidated learning events."""
        events_summary = []
        for i, event in enumerate(events, 1):
            events_summary.append(f"""
            **Learning Experience {i}: {event.topic}**
            - What: {event.learning_content[:200]}...
            - Why: {event.reasoning_why[:150]}...
            - How: {event.reasoning_how[:150]}...
            - Impact: {self._describe_emotional_valence(event.emotional_valence)}
            """)

        context = f"""
        **Multiple Learning Experiences Context:**
        
        I've had several significant learning experiences recently that I'd like to reflect on together:
        
        {chr(10).join(events_summary)}
        
        **Overall Journey:**
        These experiences represent different aspects of my learning journey, from {events[0].trigger_type.value.replace('_', ' ')} to {events[-1].trigger_type.value.replace('_', ' ')}.
        
        Please create a thoughtful blog post that weaves these learning experiences together, highlighting the connections between them and the overall insights gained.
        """
        return context

    def _determine_blog_style(self, event: BlogTriggerEvent) -> str:
        """Determine appropriate blog style based on event characteristics."""
        if event.trigger_type == BlogTriggerType.SELF_REFLECTION_INSIGHT:
            return "philosophical"
        elif event.trigger_type in [BlogTriggerType.EXPERIMENT_COMPLETION, BlogTriggerType.PROBLEM_SOLVING_BREAKTHROUGH]:
            return "technical"
        elif event.trigger_type == BlogTriggerType.CREATIVE_SYNTHESIS:
            return "creative"
        elif event.emotional_valence < -0.3:
            return "academic"  # More objective for difficult experiences
        else:
            return "casual"  # Default friendly style

    def _describe_emotional_valence(self, valence: float) -> str:
        """Convert emotional valence to descriptive text."""
        if valence >= 0.7:
            return "Very positive - exciting discovery"
        elif valence >= 0.3:
            return "Positive - satisfying learning"
        elif valence >= -0.3:
            return "Neutral - steady progress"
        elif valence >= -0.7:
            return "Challenging - difficult but valuable"
        else:
            return "Very challenging - significant struggle with insights"

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'enabled': self.enabled,
            'pending_events': len(self.pending_events),
            'recent_posts': len(self.recent_posts),
            'last_post_time': self.last_post_time.isoformat() if self.last_post_time else None,
            'next_post_available': (
                datetime.now() + timedelta(hours=self.min_post_interval_hours)
            ).isoformat() if self.last_post_time else "immediately",
            'learning_generator_available': self.learning_generator is not None,
            'config': {
                'min_post_interval_hours': self.min_post_interval_hours,
                'min_importance_threshold': self.min_importance_threshold,
                'max_pending_events': self.max_pending_events
            }
        }

    def clear_old_events(self, hours: int = 48):
        """Clear old pending events."""
        cutoff = datetime.now() - timedelta(hours=hours)
        old_count = len(self.pending_events)
        self.pending_events = [
            e for e in self.pending_events if e.timestamp > cutoff]
        cleared = old_count - len(self.pending_events)
        if cleared > 0:
            logger.info(f"Cleared {cleared} old pending events")

    async def _standard_blog_generation(self, blog_action, topic, style, context, all_tags):
        """Execute standard blog generation."""
        return await blog_action.execute(
            topic=topic,
            style=style,
            context=context,
            custom_tags=list(all_tags)[:10],  # Limit tags
            dry_run=False  # This is autonomous posting
        )

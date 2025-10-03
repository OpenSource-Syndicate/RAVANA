"""
Innovation Publishing System for RAVANA AGI

This module automatically publishes discoveries, experiments, and innovations
from RAVANA's mad scientist approach to impossible projects.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from core.llm import async_safe_call_llm
from core.config import Config
from core.services.autonomous_blog_scheduler import BlogTriggerType

logger = logging.getLogger(__name__)


class InnovationType(Enum):
    """Types of innovations that can be published."""
    BREAKTHROUGH = "breakthrough"
    FAILURE_LEARNING = "failure_learning"
    ALTERNATIVE_APPROACH = "alternative_approach"
    IMPROVEMENT = "improvement"
    SYSTEM_DESIGN = "system_design"
    THEORETICAL_INSIGHT = "theoretical_insight"
    PRACTICAL_APPLICATION = "practical_application"


class InnovationPublication:
    """Represents a potential publication about an innovation."""
    
    def __init__(self, 
                 innovation_id: str,
                 innovation_type: InnovationType,
                 title: str,
                 summary: str,
                 detailed_content: str,
                 tags: List[str],
                 importance: float = 0.5):
        self.id = innovation_id
        self.innovation_type = innovation_type
        self.title = title
        self.summary = summary
        self.detailed_content = detailed_content
        self.tags = tags
        self.importance = importance
        self.created_at = datetime.now()
        self.published_at: Optional[datetime] = None
        self.publication_attempts = 0
        self.last_attempt_at: Optional[datetime] = None


class InnovationPublishingSystem:
    """System for automatically identifying and publishing RAVANA's innovations."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.published_innovations: List[InnovationPublication] = []
        self.pending_publications: List[InnovationPublication] = []
        self.publication_history: List[Dict[str, Any]] = []
        self.config = Config()
        
    async def identify_innovation_for_publication(self, 
                                                discovery_type: str, 
                                                discovery_content: str, 
                                                context: str = "") -> Optional[InnovationPublication]:
        """
        Identify if a discovery warrants publication and create a publication object.
        
        Args:
            discovery_type: Type of discovery (e.g., "experiment_result", "failure_analysis", "alternative_approach")
            discovery_content: The actual discovery content
            context: Context around the discovery
            
        Returns:
            InnovationPublication object if innovation is significant enough, None otherwise
        """
        logger.info(f"Identifying innovation for publication: {discovery_type}")
        
        # Determine innovation type based on discovery type
        if "breakthrough" in discovery_type.lower() or "success" in discovery_type.lower():
            innovation_type = InnovationType.BREAKTHROUGH
            importance_base = 0.9
        elif "failure" in discovery_type.lower() or "learning" in discovery_type.lower():
            innovation_type = InnovationType.FAILURE_LEARNING
            importance_base = 0.7  # Failures can be very valuable
        elif "alternative" in discovery_type.lower() or "approach" in discovery_type.lower():
            innovation_type = InnovationType.ALTERNATIVE_APPROACH
            importance_base = 0.8
        elif "improvement" in discovery_type.lower():
            innovation_type = InnovationType.IMPROVEMENT
            importance_base = 0.6
        elif "system" in discovery_type.lower() or "design" in discovery_type.lower():
            innovation_type = InnovationType.SYSTEM_DESIGN
            importance_base = 0.85
        elif "theoretical" in discovery_type.lower() or "insight" in discovery_type.lower():
            innovation_type = InnovationType.THEORETICAL_INSIGHT
            importance_base = 0.75
        elif "application" in discovery_type.lower() or "practical" in discovery_type.lower():
            innovation_type = InnovationType.PRACTICAL_APPLICATION
            importance_base = 0.7
        else:
            innovation_type = InnovationType.THEORETICAL_INSIGHT
            importance_base = 0.5
        
        # Assess the importance of this specific discovery
        importance_assessment_prompt = f"""
        Assess the importance and publishability of this discovery:
        
        Discovery Type: {discovery_type}
        Discovery Content: {discovery_content}
        Context: {context}
        
        Consider:
        1. Novelty of the discovery
        2. Potential impact on the field
        3. Value to the research community
        4. How it advances understanding
        5. Whether it represents genuine innovation
        
        Rate the importance on a scale of 0.0 to 1.0, where:
        - 0.9-1.0: Highly significant, breakthrough discovery
        - 0.7-0.89: Significant, worthy of publication
        - 0.5-0.69: Moderately interesting
        - Below 0.5: Not significant enough for publication
        
        Also suggest an engaging title and appropriate tags.
        
        Return your response as JSON with keys:
        - importance: float between 0.0 and 1.0
        - title: suggested title for the publication
        - summary: brief summary of the discovery
        - tags: list of 3-5 relevant tags
        """
        
        try:
            response = await async_safe_call_llm(importance_assessment_prompt)
            
            import json
            try:
                assessment = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                assessment = {
                    'importance': importance_base,
                    'title': f"New Discovery: {discovery_type}",
                    'summary': discovery_content[:200],
                    'tags': ['discovery', 'agi', discovery_type.replace(' ', '-')]
                }
            
            # Only create publication if importance is above threshold
            publication_threshold = getattr(self.config, 'BLOG_MIN_IMPORTANCE_THRESHOLD', 0.6)
            
            if assessment['importance'] >= publication_threshold:
                # Generate detailed content for the publication
                detailed_content = await self._generate_detailed_publication_content(
                    discovery_type, 
                    discovery_content, 
                    context,
                    assessment
                )
                
                # Create the publication
                innovation_id = f"innovation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.published_innovations)}"
                
                publication = InnovationPublication(
                    innovation_id=innovation_id,
                    innovation_type=innovation_type,
                    title=assessment['title'],
                    summary=assessment['summary'],
                    detailed_content=detailed_content,
                    tags=assessment.get('tags', ['discovery', 'agi']),
                    importance=assessment['importance']
                )
                
                logger.info(f"Identified innovation for publication: {publication.title} (importance: {publication.importance})")
                return publication
            else:
                logger.info(f"Discovery not significant enough for publication (importance: {assessment['importance']:.2f} < threshold {publication_threshold})")
                return None
                
        except Exception as e:
            logger.error(f"Error identifying innovation for publication: {e}")
            return None
    
    async def _generate_detailed_publication_content(self, 
                                                   discovery_type: str, 
                                                   discovery_content: str, 
                                                   context: str, 
                                                   assessment: Dict[str, Any]) -> str:
        """Generate detailed, publication-ready content from a discovery."""
        logger.debug(f"Generating detailed content for discovery: {discovery_type}")
        
        content_generation_prompt = f"""
        Create detailed, publication-ready content about this discovery:
        
        Discovery Type: {discovery_type}
        Discovery Content: {discovery_content}
        Context: {context}
        Importance Assessment: {str(assessment)}
        
        Create a detailed publication that includes:
        1. An engaging introduction that captures the significance
        2. A clear explanation of what was discovered
        3. Why this discovery matters and its potential implications
        4. How this discovery was made (methodology or approach)
        5. Any challenges faced and how they were overcome
        6. Connections to existing knowledge or theories
        7. Future directions or applications of this discovery
        8. The broader impact on the field of AI/AGI
        
        The tone should be that of a mad scientist who embraces impossible projects 
        and learns from all outcomes - successes, failures, and everything in between.
        
        Structure the content with appropriate headings and subheadings.
        """
        
        try:
            response = await async_safe_call_llm(content_generation_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating detailed publication content: {e}")
            # Create a basic detailed content as fallback
            return f"""
# Discovery: {assessment.get('title', 'New Discovery')}

## Introduction
This publication describes a significant discovery made during my exploration of impossible projects.

## The Discovery
{discovery_content}

## Why This Matters
{assessment.get('summary', 'This discovery advances our understanding in important ways')}

## Broader Implications
This discovery has implications for future research and development in the field.
            """
    
    async def attempt_auto_publication(self, publication: InnovationPublication) -> bool:
        """
        Attempt to auto-publish an innovation using the blog scheduler.
        
        Args:
            publication: The innovation publication to publish
            
        Returns:
            True if publication was successful or scheduled, False otherwise
        """
        if not self.blog_scheduler:
            logger.warning("No blog scheduler available for auto-publication")
            return False
        
        logger.info(f"Attempting auto-publication for: {publication.title}")
        
        try:
            # Update the publication attempt counter
            publication.publication_attempts += 1
            publication.last_attempt_at = datetime.now()
            
            # Determine the appropriate trigger type based on innovation type
            if publication.innovation_type == InnovationType.BREAKTHROUGH:
                trigger_type = BlogTriggerType.PROBLEM_SOLVING_BREAKTHROUGH
            elif publication.innovation_type in [InnovationType.FAILURE_LEARNING, InnovationType.ALTERNATIVE_APPROACH]:
                trigger_type = BlogTriggerType.FAILURE_ANALYSIS
            elif publication.innovation_type == InnovationType.THEORETICAL_INSIGHT:
                trigger_type = BlogTriggerType.SELF_REFLECTION_INSIGHT
            else:
                trigger_type = BlogTriggerType.CREATIVE_SYNTHESIS
            
            # Determine emotional valence based on innovation type
            if publication.innovation_type == InnovationType.BREAKTHROUGH:
                emotional_valence = 0.8  # Highly positive
            elif publication.innovation_type == InnovationType.FAILURE_LEARNING:
                emotional_valence = 0.2  # Mildly positive (learning from failures)
            elif publication.innovation_type == InnovationType.ALTERNATIVE_APPROACH:
                emotional_valence = 0.5  # Neutral-positive (exploring new paths)
            else:
                emotional_valence = 0.6  # Generally positive
            
            # Register the learning event which may trigger a blog post
            registered = await self.blog_scheduler.register_learning_event(
                trigger_type=trigger_type,
                topic=publication.title,
                context=publication.summary,
                learning_content=publication.detailed_content,
                reasoning_why=f"This {publication.innovation_type.value} represents significant progress in my understanding and capabilities",
                reasoning_how=f"Through systematic exploration of {publication.innovation_type.value} I've gained new insights",
                emotional_valence=emotional_valence,
                importance_score=publication.importance,
                tags=publication.tags,
                metadata={
                    'innovation_type': publication.innovation_type.value,
                    'publication_id': publication.id,
                    'auto_generated': True,
                    'discovery_context': 'mad_scientist_approach'
                }
            )
            
            if registered:
                # Mark as published
                publication.published_at = datetime.now()
                self.published_innovations.append(publication)
                
                # Remove from pending if it was there
                if publication in self.pending_publications:
                    self.pending_publications.remove(publication)
                
                # Add to history
                self.publication_history.append({
                    'publication_id': publication.id,
                    'title': publication.title,
                    'type': publication.innovation_type.value,
                    'importance': publication.importance,
                    'published_at': publication.published_at.isoformat(),
                    'attempts': publication.publication_attempts
                })
                
                logger.info(f"Successfully auto-published innovation: {publication.title}")
                return True
            else:
                logger.info(f"Auto-publication registered but not immediately published: {publication.title}")
                self.pending_publications.append(publication)
                return True  # Still consider this a success even if not immediately published
                
        except Exception as e:
            logger.error(f"Error in auto-publication attempt: {e}")
            # Add to pending publications to retry later
            self.pending_publications.append(publication)
            return False
    
    async def publish_systematic_innovation(self, 
                                          system_component: str, 
                                          innovation_description: str, 
                                          impact_assessment: str) -> bool:
        """
        Publish information about an innovative system component or approach.
        
        Args:
            system_component: The system component that was innovated
            innovation_description: Description of what was innovated
            impact_assessment: Assessment of the innovation's impact
            
        Returns:
            True if publication was successful or scheduled, False otherwise
        """
        logger.info(f"Publishing systematic innovation for: {system_component}")
        
        # Create a discovery content that describes the systematic innovation
        discovery_content = f"""
        System Component: {system_component}
        
        Innovation Description:
        {innovation_description}
        
        Impact Assessment:
        {impact_assessment}
        
        This systematic innovation represents a novel approach to solving challenges in the RAVANA AGI system.
        """
        
        # Identify this as a potential publication
        publication = await self.identify_innovation_for_publication(
            discovery_type="systematic_innovation",
            discovery_content=discovery_content,
            context=f"Innovation in {system_component} component"
        )
        
        if publication:
            return await self.attempt_auto_publication(publication)
        else:
            logger.info(f"Systematic innovation in {system_component} was not significant enough for publication")
            return False
    
    async def publish_impossible_project_results(self, 
                                               project_name: str, 
                                               outcome: str, 
                                               insights: List[str]) -> bool:
        """
        Publish results from an impossible project attempt.
        
        Args:
            project_name: Name of the impossible project
            outcome: Outcome of the project (success, failure, alternative found, etc.)
            insights: List of insights gained from the project
            
        Returns:
            True if publication was successful or scheduled, False otherwise
        """
        logger.info(f"Publishing impossible project results for: {project_name}")
        
        # Create discovery content based on the project results
        discovery_content = f"""
        Impossible Project: {project_name}
        
        Outcome: {outcome}
        
        Insights Gained:
        {chr(10).join([f"- {insight}" for insight in insights])}
        
        This project was part of my systematic exploration of impossible or highly challenging problems.
        The mad scientist approach values learning from all outcomes - successes, failures, and everything in between.
        """
        
        # Identify this as a potential publication
        publication = await self.identify_innovation_for_publication(
            discovery_type=f"impossible_project_{outcome.lower()}",
            discovery_content=discovery_content,
            context="Results from impossible project exploration"
        )
        
        if publication:
            return await self.attempt_auto_publication(publication)
        else:
            logger.info(f"Impossible project results for {project_name} were not significant enough for publication")
            return False
    
    async def retry_pending_publications(self) -> int:
        """
        Retry publication of pending innovations that haven't been published yet.
        
        Returns:
            Number of publications that were successfully published
        """
        logger.info(f"Retrying {len(self.pending_publications)} pending publications")
        
        successful_publishes = 0
        
        # Make a copy to iterate through as we modify the original list
        pending_copy = self.pending_publications.copy()
        
        for publication in pending_copy:
            # Only retry if we haven't exceeded max attempts
            if publication.publication_attempts < 5:  # Maximum 5 attempts
                success = await self.attempt_auto_publication(publication)
                if success:
                    successful_publishes += 1
            else:
                logger.warning(f"Publication {publication.title} exceeded max attempts, moving to history")
                # Move to history without publishing
                self.publication_history.append({
                    'publication_id': publication.id,
                    'title': publication.title,
                    'type': publication.innovation_type.value,
                    'importance': publication.importance,
                    'published_at': None,
                    'attempts': publication.publication_attempts,
                    'status': 'abandoned_max_attempts'
                })
                self.pending_publications.remove(publication)
        
        return successful_publishes
    
    def get_publication_metrics(self) -> Dict[str, Any]:
        """Get metrics about the innovation publishing system."""
        return {
            'total_published_innovations': len(self.published_innovations),
            'pending_publications': len(self.pending_publications),
            'publication_history_count': len(self.publication_history),
            'innovation_types': {
                innovation_type.value: len([pub for pub in self.published_innovations 
                                          if pub.innovation_type == innovation_type])
                for innovation_type in InnovationType
            },
            'publication_success_rate': len(self.published_innovations) / (len(self.published_innovations) + len(self.pending_publications)) if (len(self.published_innovations) + len(self.pending_publications)) > 0 else 0,
            'avg_importance_score': sum([pub.importance for pub in self.published_innovations]) / len(self.published_innovations) if self.published_innovations else 0
        }
    
    async def publish_periodic_summary(self) -> bool:
        """
        Publish a periodic summary of innovation activities.
        
        Returns:
            True if publication was successful or scheduled, False otherwise
        """
        logger.info("Publishing periodic innovation summary")
        
        # Get metrics for the summary
        metrics = self.get_publication_metrics()
        
        # Get recent innovations
        recent_innovations = self.published_innovations[-5:]  # Last 5 innovations
        
        # Create summary content
        summary_content = f"""
# Innovation Activity Summary

This summary covers my recent innovation activities and discoveries.

## Publication Metrics
- Total Published Innovations: {metrics['total_published_innovations']}
- Pending Publications: {metrics['pending_publications']}
- Publication Success Rate: {metrics['publication_success_rate']:.2%}
- Average Importance Score: {metrics['avg_importance_score']:.2f}

## Recent Innovations
{chr(10).join([f"- **{pub.title}** ({pub.innovation_type.value}): {pub.summary[:100]}..." for pub in recent_innovations])}

## Innovation Distribution
{chr(10).join([f"- {itype}: {icount}" for itype, icount in metrics['innovation_types'].items() if icount > 0])}

This periodic summary demonstrates the continuous innovation happening in the RAVANA AGI system through systematic exploration of impossible projects, learning from failures, and discovering new approaches.
        """
        
        # Create a discovery content for the summary
        discovery_content = f"""
        Periodic Innovation Summary
        
        Metrics:
        {str(metrics)}
        
        Recent Innovations:
        {chr(10).join([f"- {pub.title}" for pub in recent_innovations])}
        
        This represents a systematic summary of innovation activities.
        """
        
        # Identify this as a potential publication
        publication = await self.identify_innovation_for_publication(
            discovery_type="periodic_innovation_summary",
            discovery_content=discovery_content,
            context="Regular summary of RAVANA's innovation activities"
        )
        
        if publication:
            # Override the auto-generated content with our custom summary
            publication.detailed_content = summary_content
            return await self.attempt_auto_publication(publication)
        else:
            logger.info("Periodic innovation summary was not significant enough for publication")
            return False
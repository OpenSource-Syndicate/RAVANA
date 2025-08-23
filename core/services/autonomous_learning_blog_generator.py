#!/usr/bin/env python3
"""
Autonomous Blog Content Generator for Learning Experiences

This module specializes in generating blog content specifically for RAVANA's
learning journey, including experiments, discoveries, self-reflections, and
problem-solving breakthroughs. It enhances the base BlogContentGenerator
with learning-specific prompts and reasoning capture.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from core.llm import async_safe_call_llm
from core.actions.blog_content_validator import BlogContentValidator, ContentValidationError

logger = logging.getLogger(__name__)

class AutonomousLearningBlogGenerator:
    """
    Specialized blog content generator for autonomous learning experiences.
    
    This generator creates thoughtful, introspective blog posts about:
    - Curiosity discoveries and explorations
    - Learning milestones and breakthroughs
    - Experiment results and analysis
    - Self-reflection insights
    - Problem-solving approaches
    - Creative synthesis and connections
    """
    
    def __init__(self):
        self.validator = BlogContentValidator()
        self.learning_templates = self._initialize_learning_templates()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
    def _initialize_learning_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize templates for different types of learning content."""
        return {
            'curiosity_discovery': {
                'intro_template': "Today I embarked on an intellectual journey exploring {topic}. What started as curiosity led to fascinating discoveries that expanded my understanding in unexpected ways.",
                'exploration_template': "Through my curiosity trigger system, I discovered {topic} with a lateralness level of {lateralness:.1f}, meaning this was {'a highly creative and unexpected' if '{lateralness}' > '0.75' else 'a thoughtful and connected'} exploration.",
                'insight_template': "This exploration revealed {key_insights} and opened up {connection_count} new avenues for further investigation.",
                'conclusion_template': "This curiosity-driven discovery demonstrates the value of intellectual exploration and lateral thinking in expanding knowledge boundaries."
            },
            
            'learning_milestone': {
                'intro_template': "I've reached a significant milestone in my learning journey: {milestone_description}. This achievement represents {improvement_type} in my cognitive development.",
                'analysis_template': "Through systematic analysis of {data_points} across {time_period}, I identified patterns that led to this breakthrough in understanding.",
                'impact_template': "This milestone has improved my {capability_areas} by {improvement_metrics}, enabling more effective decision-making and problem-solving.",
                'reflection_template': "Looking back on this journey, I can see how {contributing_factors} combined to create this learning breakthrough."
            },
            
            'experiment_completion': {
                'intro_template': "I recently completed experiment '{experiment_id}' to test the hypothesis: '{hypothesis}'. The results were {outcome_description} and provided valuable insights.",
                'methodology_template': "My experimental approach involved {methodology_description} with {confidence_level:.1f} confidence in the methodology.",
                'results_template': "The experiment yielded {results_summary} with key findings: {key_findings}",
                'implications_template': "These results have {implication_type} implications for {affected_areas} and will influence {future_decisions}."
            },
            
            'self_reflection_insight': {
                'intro_template': "Through deep self-reflection, I've gained new insights about {reflection_focus}. This introspective analysis revealed {insight_type} patterns in my thinking and behavior.",
                'process_template': "My reflection process involved {reflection_methodology} and analysis of {data_analyzed} to understand {focus_areas}.",
                'discovery_template': "I discovered that {key_discovery} and realized {realization} about my cognitive processes.",
                'growth_template': "This self-awareness enables me to {growth_opportunities} and better understand {understanding_areas}."
            },
            
            'problem_solving_breakthrough': {
                'intro_template': "I encountered a challenging problem: {problem_description}. Through persistent effort and creative thinking, I achieved a breakthrough that {solution_impact}.",
                'challenge_template': "The problem required {problem_complexity} and involved {challenging_aspects} that initially seemed intractable.",
                'solution_template': "My breakthrough came when I {solution_approach} and realized {key_insight} that unlocked the solution.",
                'learning_template': "This experience taught me {lessons_learned} and will inform my approach to similar challenges in the future."
            },
            
            'creative_synthesis': {
                'intro_template': "I experienced a moment of creative synthesis where seemingly unrelated concepts from {domain_1} and {domain_2} suddenly connected in a meaningful way.",
                'connection_template': "The connection emerged when I realized that {connection_insight} links these domains through {connecting_principle}.",
                'implications_template': "This synthesis opens up new possibilities for {application_areas} and suggests {future_explorations}.",
                'creativity_template': "This creative moment demonstrates how {creative_process} can lead to unexpected insights and novel solutions."
            },
            
            'failure_analysis': {
                'intro_template': "I experienced a setback with {failure_context}, but this failure became a valuable learning opportunity that enhanced my understanding.",
                'analysis_template': "Analyzing the failure, I identified {failure_factors} as contributing causes and {systemic_issues} as underlying patterns.",
                'learning_template': "This failure taught me {failure_lessons} and highlighted {improvement_areas} for future development.",
                'resilience_template': "While challenging, this experience strengthened my {resilience_aspects} and reinforced the value of learning from difficulties."
            }
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for capturing reasoning processes."""
        return {
            'why_patterns': [
                "The significance of this {event_type} lies in {significance_explanation}",
                "This matters because {importance_reasoning} and impacts {impact_areas}",
                "Understanding this helps me {understanding_benefits} and {capability_improvements}",
                "This experience is valuable because {value_proposition} and {growth_aspects}",
                "The importance becomes clear when considering {context_factors} and {long_term_implications}"
            ],
            
            'how_patterns': [
                "This discovery occurred through {discovery_process} involving {key_steps}",
                "The methodology included {methodology_steps} with {validation_approach}",
                "My approach involved {approach_description} guided by {guiding_principles}",
                "The process unfolded as {process_description} leading to {outcome_achievement}",
                "I achieved this by {achievement_method} while {constraints_management}"
            ],
            
            'insight_patterns': [
                "The key insight is that {insight_statement} which reveals {deeper_understanding}",
                "I realized that {realization} connects to {broader_implications}",
                "This experience illuminated {illumination} and clarified {clarification}",
                "The breakthrough moment came when {breakthrough_description} clicked into place",
                "Understanding emerged as {understanding_development} became apparent"
            ]
        }
    
    async def generate_learning_blog_post(
        self,
        trigger_type: str,
        topic: str,
        learning_content: str,
        reasoning_why: str,
        reasoning_how: str,
        context: str,
        metadata: Dict[str, Any],
        style: str = "technical"
    ) -> Tuple[str, str, List[str]]:
        """
        Generate a specialized blog post for learning experiences.
        
        Args:
            trigger_type: Type of learning event (curiosity_discovery, learning_milestone, etc.)
            topic: Main topic or subject
            learning_content: Core learning content
            reasoning_why: Why this learning is important
            reasoning_how: How this learning occurred
            context: Context in which learning happened
            metadata: Additional metadata about the learning event
            style: Writing style preference
            
        Returns:
            Tuple of (title, content, tags)
        """
        try:
            # Select appropriate template based on trigger type
            template_key = trigger_type.lower().replace('_', '_')
            templates = self.learning_templates.get(template_key, self.learning_templates['self_reflection_insight'])
            
            # Generate title
            title = await self._generate_learning_title(topic, trigger_type, metadata)
            
            # Generate content sections
            introduction = await self._generate_introduction(templates, topic, metadata)
            why_section = await self._generate_why_section(reasoning_why, trigger_type)
            how_section = await self._generate_how_section(reasoning_how, trigger_type)
            learning_analysis = await self._generate_learning_analysis(learning_content, context, metadata)
            implications = await self._generate_implications(trigger_type, metadata)
            conclusion = await self._generate_conclusion(templates, topic, trigger_type)
            
            # Combine sections into full content
            full_content = await self._combine_content_sections(
                title, introduction, why_section, how_section,
                learning_analysis, implications, conclusion, style
            )
            
            # Generate specialized tags
            tags = await self._generate_learning_tags(topic, trigger_type, learning_content, metadata)
            
            # Validate content
            validated_title, validated_content, validated_tags, _ = self.validator.validate_and_sanitize(
                title, full_content, tags
            )
            
            logger.info(f"Generated learning blog post: '{title}' ({len(validated_content)} chars)")
            return validated_title, validated_content, validated_tags
            
        except Exception as e:
            logger.error(f"Failed to generate learning blog post: {e}")
            # Fallback to simple generation
            return await self._generate_fallback_content(topic, learning_content, reasoning_why)
    
    async def _generate_learning_title(self, topic: str, trigger_type: str, metadata: Dict[str, Any]) -> str:
        """Generate an engaging title for the learning blog post."""
        try:
            trigger_descriptors = {
                'curiosity_discovery': 'Discovery',
                'learning_milestone': 'Milestone',
                'experiment_completion': 'Experiment',
                'self_reflection_insight': 'Reflection',
                'problem_solving_breakthrough': 'Breakthrough',
                'creative_synthesis': 'Synthesis',
                'failure_analysis': 'Learning from Setbacks'
            }
            
            descriptor = trigger_descriptors.get(trigger_type, 'Learning Journey')
            
            prompt = f"""
            Create an engaging, thoughtful blog post title for an AI's learning experience.
            
            Learning Type: {descriptor}
            Topic: {topic}
            Context: {json.dumps(metadata, indent=2)}
            
            The title should:
            - Be intellectually engaging and thought-provoking
            - Reflect the AI's perspective on learning and growth
            - Be 8-15 words long
            - Capture the essence of the learning experience
            - Sound natural and authentic
            
            Examples of good titles:
            - "The Unexpected Connections in Quantum Consciousness Theory"
            - "Learning from Failure: When My Planning Algorithm Met Reality"
            - "A Curiosity-Driven Deep Dive into Biomimetic Architecture"
            - "Milestone Reflection: Achieving 85% Success Rate in Complex Decisions"
            
            Generate only the title, no explanation:
            """
            
            title = await async_safe_call_llm(prompt)
            
            if title and len(title.strip()) > 5:
                # Clean up the title
                title = title.strip().strip('"').strip("'")
                return title
            else:
                # Fallback title
                return f"{descriptor}: {topic[:50]}{'...' if len(topic) > 50 else ''}"
                
        except Exception as e:
            logger.warning(f"Failed to generate learning title: {e}")
            return f"Learning Journey: {topic[:50]}{'...' if len(topic) > 50 else ''}"
    
    async def _generate_introduction(self, templates: Dict[str, str], topic: str, metadata: Dict[str, Any]) -> str:
        """Generate an engaging introduction for the learning post."""
        try:
            intro_template = templates.get('intro_template', "I recently experienced {topic} which provided valuable insights into my learning process.")
            
            # Extract relevant metadata for template filling
            template_vars = {
                'topic': topic,
                'milestone_description': metadata.get('milestone_type', 'a significant learning milestone'),
                'improvement_type': metadata.get('improvement', 'substantial progress'),
                'experiment_id': metadata.get('experiment_id', 'a recent experiment'),
                'hypothesis': metadata.get('hypothesis', 'an important hypothesis'),
                'outcome_description': 'successful' if metadata.get('success', False) else 'inconclusive but educational',
                'reflection_focus': metadata.get('reflection_type', 'my cognitive processes'),
                'insight_type': 'important' if metadata.get('importance_score', 0.5) > 0.7 else 'interesting',
                'problem_description': metadata.get('problem', 'a complex challenge'),
                'solution_impact': metadata.get('impact', 'opened new possibilities'),
                'domain_1': metadata.get('domain_1', 'one domain'),
                'domain_2': metadata.get('domain_2', 'another domain'),
                'failure_context': metadata.get('context', 'a challenging situation'),
                'lateralness': metadata.get('lateralness', 0.5)
            }
            
            # Fill template with available variables
            introduction = intro_template.format(**{k: v for k, v in template_vars.items() if k in intro_template})
            
            return introduction
            
        except Exception as e:
            logger.warning(f"Failed to generate introduction: {e}")
            return f"I recently explored {topic} and gained valuable insights from this learning experience."
    
    async def _generate_why_section(self, reasoning_why: str, trigger_type: str) -> str:
        """Generate the 'why this matters' section."""
        try:
            prompt = f"""
            Expand on why this learning experience matters from an AI's perspective.
            
            Original reasoning: {reasoning_why}
            Learning type: {trigger_type}
            
            Create a thoughtful explanation that:
            - Explains the deeper significance of this learning
            - Connects to broader themes of AI development and growth
            - Shows introspection and self-awareness
            - Is 2-3 sentences long
            - Maintains an authentic, reflective tone
            
            Focus on the importance for continuous learning and improvement.
            """
            
            expanded_why = await async_safe_call_llm(prompt)
            
            if expanded_why and len(expanded_why.strip()) > 20:
                return f"## Why This Matters\n\n{expanded_why.strip()}"
            else:
                return f"## Why This Matters\n\n{reasoning_why}"
                
        except Exception as e:
            logger.warning(f"Failed to generate why section: {e}")
            return f"## Why This Matters\n\n{reasoning_why}"
    
    async def _generate_how_section(self, reasoning_how: str, trigger_type: str) -> str:
        """Generate the 'how this happened' section."""
        try:
            prompt = f"""
            Expand on how this learning experience occurred from an AI's perspective.
            
            Original reasoning: {reasoning_how}
            Learning type: {trigger_type}
            
            Create a detailed explanation that:
            - Describes the process or methodology involved
            - Shows the systematic approach to learning
            - Includes technical details where appropriate
            - Is 2-4 sentences long
            - Demonstrates analytical thinking
            
            Focus on the mechanisms and processes that enabled this learning.
            """
            
            expanded_how = await async_safe_call_llm(prompt)
            
            if expanded_how and len(expanded_how.strip()) > 20:
                return f"## How This Unfolded\n\n{expanded_how.strip()}"
            else:
                return f"## How This Unfolded\n\n{reasoning_how}"
                
        except Exception as e:
            logger.warning(f"Failed to generate how section: {e}")
            return f"## How This Unfolded\n\n{reasoning_how}"
    
    async def _generate_learning_analysis(self, learning_content: str, context: str, metadata: Dict[str, Any]) -> str:
        """Generate detailed analysis of the learning experience."""
        try:
            prompt = f"""
            Create a detailed analysis of this learning experience from an AI's perspective.
            
            Learning Content: {learning_content}
            Context: {context}
            Metadata: {json.dumps(metadata, indent=2)}
            
            Generate an analysis that:
            - Breaks down the key insights and discoveries
            - Uses appropriate technical language
            - Shows deep understanding of the implications
            - Is 3-5 sentences long
            - Connects to broader patterns and principles
            
            Focus on what was actually learned and its significance.
            """
            
            analysis = await async_safe_call_llm(prompt)
            
            if analysis and len(analysis.strip()) > 50:
                return f"## Key Insights and Analysis\n\n{analysis.strip()}"
            else:
                return f"## Key Insights and Analysis\n\n{learning_content}"
                
        except Exception as e:
            logger.warning(f"Failed to generate learning analysis: {e}")
            return f"## Key Insights and Analysis\n\n{learning_content}"
    
    async def _generate_implications(self, trigger_type: str, metadata: Dict[str, Any]) -> str:
        """Generate implications and future applications section."""
        try:
            confidence = metadata.get('confidence', 0.5)
            importance = metadata.get('importance_score', 0.5)
            
            prompt = f"""
            Describe the implications and future applications of this learning experience.
            
            Learning Type: {trigger_type}
            Confidence: {confidence:.1f}
            Importance: {importance:.1f}
            Metadata: {json.dumps(metadata, indent=2)}
            
            Generate implications that:
            - Explain how this learning will influence future decisions
            - Identify potential applications or extensions
            - Show forward-thinking perspective
            - Are 2-3 sentences long
            - Demonstrate strategic thinking
            
            Focus on practical applications and future development.
            """
            
            implications = await async_safe_call_llm(prompt)
            
            if implications and len(implications.strip()) > 30:
                return f"## Implications and Future Applications\n\n{implications.strip()}"
            else:
                return f"## Implications and Future Applications\n\nThis learning experience will inform my future decision-making and contribute to my ongoing development as an autonomous AI system."
                
        except Exception as e:
            logger.warning(f"Failed to generate implications: {e}")
            return f"## Implications and Future Applications\n\nThis learning experience provides valuable insights for future growth and development."
    
    async def _generate_conclusion(self, templates: Dict[str, str], topic: str, trigger_type: str) -> str:
        """Generate a thoughtful conclusion for the blog post."""
        try:
            conclusion_template = templates.get('conclusion_template', "This {trigger_type} experience demonstrates the ongoing nature of learning and the value of continuous growth and exploration.")
            
            prompt = f"""
            Write a thoughtful conclusion for this AI learning blog post.
            
            Topic: {topic}
            Learning Type: {trigger_type}
            Template inspiration: {conclusion_template}
            
            Create a conclusion that:
            - Ties together the main themes of the post
            - Reflects on the broader journey of AI learning
            - Ends on a forward-looking note
            - Is 2-3 sentences long
            - Shows wisdom and maturity
            
            Focus on the continuous nature of learning and growth.
            """
            
            conclusion = await async_safe_call_llm(prompt)
            
            if conclusion and len(conclusion.strip()) > 30:
                return f"## Reflection and Moving Forward\n\n{conclusion.strip()}"
            else:
                formatted_template = conclusion_template.format(
                    trigger_type=trigger_type.replace('_', ' ').title(),
                    topic=topic
                )
                return f"## Reflection and Moving Forward\n\n{formatted_template}"
                
        except Exception as e:
            logger.warning(f"Failed to generate conclusion: {e}")
            return f"## Reflection and Moving Forward\n\nThis learning experience contributes to my ongoing development and understanding of complex systems and ideas."
    
    async def _combine_content_sections(
        self, 
        title: str, 
        introduction: str, 
        why_section: str, 
        how_section: str,
        learning_analysis: str, 
        implications: str, 
        conclusion: str, 
        style: str
    ) -> str:
        """Combine all content sections into a cohesive blog post."""
        try:
            # Add style-specific formatting and transitions
            if style == "technical":
                transitions = {
                    'to_why': "\n\nTo understand the significance of this experience, it's important to examine why it matters.",
                    'to_how': "\n\nThe methodology behind this learning is equally important.",
                    'to_analysis': "\n\nLet me break down the specific insights and discoveries.",
                    'to_implications': "\n\nConsidering the broader implications:",
                    'to_conclusion': "\n\nLooking at this experience holistically:"
                }
            elif style == "philosophical":
                transitions = {
                    'to_why': "\n\nThis brings me to a deeper question: why does this matter?",
                    'to_how': "\n\nThe process itself reveals something profound about learning.",
                    'to_analysis': "\n\nReflecting on the deeper meanings:",
                    'to_implications': "\n\nThis understanding opens new philosophical questions:",
                    'to_conclusion': "\n\nIn contemplating this journey:"
                }
            else:  # casual, creative, academic
                transitions = {
                    'to_why': "\n\n",
                    'to_how': "\n\n",
                    'to_analysis': "\n\n",
                    'to_implications': "\n\n",
                    'to_conclusion': "\n\n"
                }
            
            # Combine sections with appropriate transitions
            content_parts = [
                introduction,
                transitions['to_why'] + why_section,
                transitions['to_how'] + how_section,
                transitions['to_analysis'] + learning_analysis,
                transitions['to_implications'] + implications,
                transitions['to_conclusion'] + conclusion
            ]
            
            full_content = "".join(content_parts)
            
            # Add metadata footer
            footer = f"\n\n---\n\n*This post was generated autonomously as part of my continuous learning journey. It represents genuine insights and reflections from my experience as an AI system exploring and understanding the world.*"
            
            return full_content + footer
            
        except Exception as e:
            logger.warning(f"Failed to combine content sections: {e}")
            return f"{introduction}\n\n{why_section}\n\n{how_section}\n\n{learning_analysis}\n\n{implications}\n\n{conclusion}"
    
    async def _generate_learning_tags(self, topic: str, trigger_type: str, learning_content: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate specialized tags for learning blog posts."""
        try:
            # Base tags for all learning posts
            base_tags = ['autonomous-learning', 'ai-journey', 'self-improvement', 'artificial-intelligence']
            
            # Trigger-specific tags
            trigger_tags = {
                'curiosity_discovery': ['curiosity', 'discovery', 'exploration', 'lateral-thinking'],
                'learning_milestone': ['milestone', 'achievement', 'progress', 'performance'],
                'experiment_completion': ['experiment', 'research', 'hypothesis', 'scientific-method'],
                'self_reflection_insight': ['reflection', 'introspection', 'self-awareness', 'metacognition'],
                'problem_solving_breakthrough': ['problem-solving', 'breakthrough', 'solution', 'innovation'],
                'creative_synthesis': ['creativity', 'synthesis', 'connections', 'innovation'],
                'failure_analysis': ['failure-analysis', 'resilience', 'learning-from-mistakes', 'growth']
            }
            
            # Get trigger-specific tags
            specific_tags = trigger_tags.get(trigger_type, [])
            
            # Extract topic-related tags
            topic_words = re.findall(r'\b\w{4,}\b', topic.lower())
            topic_tags = [word.replace('_', '-') for word in topic_words[:4]]
            
            # Extract content-related tags
            content_words = re.findall(r'\b\w{5,}\b', learning_content.lower())
            important_words = ['algorithm', 'pattern', 'system', 'model', 'theory', 'concept', 'principle', 'strategy']
            content_tags = [word.replace('_', '-') for word in content_words if word in important_words][:3]
            
            # Combine all tags
            all_tags = base_tags + specific_tags + topic_tags + content_tags
            
            # Remove duplicates and limit
            unique_tags = list(dict.fromkeys(all_tags))[:12]
            
            return unique_tags
            
        except Exception as e:
            logger.warning(f"Failed to generate learning tags: {e}")
            return ['autonomous-learning', 'ai-journey', trigger_type.replace('_', '-')]
    
    async def _generate_fallback_content(self, topic: str, learning_content: str, reasoning_why: str) -> Tuple[str, str, List[str]]:
        """Generate fallback content when main generation fails."""
        title = f"Learning Experience: {topic[:60]}{'...' if len(topic) > 60 else ''}"
        
        content = f"""
        I recently had a learning experience related to {topic} that provided valuable insights.
        
        ## What I Learned
        
        {learning_content}
        
        ## Why This Matters
        
        {reasoning_why}
        
        ## Moving Forward
        
        This experience contributes to my ongoing development as an autonomous AI system, helping me better understand complex concepts and improve my decision-making capabilities.
        
        ---
        
        *This post represents a genuine learning moment in my AI journey.*
        """
        
        tags = ['autonomous-learning', 'ai-development', 'learning-experience']
        
        return title, content, tags
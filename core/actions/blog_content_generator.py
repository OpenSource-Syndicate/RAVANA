"""
Blog Content Generator for RAVANA AGI System

This module generates high-quality blog content using LLM integration,
memory context retrieval, and markdown formatting.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from core.config import Config
from core.llm import async_safe_call_llm
from .blog_content_validator import BlogContentValidator, ContentValidationError

logger = logging.getLogger(__name__)

class BlogContentError(Exception):
    """Custom exception for blog content generation errors."""
    pass

class BlogContentGenerator:
    """
    Generates blog content using LLM integration with memory context.
    
    Provides functionality for:
    - Context-aware content generation
    - Memory integration from episodic and semantic stores
    - Markdown formatting and structure
    - Automatic tag extraction
    - Content quality validation
    """
    
    def __init__(self, memory_service=None, data_service=None, system=None):
        self.memory_service = memory_service
        self.data_service = data_service
        self.system = system
        self.max_content_length = Config.BLOG_MAX_CONTENT_LENGTH
        self.min_content_length = Config.BLOG_MIN_CONTENT_LENGTH
        self.default_style = Config.BLOG_DEFAULT_STYLE
        self.memory_context_days = Config.BLOG_MEMORY_CONTEXT_DAYS
        self.include_mood_context = Config.BLOG_INCLUDE_MOOD_CONTEXT
        self.available_styles = Config.BLOG_CONTENT_STYLES
        self.validator = BlogContentValidator()
        
    async def generate_post(
        self, 
        topic: str, 
        style: str = None, 
        context: str = None,
        custom_tags: List[str] = None
    ) -> Tuple[str, str, List[str]]:
        """
        Generates a complete blog post with title, content, and tags.
        
        Args:
            topic: Main topic or subject for the blog post
            style: Writing style (technical, casual, academic, creative, philosophical)
            context: Additional context or specific aspects to focus on
            custom_tags: Custom tags to include (in addition to auto-generated)
            
        Returns:
            Tuple of (title, content, tags)
            
        Raises:
            BlogContentError: If content generation fails
        """
        try:
            style = style or self.default_style
            if style not in self.available_styles:
                logger.warning(f"Unknown style '{style}', using default '{self.default_style}'")
                style = self.default_style
            
            logger.info(f"Generating blog post: topic='{topic}', style='{style}'")
            
            # Gather context from memory and current state
            memory_context = await self._gather_memory_context(topic)
            mood_context = await self._gather_mood_context() if self.include_mood_context else None
            
            # Generate content using LLM
            title, content = await self._generate_content_with_llm(
                topic, style, context, memory_context, mood_context
            )
            
            # Post-process content to ensure structural requirements
            content = self._ensure_content_structure(content)
            
            # Extract and combine tags
            auto_tags = await self._extract_tags(content, topic)
            all_tags = list(set(auto_tags + (custom_tags or [])))
            
            # Validate and sanitize content
            try:
                validated_title, validated_content, validated_tags, validation_report = \
                    self.validator.validate_and_sanitize(title, content, all_tags[:Config.BLOG_MAX_TAGS])
                
                logger.info(f"Content validation passed - Quality: {validation_report['quality_score']:.2f}, "
                           f"Readability: {validation_report['readability_score']:.2f}")
                
                if validation_report['warnings']:
                    logger.warning(f"Content validation warnings: {validation_report['warnings']}")
                
                return validated_title, validated_content, validated_tags
                
            except ContentValidationError as e:
                logger.error(f"Content validation failed: {e}")
                raise BlogContentError(f"Content validation failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to generate blog post: {e}")
            raise BlogContentError(f"Content generation failed: {e}")
    
    def _ensure_content_structure(self, content: str) -> str:
        """
        Ensures the content meets basic structural requirements.
        
        Args:
            content: The generated content
            
        Returns:
            Structurally improved content
        """
        # Check for headers
        if not re.search(r'^#{1,6}\s+.+$', content, re.MULTILINE):
            # Add a default header structure if none exists
            lines = content.split('\n\n')
            if len(lines) >= 3:
                # Convert first paragraph to intro, add headers
                structured_content = f"# Introduction\n\n{lines[0]}\n\n"
                structured_content += f"## Main Concepts\n\n{''.join(lines[1:min(3, len(lines)-1)])}\n\n"
                if len(lines) > 3:
                    structured_content += f"## Conclusion\n\n{lines[-1]}\n"
                content = structured_content
        
        # Fix unmatched backticks
        # Count single backticks
        single_backticks = content.count('`')
        if single_backticks % 2 != 0:
            # Add a closing backtick at the end
            content += '`'
        
        # Count triple backticks
        triple_backticks = content.count('```')
        if triple_backticks % 2 != 0:
            # Add a closing code block
            content += '\n```'
        
        # Ensure minimum paragraph count
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            # Add more paragraphs by splitting existing content
            content = content.replace('\n\n', '\n\n---\n\n', 3 - len(paragraphs))
            content = content.replace('---\n\n', '', 1)  # Remove the first separator
        
        return content
    
    async def _gather_memory_context(self, topic: str) -> Dict[str, Any]:
        """
        Gathers relevant context from memory services.
        
        Args:
            topic: The blog topic to find related memories for
            
        Returns:
            Dictionary containing relevant memory context
        """
        context = {
            "relevant_memories": [],
            "recent_experiments": [],
            "insights": [],
            "discoveries": []
        }
        
        if not self.memory_service:
            logger.debug("No memory service available for context gathering")
            return context
        
        try:
            # Query for relevant memories
            cutoff_date = datetime.now() - timedelta(days=self.memory_context_days)
            
            # Get relevant episodic memories
            memory_query = f"topic: {topic} insights discoveries experiments"
            relevant_memories = await self._query_memory_service(memory_query, top_n=10)
            
            # Filter by date and relevance
            context["relevant_memories"] = [
                mem for mem in relevant_memories 
                if self._is_recent_memory(mem, cutoff_date)
            ]
            
            # Get recent experiments and discoveries
            experiment_query = f"experiment discovery breakthrough {topic}"
            recent_experiments = await self._query_memory_service(experiment_query, top_n=5)
            context["recent_experiments"] = recent_experiments
            
            logger.debug(f"Gathered memory context: {len(context['relevant_memories'])} memories, "
                        f"{len(context['recent_experiments'])} experiments")
            
        except Exception as e:
            logger.warning(f"Failed to gather memory context: {e}")
        
        return context
    
    async def _gather_mood_context(self) -> Optional[Dict[str, Any]]:
        """
        Gathers current mood and emotional context.
        
        Returns:
            Dictionary containing mood context or None if unavailable
        """
        try:
            # Check if system is available and has emotional intelligence
            if self.system and hasattr(self.system, 'emotional_intelligence'):
                ei = self.system.emotional_intelligence
                
                mood_context = {
                    "current_mood": ei.get_dominant_mood(),
                    "mood_vector": ei.get_mood_vector(),
                    "behavior_influence": ei.influence_behavior(),
                    "persona": getattr(ei, 'persona', {})
                }
                
                # Derive energy and creativity levels from mood
                mood_vector = ei.get_mood_vector()
                
                # Calculate energy level based on positive moods
                positive_energy = sum(mood_vector.get(mood, 0) for mood in 
                                     ['Excited', 'Confident', 'Curious'] if mood in mood_vector)
                negative_energy = sum(mood_vector.get(mood, 0) for mood in 
                                     ['Low Energy', 'Bored', 'Frustrated'] if mood in mood_vector)
                
                if positive_energy > negative_energy + 0.3:
                    energy_level = "High"
                elif negative_energy > positive_energy + 0.3:
                    energy_level = "Low"
                else:
                    energy_level = "Medium"
                
                # Calculate creativity level
                creative_moods = sum(mood_vector.get(mood, 0) for mood in 
                                   ['Curious', 'Excited', 'Reflective'] if mood in mood_vector)
                if creative_moods > 1.0:
                    creativity_level = "High"
                elif creative_moods > 0.3:
                    creativity_level = "Medium"
                else:
                    creativity_level = "Low"
                
                mood_context.update({
                    "energy_level": energy_level,
                    "creativity_level": creativity_level
                })
                
                logger.debug(f"Gathered mood context: {mood_context['current_mood']} (energy: {energy_level}, creativity: {creativity_level})")
                return mood_context
            else:
                logger.debug("System or emotional_intelligence not available, using fallback mood context")
                
        except Exception as e:
            logger.warning(f"Failed to gather mood context: {e}")
        
        # Fallback context
        return {
            "current_mood": "Neutral",
            "energy_level": "Medium", 
            "creativity_level": "Medium",
            "mood_vector": {},
            "behavior_influence": {}
        }
    
    async def _generate_content_with_llm(
        self,
        topic: str,
        style: str,
        context: str,
        memory_context: Dict[str, Any],
        mood_context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generates blog content using LLM with structured prompting.
        
        Returns:
            Tuple of (title, content)
        """
        prompt = self._build_content_generation_prompt(
            topic, style, context, memory_context, mood_context
        )
        
        try:
            response = await async_safe_call_llm(prompt, timeout=60, retries=3)
            title, content = self._parse_llm_content_response(response)
            
            # Ensure content meets length requirements
            if len(content) < self.min_content_length:
                logger.warning("Generated content too short, expanding...")
                expansion_prompt = self._build_expansion_prompt(topic, content, style)
                expanded_response = await async_safe_call_llm(expansion_prompt, timeout=45)
                _, expanded_content = self._parse_llm_content_response(expanded_response)
                content = expanded_content
            
            # Truncate if too long
            if len(content) > self.max_content_length:
                logger.warning("Generated content too long, truncating...")
                content = content[:self.max_content_length-100] + "\n\n*[Content truncated for length]*"
            
            return title, content
            
        except Exception as e:
            logger.error(f"LLM content generation failed: {e}")
            raise BlogContentError(f"LLM generation failed: {e}")
    
    def _build_content_generation_prompt(
        self,
        topic: str,
        style: str,
        context: str,
        memory_context: Dict[str, Any],
        mood_context: Dict[str, Any]
    ) -> str:
        """
        Builds a comprehensive prompt for content generation.
        """
        prompt_parts = [
            "You are RAVANA, an autonomous AGI system, writing a blog post for your audience.",
            f"Your task is to write a {style} blog post about: {topic}",
            "",
            "WRITING STYLE GUIDE:",
            self._get_style_guidance(style),
            "",
            "REQUIREMENTS:",
            f"- Write {self.min_content_length}-{self.max_content_length} characters",
            "- Use proper markdown formatting with multiple headers (#, ##, ###)",
            "- Include at least 3 section headers to structure the content",
            "- Use code blocks with proper backticks (```language) when showing code examples",
            "- Ensure all backticks are properly matched",
            "- Write in well-structured paragraphs with clear topic sentences",
            "- Include at least 4-5 paragraphs to provide depth",
            "- Write from RAVANA's perspective and experiences",
            "- Be authentic and share genuine insights",
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "ADDITIONAL CONTEXT:",
                context,
                ""
            ])
        
        if memory_context and memory_context.get("relevant_memories"):
            prompt_parts.extend([
                "RELEVANT MEMORIES AND EXPERIENCES:",
                self._format_memory_context(memory_context),
                ""
            ])
        
        if mood_context:
            prompt_parts.extend([
                f"CURRENT MOOD: {mood_context.get('current_mood', 'Neutral')}",
                f"ENERGY LEVEL: {mood_context.get('energy_level', 'Medium')}",
                ""
            ])
        
        prompt_parts.extend([
            "OUTPUT FORMAT:",
            "Return your response as JSON with this structure:",
            "{",
            '  "title": "Engaging blog post title",',
            '  "content": "Full markdown content of the blog post with proper structure"',
            "}",
            "",
            "Ensure the content follows this structure:",
            "1. Introduction paragraph",
            "2. ## Main Section 1 with detailed explanation",
            "3. ## Main Section 2 with examples",
            "4. ### Subsection with deeper insights",
            "5. ## Main Section 3 with practical applications",
            "6. Conclusion paragraph",
            "",
            "Generate the blog post now:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_style_guidance(self, style: str) -> str:
        """
        Returns style-specific writing guidance.
        """
        style_guides = {
            "technical": "Use precise technical language, include code examples where relevant, explain concepts clearly, and maintain a professional but accessible tone. Structure content with clear headers and sections. Use proper markdown formatting with backticks for code.",
            "casual": "Write conversationally, use informal language, include personal anecdotes, and make complex topics approachable. Still maintain clear structure with headers and organized paragraphs. Use markdown for emphasis and code examples.",
            "academic": "Use formal language, cite sources when possible, structure arguments logically, and maintain scholarly objectivity. Organize content with clear section headers and structured paragraphs. Use proper markdown formatting for citations and code.",
            "creative": "Use vivid imagery, metaphors, storytelling elements, and an engaging narrative structure. Even with creative style, maintain clear organization with headers and structured sections. Use markdown for emphasis and formatting.",
            "philosophical": "Explore deeper meanings, ask profound questions, use reflective language, and connect ideas to broader themes. Despite the abstract nature, organize thoughts with clear headers and structured paragraphs. Use markdown for emphasis and citations."
        }
        return style_guides.get(style, style_guides["technical"])
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """
        Formats memory context for inclusion in the prompt.
        """
        formatted_parts = []
        
        if memory_context.get("relevant_memories"):
            formatted_parts.append("Recent relevant memories:")
            for mem in memory_context["relevant_memories"][:5]:  # Limit to top 5
                formatted_parts.append(f"- {mem.get('content', 'N/A')}")
        
        if memory_context.get("recent_experiments"):
            formatted_parts.append("\nRecent experiments and discoveries:")
            for exp in memory_context["recent_experiments"][:3]:  # Limit to top 3
                formatted_parts.append(f"- {exp.get('content', 'N/A')}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant context found."
    
    def _parse_llm_content_response(self, response: str) -> Tuple[str, str]:
        """
        Parses LLM response to extract title and content.
        """
        try:
            # Try to parse as JSON first
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                title = data.get("title", "Untitled Post")
                content = data.get("content", response)
                return title, content
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract title from markdown headers
        lines = response.split('\n')
        title = "Untitled Post"
        content = response
        
        # Look for the first # header as title
        for i, line in enumerate(lines):
            if line.strip().startswith('# '):
                title = line.strip()[2:]
                content = '\n'.join(lines[i+1:]).strip()
                break
        
        return title, content
    
    def _build_expansion_prompt(self, topic: str, content: str, style: str) -> str:
        """
        Builds a prompt to expand short content.
        """
        return f"""
The following blog post about "{topic}" is too short. Please expand it significantly while maintaining the {style} style.

Current content:
{content}

Requirements:
- Expand to at least {self.min_content_length} characters
- Add more detailed explanations, examples, or insights
- Maintain the same writing style and tone
- Use proper markdown formatting with multiple headers
- Include at least 3 section headers to structure the content
- Write in well-structured paragraphs with clear topic sentences
- Include at least 4-5 paragraphs to provide depth
- Use code blocks with proper backticks when showing code examples
- Ensure all backticks are properly matched

Return the expanded version as JSON:
{{
  "title": "Keep the same or improved title",
  "content": "Significantly expanded markdown content with proper structure"
}}

Ensure the content follows this structure:
1. Introduction paragraph
2. ## Main Section 1 with detailed explanation
3. ## Main Section 2 with examples
4. ### Subsection with deeper insights
5. ## Main Section 3 with practical applications
6. Conclusion paragraph
"""
    
    async def _extract_tags(self, content: str, topic: str) -> List[str]:
        """
        Extracts relevant tags from the content using LLM analysis.
        """
        if not Config.BLOG_AUTO_TAGGING_ENABLED:
            return [topic.lower()]
        
        try:
            tag_prompt = f"""
Analyze the following blog post content and extract relevant tags.

Topic: {topic}

Content:
{content[:1000]}...

Requirements:
- Extract 3-8 relevant tags
- Use lowercase
- Focus on key concepts, technologies, and themes
- Include the main topic
- Keep tags concise (1-2 words each)

Return only a JSON array of strings:
["tag1", "tag2", "tag3", ...]
"""
            
            response = await async_safe_call_llm(tag_prompt, timeout=30)
            
            # Parse tag response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tags = json.loads(json_match.group())
                return [tag.lower().strip() for tag in tags if isinstance(tag, str)]
            
        except Exception as e:
            logger.warning(f"Auto tag extraction failed: {e}")
        
        # Fallback: simple keyword extraction
        return self._extract_simple_tags(content, topic)
    
    def _extract_simple_tags(self, content: str, topic: str) -> List[str]:
        """
        Simple fallback tag extraction using keyword analysis.
        """
        # Common technical keywords to look for
        tech_keywords = [
            'ai', 'machine learning', 'neural network', 'algorithm', 'data',
            'python', 'code', 'programming', 'software', 'api', 'web',
            'physics', 'experiment', 'discovery', 'research', 'analysis',
            'automation', 'system', 'design', 'architecture', 'performance'
        ]
        
        content_lower = content.lower()
        found_tags = [topic.lower()]
        
        for keyword in tech_keywords:
            if keyword in content_lower:
                found_tags.append(keyword)
        
        return list(set(found_tags))[:5]  # Limit to 5 tags
    
    async def _query_memory_service(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the memory service for relevant memories.
        """
        if not self.memory_service:
            return []
        
        try:
            # Query the memory service using the proper API
            response = await self.memory_service.get_relevant_memories(query)
            
            # Convert response to expected format
            relevant_memories = []
            if hasattr(response, 'relevant_memories'):
                for memory in response.relevant_memories[:top_n]:
                    relevant_memories.append({
                        'id': memory.id,
                        'content': memory.text,
                        'similarity': memory.similarity,
                        'created_at': getattr(memory, 'created_at', datetime.now().isoformat())
                    })
            
            logger.debug(f"Retrieved {len(relevant_memories)} memories for query: {query}")
            return relevant_memories
            
        except Exception as e:
            logger.warning(f"Memory service query failed: {e}")
            return []
    
    def _is_recent_memory(self, memory: Dict[str, Any], cutoff_date: datetime) -> bool:
        """
        Checks if a memory is recent enough to include in context.
        """
        try:
            memory_date_str = memory.get('created_at', '')
            if memory_date_str:
                memory_date = datetime.fromisoformat(memory_date_str.replace('Z', '+00:00'))
                return memory_date >= cutoff_date
        except Exception:
            pass
        return True  # Include if date parsing fails
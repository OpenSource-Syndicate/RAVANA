"""
Blog Content Validator and Sanitizer for RAVANA AGI System

This module provides comprehensive content validation and sanitization
to ensure blog posts meet quality standards and security requirements.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from core.config import Config

logger = logging.getLogger(__name__)

class ContentValidationError(Exception):
    """Exception raised when content validation fails."""
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.error_code = error_code

class BlogContentValidator:
    """
    Validates and sanitizes blog content for quality and security.
    
    Provides functionality for:
    - Content length validation
    - Markdown structure validation
    - Security sanitization
    - Quality scoring
    - Readability analysis
    """
    
    def __init__(self):
        self.min_length = Config.BLOG_MIN_CONTENT_LENGTH
        self.max_length = Config.BLOG_MAX_CONTENT_LENGTH
        self.max_tags = Config.BLOG_MAX_TAGS
        
        # Security patterns to detect potentially harmful content
        self.security_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'<iframe[^>]*>.*?</iframe>',  # Iframes
            r'<object[^>]*>.*?</object>',  # Objects
            r'<embed[^>]*>.*?</embed>',  # Embeds
        ]
        
        # Quality indicators
        self.quality_patterns = {
            'headers': r'^#{1,6}\s+.+$',  # Markdown headers
            'code_blocks': r'```[\s\S]*?```',  # Code blocks
            'inline_code': r'`[^`]+`',  # Inline code
            'links': r'\[([^\]]+)\]\(([^)]+)\)',  # Links
            'bold': r'\*\*[^*]+\*\*',  # Bold text
            'italic': r'\*[^*]+\*',  # Italic text
            'lists': r'^[\s]*[-*+]\s+.+$',  # Lists
        }
        
    def validate_and_sanitize(
        self, 
        title: str, 
        content: str, 
        tags: List[str]
    ) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """
        Validates and sanitizes blog content.
        
        Args:
            title: Blog post title
            content: Blog post content in markdown
            tags: List of tags
            
        Returns:
            Tuple of (sanitized_title, sanitized_content, sanitized_tags, validation_report)
            
        Raises:
            ContentValidationError: If content fails validation
        """
        validation_report = {
            "validation_passed": True,
            "warnings": [],
            "fixes_applied": [],
            "quality_score": 0.0,
            "readability_score": 0.0
        }
        
        try:
            # Validate and sanitize title
            sanitized_title = self._validate_and_sanitize_title(title, validation_report)
            
            # Validate and sanitize content
            sanitized_content = self._validate_and_sanitize_content(content, validation_report)
            
            # Validate and sanitize tags
            sanitized_tags = self._validate_and_sanitize_tags(tags, validation_report)
            
            # Calculate quality scores
            validation_report["quality_score"] = self._calculate_quality_score(sanitized_content)
            validation_report["readability_score"] = self._calculate_readability_score(sanitized_content)
            
            # Final validation
            self._perform_final_validation(
                sanitized_title, sanitized_content, sanitized_tags, validation_report
            )
            
            logger.info(f"Content validation passed - Quality: {validation_report['quality_score']:.2f}, "
                       f"Readability: {validation_report['readability_score']:.2f}")
            
            return sanitized_title, sanitized_content, sanitized_tags, validation_report
            
        except Exception as e:
            validation_report["validation_passed"] = False
            logger.error(f"Content validation failed: {e}")
            raise ContentValidationError(str(e))
    
    def _validate_and_sanitize_title(self, title: str, report: Dict[str, Any]) -> str:
        """Validates and sanitizes the blog title."""
        if not title or not title.strip():
            raise ContentValidationError("Title cannot be empty", "EMPTY_TITLE")
        
        # Sanitize title
        sanitized_title = title.strip()
        
        # Remove potentially harmful content
        sanitized_title = self._remove_security_threats(sanitized_title, "title")
        
        # Length validation
        if len(sanitized_title) < 5:
            raise ContentValidationError("Title is too short (minimum 5 characters)", "TITLE_TOO_SHORT")
        
        if len(sanitized_title) > 200:
            sanitized_title = sanitized_title[:197] + "..."
            report["fixes_applied"].append("Title truncated to 200 characters")
        
        # Check for basic quality
        if not re.search(r'[A-Za-z]', sanitized_title):
            report["warnings"].append("Title contains no alphabetic characters")
        
        return sanitized_title
    
    def _validate_and_sanitize_content(self, content: str, report: Dict[str, Any]) -> str:
        """Validates and sanitizes the blog content."""
        if not content or not content.strip():
            raise ContentValidationError("Content cannot be empty", "EMPTY_CONTENT")
        
        sanitized_content = content.strip()
        
        # Remove security threats
        sanitized_content = self._remove_security_threats(sanitized_content, "content")
        
        # Length validation (only enforce minimum, not maximum)
        if len(sanitized_content) < self.min_length:
            raise ContentValidationError(
                f"Content is too short (minimum {self.min_length} characters, got {len(sanitized_content)})",
                "CONTENT_TOO_SHORT"
            )
        
        # Validate markdown structure
        self._validate_markdown_structure(sanitized_content, report)
        
        # Check for minimum quality indicators
        self._check_content_quality(sanitized_content, report)
        
        return sanitized_content
    
    def _validate_and_sanitize_tags(self, tags: List[str], report: Dict[str, Any]) -> List[str]:
        """Validates and sanitizes the tags."""
        if not tags:
            report["warnings"].append("No tags provided")
            return []
        
        sanitized_tags = []
        
        for tag in tags[:self.max_tags]:  # Limit number of tags
            # Sanitize individual tag
            sanitized_tag = tag.strip().lower()
            sanitized_tag = re.sub(r'[^a-z0-9\-_]', '', sanitized_tag)
            
            if sanitized_tag and len(sanitized_tag) >= 2:
                sanitized_tags.append(sanitized_tag)
            else:
                report["warnings"].append(f"Tag '{tag}' removed (invalid or too short)")
        
        if len(tags) > self.max_tags:
            report["fixes_applied"].append(f"Reduced tags from {len(tags)} to {self.max_tags}")
        
        # Ensure at least one tag
        if not sanitized_tags:
            sanitized_tags = ["general"]
            report["fixes_applied"].append("Added default tag 'general'")
        
        return sanitized_tags
    
    def _remove_security_threats(self, text: str, context: str) -> str:
        """Removes potentially harmful content."""
        original_text = text
        
        for pattern in self.security_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove suspicious URLs
        text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]*[^\s.,;:!?<>"{}|\\^`\[\]]', 
                     '[URL REMOVED]', text)
        
        if text != original_text:
            logger.warning(f"Security threats removed from {context}")
        
        return text
    
    def _smart_truncate(self, content: str, max_length: int) -> str:
        """Intelligently truncates content at paragraph or sentence boundaries."""
        if len(content) <= max_length:
            return content
        
        # Try to truncate at paragraph boundary
        paragraphs = content.split('\n\n')
        truncated = ""
        
        for paragraph in paragraphs:
            if len(truncated + paragraph) + 2 <= max_length - 100:  # Leave room for truncation notice
                truncated += paragraph + '\n\n'
            else:
                break
        
        if truncated:
            return truncated.strip() + '\n\n*[Content truncated for length]*'
        
        # Fallback: truncate at sentence boundary
        sentences = re.split(r'[.!?]+', content[:max_length-100])
        if len(sentences) > 1:
            sentences = sentences[:-1]  # Remove incomplete last sentence
            return '. '.join(sentences) + '.\n\n*[Content truncated for length]*'
        
        # Last resort: hard truncate
        return content[:max_length-100] + '...\n\n*[Content truncated for length]*'
    
    def _validate_markdown_structure(self, content: str, report: Dict[str, Any]) -> None:
        """Validates markdown structure and formatting."""
        issues = []
        
        # Check for headers
        if not re.search(self.quality_patterns['headers'], content, re.MULTILINE):
            issues.append("No headers found - content may lack structure")
        
        # Check for unmatched markdown syntax
        unmatched_code = re.findall(r'(?:^|[^`])`(?:[^`]|$)', content)
        if unmatched_code:
            issues.append("Unmatched backticks found - may cause formatting issues")
        
        # Check for very long lines (readability)
        lines = content.split('\n')
        long_lines = [i for i, line in enumerate(lines) if len(line) > 120 and not line.startswith('#')]
        if len(long_lines) > len(lines) * 0.3:  # More than 30% of lines are too long
            issues.append("Many lines exceed 120 characters - may affect readability")
        
        if issues:
            report["warnings"].extend(issues)
    
    def _check_content_quality(self, content: str, report: Dict[str, Any]) -> None:
        """Checks content for quality indicators."""
        issues = []
        
        # Check paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            issues.append("Content has few paragraphs - may lack depth")
        
        # Check for very short paragraphs
        short_paragraphs = [p for p in paragraphs if len(p) < 50]
        if len(short_paragraphs) > len(paragraphs) * 0.5:
            issues.append("Many paragraphs are very short - may lack substance")
        
        # Check word count
        word_count = len(re.findall(r'\b\w+\b', content))
        if word_count < 100:
            issues.append("Low word count - content may be too brief")
        
        if issues:
            report["warnings"].extend(issues)
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculates a quality score based on content features."""
        score = 0.0
        max_score = 10.0
        
        # Check for various markdown features
        for feature, pattern in self.quality_patterns.items():
            matches = len(re.findall(pattern, content, re.MULTILINE))
            if matches > 0:
                score += min(1.0, matches * 0.3)  # Cap contribution per feature
        
        # Check content length (optimal range)
        length_score = min(1.0, len(content) / 1000)  # Up to 1 point for 1000+ chars
        score += length_score
        
        # Check paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_score = min(1.0, len(paragraphs) / 5)  # Up to 1 point for 5+ paragraphs
        score += paragraph_score
        
        # Normalize to 0-10 scale
        return min(10.0, score)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculates a readability score (simplified)."""
        # Remove markdown formatting for readability analysis
        text = re.sub(r'[#*`\[\]()_~]', '', content)
        text = re.sub(r'\n+', ' ', text)
        
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Simple readability metrics
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = self._estimate_syllables(words)
        
        # Simplified Flesch Reading Ease approximation
        # Higher score = more readable
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-10 scale
        return max(0.0, min(10.0, score / 10))
    
    def _estimate_syllables(self, words: List[str]) -> float:
        """Estimates average syllables per word."""
        total_syllables = 0
        
        for word in words:
            # Simple syllable counting heuristic
            vowels = 'aeiouy'
            syllables = 0
            prev_was_vowel = False
            
            for char in word.lower():
                if char in vowels:
                    if not prev_was_vowel:
                        syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Adjust for silent e
            if word.lower().endswith('e') and syllables > 1:
                syllables -= 1
            
            # Minimum of 1 syllable per word
            total_syllables += max(1, syllables)
        
        return total_syllables / len(words) if words else 1.0
    
    def _perform_final_validation(
        self, 
        title: str, 
        content: str, 
        tags: List[str], 
        report: Dict[str, Any]
    ) -> None:
        """Performs final validation checks."""
        # Check minimum quality thresholds
        if report["quality_score"] < 2.0:
            report["warnings"].append("Content quality score is low")
        
        if report["readability_score"] < 3.0:
            report["warnings"].append("Content readability score is low")
        
        # Check for duplicate content patterns (simplified)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(set(sentences)) < len(sentences) * 0.8:  # More than 20% duplicate sentences
            report["warnings"].append("Content contains many similar sentences")
    
    def quick_validate(self, title: str, content: str, tags: List[str]) -> bool:
        """
        Quick validation check without full sanitization.
        
        Returns:
            True if content passes basic validation, False otherwise
        """
        try:
            # Basic checks
            if not title or len(title.strip()) < 5:
                return False
            
            if not content or len(content.strip()) < self.min_length:
                return False
            
            if len(content) > self.max_length * 1.2:  # Allow some overage
                return False
            
            # Check for security threats
            for pattern in self.security_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    return False
            
            return True
            
        except Exception:
            return False
import aiohttp
import asyncio
import requests
import random
import re
import json
import logging
import hashlib
import ssl
from typing import List, Tuple, Dict, Optional
from cachetools import TTLCache
from sentence_transformers import util
from core.llm import call_llm  # Import the LLM utility
from core.embeddings_manager import embeddings_manager, ModelPurpose
import wikipedia

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global caches
_FACT_CACHE = TTLCache(maxsize=500, ttl=7200)  # 2 hours cache
_TOPIC_CACHE = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache for topics


def _get_embedding_model():
    """Get the shared embedding model from the embeddings manager."""
    # Instead of loading our own model, we'll use the global embeddings manager
    return embeddings_manager


async def fetch_html_async(url: str, headers: Optional[Dict] = None, timeout: int = 10) -> str:
    """Async HTML fetcher with timeout and error handling.

    This function handles SSL certificate verification issues that commonly occur
    on Windows systems when accessing certain APIs like arXiv. It includes:
    - SSL context with relaxed certificate verification
    - Fallback to HTTP for HTTPS URLs when SSL fails
    - Proper error handling and logging
    """
    try:
        # Create SSL context that bypasses certificate verification for known issues
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=connector
        ) as session:
            async with session.get(url, headers=headers or {}) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientSSLError as ssl_error:
        logger.warning(
            f"SSL error for {url}: {ssl_error}. Trying with no SSL verification...")
        try:
            # Fallback: Try with completely disabled SSL
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=connector
            ) as session:
                # Convert HTTPS to HTTP for the fallback
                fallback_url = url.replace('https://', 'http://')
                async with session.get(fallback_url, headers=headers or {}) as response:
                    response.raise_for_status()
                    return await response.text()
        except Exception as fallback_error:
            logger.warning(f"Fallback also failed for {url}: {fallback_error}")
            return ""
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def _filter_similar_topics(candidates: List[str], recent: List[str], threshold: float = 0.7) -> List[str]:
    """Filter out topics too similar to recent ones using embeddings."""
    embeddings_mgr = _get_embedding_model()
    if not embeddings_mgr or not recent or not candidates:
        return candidates

    try:
        # Use the shared embeddings manager to generate embeddings
        candidate_embeddings = embeddings_mgr.get_embedding(candidates, purpose=ModelPurpose.SEMANTIC_SEARCH)
        recent_embeddings = embeddings_mgr.get_embedding(recent, purpose=ModelPurpose.SEMANTIC_SEARCH)

        filtered = []
        # Calculate similarities for each candidate against all recent topics
        for i, candidate_text in enumerate(candidates):
            candidate_emb = candidate_embeddings[i] if len(candidate_embeddings.shape) > 1 else candidate_embeddings
            max_similarity = 0.0
            
            for recent_emb in recent_embeddings:
                # Calculate cosine similarity manually since we're using numpy arrays
                dot_product = sum(a * b for a, b in zip(candidate_emb, recent_emb))
                norm_a = sum(a * a for a in candidate_emb) ** 0.5
                norm_b = sum(a * a for a in recent_emb) ** 0.5
                
                if norm_a == 0 or norm_b == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_a * norm_b)
                
                if similarity > max_similarity:
                    max_similarity = similarity
            
            if max_similarity < threshold:
                filtered.append(candidates[i])

        logger.info(
            f"Filtered {len(candidates)} candidates to {len(filtered)} unique topics")
        return filtered

    except Exception as e:
        logger.warning(f"Embedding filtering failed: {e}")
        return candidates


class CuriosityTrigger:
    WIKI_DYK_URL = "https://en.wikipedia.org/wiki/Wikipedia:Recent_additions"
    REDDIT_TIL_URL = "https://www.reddit.com/r/todayilearned/top.json?limit=50&t=week"
    HACKERNEWS_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ARXIV_URL = "http://export.arxiv.org/api/query?search_query=all&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending"
    USER_AGENT = {'User-agent': 'CuriosityTriggerBot/0.3'}
    WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

    def __init__(self, blog_scheduler=None):
        self.sources = {
            'wikipedia': self.fetch_wikipedia_dyk_async,
            'reddit': self.fetch_reddit_til_async,
            'hackernews': self.fetch_hackernews_async,
            'arxiv': self.fetch_arxiv_async
        }
        self.blog_scheduler = blog_scheduler
        self.discovery_count = 0  # Track discoveries for importance scoring

    async def fetch_wikipedia_dyk_async(self) -> List[str]:
        """Async fetch of Wikipedia 'Did you know?' facts."""
        cache_key = "wiki_dyk"
        if cache_key in _FACT_CACHE:
            return _FACT_CACHE[cache_key]

        try:
            html = await fetch_html_async(self.WIKI_DYK_URL)
            if not html:
                return []

            # Extract facts from the HTML using regex
            facts = re.findall(r'<li>(.*?)</li>', html, re.DOTALL)
            # Clean up HTML tags and filter
            clean_facts = []
            for fact in facts:
                clean = re.sub('<.*?>', '', fact).strip()
                if len(clean) > 30 and len(clean) < 500:  # Reasonable length
                    clean_facts.append(clean)

            _FACT_CACHE[cache_key] = clean_facts
            logger.info(f"Fetched {len(clean_facts)} Wikipedia DYK facts")
            return clean_facts

        except Exception as e:
            logger.warning(f"Failed to fetch Wikipedia DYK: {e}")
            return []

    async def fetch_reddit_til_async(self) -> List[str]:
        """Async fetch of Reddit TIL facts."""
        cache_key = "reddit_til"
        if cache_key in _FACT_CACHE:
            return _FACT_CACHE[cache_key]

        try:
            html = await fetch_html_async(self.REDDIT_TIL_URL, headers=self.USER_AGENT)
            if not html:
                return []

            data = json.loads(html)
            facts = []
            for post in data.get('data', {}).get('children', []):
                title = post.get('data', {}).get('title', '')
                if title:
                    # Remove 'TIL that' or 'TIL' prefix
                    clean_title = re.sub(
                        r'^TIL( that)?[\s:,-]*', '', title, flags=re.IGNORECASE)
                    if len(clean_title) > 20:
                        facts.append(clean_title)

            _FACT_CACHE[cache_key] = facts
            logger.info(f"Fetched {len(facts)} Reddit TIL facts")
            return facts

        except Exception as e:
            logger.warning(f"Failed to fetch Reddit TIL: {e}")
            return []

    async def fetch_hackernews_async(self) -> List[str]:
        """Fetch trending topics from Hacker News."""
        cache_key = "hackernews"
        if cache_key in _FACT_CACHE:
            return _FACT_CACHE[cache_key]

        try:
            # Get top story IDs
            story_ids_html = await fetch_html_async(self.HACKERNEWS_URL)
            if not story_ids_html:
                return []

            story_ids = json.loads(story_ids_html)[:10]  # Top 10 stories

            stories = []
            for story_id in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_html = await fetch_html_async(story_url)
                if story_html:
                    story_data = json.loads(story_html)
                    title = story_data.get('title', '')
                    if title and len(title) > 10:
                        stories.append(title)

            _FACT_CACHE[cache_key] = stories
            logger.info(f"Fetched {len(stories)} Hacker News stories")
            return stories

        except Exception as e:
            logger.warning(f"Failed to fetch Hacker News: {e}")
            return []

    async def fetch_arxiv_async(self) -> List[str]:
        """Fetch recent papers from arXiv."""
        cache_key = "arxiv"
        if cache_key in _FACT_CACHE:
            return _FACT_CACHE[cache_key]

        try:
            xml_data = await fetch_html_async(self.ARXIV_URL)
            if not xml_data:
                return []

            # Simple regex to extract titles (could use proper XML parsing)
            titles = re.findall(r'<title>(.*?)</title>', xml_data, re.DOTALL)
            papers = []
            for title in titles[1:]:  # Skip the first title (feed title)
                clean_title = re.sub(r'\s+', ' ', title.strip())
                if len(clean_title) > 20 and not clean_title.startswith('ArXiv'):
                    papers.append(clean_title)

            _FACT_CACHE[cache_key] = papers
            logger.info(f"Fetched {len(papers)} arXiv papers")
            return papers

        except Exception as e:
            logger.warning(f"Failed to fetch arXiv: {e}")
            return []

    @staticmethod
    def is_unrelated(fact: str, recent_topics: List[str]) -> bool:
        """Checks if the fact is unrelated to any of the recent topics (simple keyword check)."""
        fact_lower = fact.lower()
        for topic in recent_topics:
            topic_words = re.findall(r'\w+', topic.lower())
            if any(word in fact_lower for word in topic_words if len(word) > 3):
                return False
        return True

    async def get_curiosity_topics_llm(self, recent_topics: List[str], n: int = 5, lateralness: float = 1.0) -> List[str]:
        """Enhanced LLM-based topic generation with caching and filtering."""
        # Create cache key
        topics_hash = hashlib.md5(
            ','.join(sorted(recent_topics)).encode()).hexdigest()[:8]
        cache_key = f"topics_{topics_hash}_{n}_{lateralness}"

        if cache_key in _TOPIC_CACHE:
            return _TOPIC_CACHE[cache_key]

        lateralness = min(max(lateralness, 0.0), 1.0)

        # Enhanced relatedness descriptions
        if lateralness < 0.25:
            relatedness_phrase = "directly related and complementary topics"
            creativity_level = "build upon existing knowledge"
        elif lateralness < 0.5:
            relatedness_phrase = "adjacent or tangentially related topics"
            creativity_level = "explore connected domains"
        elif lateralness < 0.75:
            relatedness_phrase = "loosely related or cross-disciplinary topics"
            creativity_level = "make unexpected connections"
        else:
            relatedness_phrase = "completely unrelated, surprising, or wildly creative topics"
            creativity_level = "think outside all conventional boundaries"

        # Get diverse facts from multiple sources
        all_facts = []
        try:
            for source_name, source_func in self.sources.items():
                facts = await source_func()
                all_facts.extend(facts[:5])  # Limit per source
        except Exception as e:
            logger.warning(f"Failed to fetch facts for topic generation: {e}")

        # Enhanced prompt with current trends
        prompt = f"""
        You are a creative AI assistant specializing in generating fascinating and diverse topics for exploration.
        
        **Current Context:**
        Recent topics: {', '.join(recent_topics) if recent_topics else 'No recent topics'}
        
        **Current Trends & Facts:**
        {chr(10).join(all_facts[:10]) if all_facts else 'No current trends available'}
        
        **Task:**
        Generate {n} {relatedness_phrase} that could spark deep curiosity and {creativity_level}.
        
        **Guidelines:**
        - Topics can span any field: hard sciences, philosophy, art, technology, history, speculative ideas
        - Include both concrete and abstract concepts
        - Mix practical and theoretical topics
        - Consider interdisciplinary connections
        - Include some "what if" scenarios or thought experiments
        - Avoid exact repetition of recent topics
        
        **Examples of good topics:**
        - "The mathematics of consciousness and information integration"
        - "How quantum mechanics might explain biological navigation systems"
        - "The philosophy of time in relation to artificial intelligence"
        - "What if gravity worked differently in higher dimensions?"
        - "The intersection of music theory and molecular chemistry"
        
        **Response Format:**
        Return exactly {n} topics as a simple comma-separated list, no numbering or extra text.
        """

        try:
            # Use async LLM call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, call_llm, prompt)

            if not response:
                return self._get_fallback_topics(recent_topics, n, lateralness)

            # Parse topics
            topics = [t.strip().strip('"\'')
                      for t in response.split(',') if t.strip()]
            topics = [t for t in topics if len(
                t) > 5 and len(t) < 200]  # Reasonable length

            # Filter similar topics using embeddings
            filtered_topics = _filter_similar_topics(
                topics, recent_topics, threshold=0.6)

            # Ensure we have enough topics
            if len(filtered_topics) < n and len(topics) > len(filtered_topics):
                # Add some original topics back if filtering was too aggressive
                remaining = [t for t in topics if t not in filtered_topics]
                filtered_topics.extend(remaining[:n - len(filtered_topics)])

            result = filtered_topics[:n]
            _TOPIC_CACHE[cache_key] = result

            logger.info(
                f"Generated {len(result)} curiosity topics with lateralness {lateralness}")
            return result

        except Exception as e:
            logger.warning(f"LLM topic generation failed: {e}")
            return self._get_fallback_topics(recent_topics, n, lateralness)

    def _get_fallback_topics(self, recent_topics: List[str], n: int, lateralness: float) -> List[str]:
        """Fallback topics when LLM fails."""
        fallback_pools = {
            'science': [
                "quantum consciousness theories", "dark matter detection methods",
                "synthetic biology applications", "time crystal physics",
                "neuroplasticity and learning", "extremophile organisms"
            ],
            'philosophy': [
                "the hard problem of consciousness", "moral implications of AI",
                "free will vs determinism", "the nature of mathematical truth",
                "existential risk assessment", "meaning in an infinite universe"
            ],
            'technology': [
                "brain-computer interfaces", "quantum computing applications",
                "nanotechnology ethics", "space elevator engineering",
                "artificial life simulation", "holographic data storage"
            ],
            'creative': [
                "music as a universal language", "color perception across species",
                "the mathematics of beauty", "storytelling in virtual reality",
                "synesthesia and creativity", "architectural psychology"
            ]
        }

        all_topics = []
        for pool in fallback_pools.values():
            all_topics.extend(pool)

        # Simple filtering based on recent topics
        if recent_topics:
            recent_words = set(' '.join(recent_topics).lower().split())
            filtered = []
            for topic in all_topics:
                topic_words = set(topic.lower().split())
                if len(recent_words.intersection(topic_words)) < 2:  # Less than 2 common words
                    filtered.append(topic)
            all_topics = filtered if filtered else all_topics

        random.shuffle(all_topics)
        return all_topics[:n]

    @staticmethod
    def fetch_wikipedia_article(topic: str) -> str:
        """Fetch the full article content for a given topic from Wikipedia using the wikipedia library."""
        try:
            return wikipedia.page(topic, auto_suggest=True).content
        except wikipedia.DisambiguationError as e:
            # Pick the first suggested topic if disambiguation occurs
            try:
                return wikipedia.page(e.options[0], auto_suggest=True).content
            except Exception:
                return ''
        except wikipedia.PageError:
            return ''
        except Exception:
            return ''

    async def trigger(self, recent_topics: List[str], lateralness: float = 1.0) -> Tuple[str, str]:
        """Enhanced async trigger with multiple content sources and autonomous blog integration."""
        try:
            curiosity_topics = await self.get_curiosity_topics_llm(recent_topics, n=10, lateralness=lateralness)
            random.shuffle(curiosity_topics)

            selected_topic = None
            selected_content = None
            selected_prompt = None

            for topic in curiosity_topics:
                # Try multiple content sources
                content_sources = [
                    self._fetch_wikipedia_article_async,
                    self._fetch_topic_summary_async,
                    self._generate_topic_exploration_async
                ]

                for source_func in content_sources:
                    try:
                        content = await source_func(topic)
                        if content and len(content) > 200:
                            prompt = self._create_exploration_prompt(
                                topic, lateralness, len(content))
                            logger.info(
                                f"Successfully triggered curiosity for topic: {topic}")
                            selected_topic = topic
                            selected_content = content
                            selected_prompt = prompt
                            break
                    except Exception as e:
                        logger.warning(
                            f"Content source failed for {topic}: {e}")
                        continue

                if selected_topic:
                    break

            # Fallback if no topic worked
            if not selected_topic:
                selected_topic = curiosity_topics[0] if curiosity_topics else "the nature of curiosity itself"
                selected_content = await self._generate_topic_exploration_async(selected_topic)
                selected_prompt = self._create_exploration_prompt(
                    selected_topic, lateralness, len(selected_content))

            # Register autonomous blog trigger for curiosity discovery
            await self._register_curiosity_blog_trigger(
                selected_topic,
                selected_content,
                recent_topics,
                lateralness
            )

            return selected_content, selected_prompt

        except Exception as e:
            logger.error(f"Curiosity trigger failed: {e}")
            return "Curiosity is the engine of achievement.", "Explore the concept of curiosity and its role in learning and discovery."

    async def _fetch_wikipedia_article_async(self, topic: str) -> str:
        """Async Wikipedia article fetching."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._fetch_wikipedia_sync, topic)
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed for {topic}: {e}")
            return ""

    def _fetch_wikipedia_sync(self, topic: str) -> str:
        """Synchronous Wikipedia fetching for executor."""
        try:
            page = wikipedia.page(topic, auto_suggest=True)
            return page.content
        except wikipedia.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=True)
                return page.content
            except Exception:
                return ""
        except Exception:
            return ""

    async def _fetch_topic_summary_async(self, topic: str) -> str:
        """Generate a comprehensive topic summary using LLM."""
        try:
            prompt = f"""
            Create a comprehensive, engaging exploration of the topic: "{topic}"
            
            Include:
            - Key concepts and definitions
            - Historical context and development
            - Current state of knowledge
            - Interesting facts and examples
            - Open questions and future directions
            - Connections to other fields
            
            Write in an engaging, educational style that sparks curiosity.
            Aim for 800-1200 words.
            """

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, call_llm, prompt)
            return response if response else ""

        except Exception as e:
            logger.warning(f"Topic summary generation failed for {topic}: {e}")
            return ""

    async def _generate_topic_exploration_async(self, topic: str) -> str:
        """Generate an exploratory discussion about the topic."""
        try:
            prompt = f"""
            Write a thought-provoking exploration of "{topic}" that includes:
            
            1. What makes this topic fascinating?
            2. Key questions it raises
            3. How it connects to other areas of knowledge
            4. Practical implications and applications
            5. Philosophical or theoretical considerations
            6. What we still don't understand
            
            Make it intellectually stimulating and curiosity-inducing.
            Length: 600-800 words.
            """

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, call_llm, prompt)
            return response if response else f"An exploration of {topic} and its implications for understanding our world."

        except Exception as e:
            logger.warning(
                f"Topic exploration generation failed for {topic}: {e}")
            return f"Let's explore the fascinating topic of {topic} and its many dimensions."

    def _create_exploration_prompt(self, topic: str, lateralness: float, content_length: int) -> str:
        """Create an engaging prompt for topic exploration."""
        if lateralness > 0.75:
            exploration_style = "wildly creative and unexpected"
        elif lateralness > 0.5:
            exploration_style = "cross-disciplinary and innovative"
        elif lateralness > 0.25:
            exploration_style = "thoughtful and connected"
        else:
            exploration_style = "deep and focused"

        return f"""
        🧠 CURIOSITY TRIGGER ACTIVATED! 🧠
        
        Time for a {exploration_style} exploration of: "{topic}"
        
        This topic was selected to expand your intellectual horizons and spark new connections.
        As you read through this content ({content_length} characters), consider:
        
        • What unexpected connections can you make?
        • How might this relate to your current projects?
        • What questions does this raise?
        • Could this inspire new experiments or investigations?
        
        Let your curiosity guide you through this intellectual adventure!
        
        Content begins below:
        """

    async def from_context_async(self, context: str, lateralness: float = 1.0) -> Tuple[str, str]:
        """Async version of context-based curiosity triggering."""
        try:
            prompt = f"""
            Analyze the following context and extract the main topics, themes, and areas of focus.
            Return only a comma-separated list of the key topics (maximum 10).
            
            Context:
            {context[:2000]}  # Limit context length
            
            Topics:
            """

            loop = asyncio.get_event_loop()
            topics_str = await loop.run_in_executor(None, call_llm, prompt)

            if topics_str:
                recent_topics = [t.strip()
                                 for t in topics_str.split(',') if t.strip()]
                return await self.trigger(recent_topics, lateralness=lateralness)
            else:
                return await self.trigger([], lateralness=lateralness)

        except Exception as e:
            logger.error(f"Context-based curiosity trigger failed: {e}")
            return await self.trigger([], lateralness=lateralness)

    async def _register_curiosity_blog_trigger(
        self,
        topic: str,
        content: str,
        recent_topics: List[str],
        lateralness: float
    ):
        """Register a blog trigger for curiosity discovery."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return

        try:
            self.discovery_count += 1

            # Calculate importance based on lateralness and content quality
            importance_score = min(
                0.9, 0.4 + (lateralness * 0.3) + (len(content) / 2000) * 0.2)

            # Higher importance for creative/unexpected discoveries
            if lateralness > 0.75:
                importance_score += 0.1

            # Determine emotional valence (curiosity is generally positive)
            emotional_valence = 0.3 + (lateralness * 0.4)  # 0.3 to 0.7 range

            # Create reasoning
            reasoning_why = f"""Curiosity was triggered about '{topic}' because it represents a fascinating area 
of exploration that could expand my understanding and spark new connections. The lateralness level 
of {lateralness:.1f} suggests this is {'a highly creative and unexpected' if lateralness > 0.75 else 'a connected and thoughtful'} discovery."""

            reasoning_how = f"""This discovery happened through my curiosity trigger system, which analyzed 
my recent topics ({', '.join(recent_topics[:3])}{'...' if len(recent_topics) > 3 else ''}) and generated 
novel exploration areas. The content was sourced and validated to ensure it provides valuable 
learning opportunities."""

            # Extract meaningful tags
            topic_words = re.findall(r'\b\w{4,}\b', topic.lower())
            tags = ['curiosity', 'discovery', 'learning'] + topic_words[:5]

            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.CURIOSITY_DISCOVERY,
                topic=f"Curiosity Discovery: {topic}",
                context=f"Lateral exploration level: {lateralness:.1f}, Recent context: {', '.join(recent_topics[:5])}",
                learning_content=content[:500] +
                ("..." if len(content) > 500 else ""),
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=importance_score,
                tags=tags,
                metadata={
                    'lateralness': lateralness,
                    'discovery_count': self.discovery_count,
                    'recent_topics': recent_topics,
                    'content_length': len(content)
                }
            )

            logger.info(
                f"Registered curiosity blog trigger for: {topic} (importance: {importance_score:.2f})")

        except Exception as e:
            logger.warning(f"Failed to register curiosity blog trigger: {e}")

    @staticmethod
    def from_context(context: str, lateralness: float = 1.0, blog_scheduler=None) -> Tuple[str, str]:
        """Synchronous version for compatibility."""
        trigger = CuriosityTrigger(blog_scheduler=blog_scheduler)
        return asyncio.run(trigger.from_context_async(context, lateralness))

    @staticmethod
    def trigger_sync(recent_topics: List[str], lateralness: float = 1.0, blog_scheduler=None) -> Tuple[str, str]:
        """Synchronous version for compatibility."""
        trigger = CuriosityTrigger(blog_scheduler=blog_scheduler)
        return asyncio.run(trigger.trigger(recent_topics, lateralness))


# --- Example usage ---
if __name__ == "__main__":
    # Simulate recent topics from other modules
    recent_topics = [
        "machine learning", "neural networks", "python programming",
        "artificial intelligence", "data science", "deep learning",
        "natural language processing", "reinforcement learning",
        "computer vision", "robotics"
    ]

    # Create curiosity trigger with blog scheduler
    if BLOG_SCHEDULER_AVAILABLE:
        from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler
        blog_scheduler = AutonomousBlogScheduler()
        curiosity_trigger = CuriosityTrigger(blog_scheduler=blog_scheduler)
    else:
        curiosity_trigger = CuriosityTrigger()

    fact, prompt = asyncio.run(
        curiosity_trigger.trigger(recent_topics, lateralness=1.0))
    print(prompt)
    print(fact)

    # Example: using full AGI context
    agi_context = """The AGI has recently studied neural networks, deep learning, reinforcement learning,\ncomputer vision, natural language processing, robotics, and data science. It has also explored\nthe basics of quantum computing and ethical AI."""
    curiosity_prompt, article = asyncio.run(
        curiosity_trigger.from_context_async(agi_context, lateralness=0.5))
    print("\n[From Context]", curiosity_prompt)
    print(article[:500])

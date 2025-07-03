import requests
import random
import re
from typing import List, Tuple
from .llm import call_llm  # Import the LLM utility
import wikipedia

class CuriosityTrigger:
    WIKI_DYK_URL = "https://en.wikipedia.org/wiki/Wikipedia:Recent_additions"
    REDDIT_TIL_URL = "https://www.reddit.com/r/todayilearned/top.json?limit=50&t=week"
    USER_AGENT = {'User-agent': 'CuriosityTriggerBot/0.2'}
    WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

    @staticmethod
    def fetch_wikipedia_dyk() -> List[str]:
        """Fetches 'Did you know?' facts from Wikipedia's Recent additions page."""
        try:
            resp = requests.get(CuriosityTrigger.WIKI_DYK_URL)
            if resp.status_code != 200:
                return []
            # Extract facts from the HTML using regex (simple approach)
            facts = re.findall(r'<li>(.*?)</li>', resp.text, re.DOTALL)
            # Clean up HTML tags
            clean_facts = [re.sub('<.*?>', '', fact).strip() for fact in facts]
            # Filter out empty or irrelevant entries
            return [fact for fact in clean_facts if len(fact) > 30]
        except Exception:
            return []

    @staticmethod
    def fetch_reddit_til() -> List[str]:
        """Fetches top 'Today I Learned' facts from Reddit."""
        try:
            resp = requests.get(CuriosityTrigger.REDDIT_TIL_URL, headers=CuriosityTrigger.USER_AGENT)
            if resp.status_code != 200:
                return []
            data = resp.json()
            facts = [post['data']['title'] for post in data['data']['children']]
            # Remove 'TIL that' or 'TIL' prefix
            facts = [re.sub(r'^TIL( that)?[\s:,-]*', '', fact, flags=re.IGNORECASE) for fact in facts]
            return facts
        except Exception:
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

    @staticmethod
    def get_curiosity_topics_llm(recent_topics: List[str], n: int = 5, lateralness: float = 1.0) -> List[str]:
        """Use LLM to suggest curiosity topics that are lateral or tangential to the recent context. Lateralness controls how unrelated the topics should be (0.0=related, 1.0=very lateral)."""
        lateralness = min(max(lateralness, 0.0), 1.0)
        if lateralness < 0.33:
            relatedness_phrase = "closely related or adjacent topics"
        elif lateralness < 0.66:
            relatedness_phrase = "somewhat related or tangential topics"
        else:
            relatedness_phrase = "unrelated, surprising, or lateral topics"
        prompt = (
            f"Given these topics: {', '.join(recent_topics)}\n"
            f"Suggest {n} {relatedness_phrase} that could spark creativity. "
            f"Avoid repeating the exact same topics. Return only a comma-separated list of topics."
        )
        response = call_llm(prompt)
        if not response:
            return []
        topics = [t.strip() for t in response.split(',') if t.strip()]
        return topics

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

    @classmethod
    def trigger(cls, recent_topics: List[str], lateralness: float = 1.0) -> Tuple[str, str]:
        """Use LLM to suggest curiosity topics, fetch a full article about one, and return it with a prompt. Lateralness controls how lateral the topic should be."""
        curiosity_topics = cls.get_curiosity_topics_llm(recent_topics, lateralness=lateralness)
        random.shuffle(curiosity_topics)
        for topic in curiosity_topics:
            article = cls.fetch_wikipedia_article(topic)
            if article and len(article) > 100:
                prompt = (
                    f"Learn something {'lateral' if lateralness > 0.66 else 'related'} to your last 10 topics! Here's a curiosity topic: '{topic}'. "
                    f"Here's the full Wikipedia article for you to explore: "
                )
                return article, prompt
        return super().trigger(recent_topics) if hasattr(super(), 'trigger') else ("No article available.", "Couldn't fetch a curiosity article.")

    @classmethod
    def from_context(cls, context: str, lateralness: float = 1.0) -> Tuple[str, str]:
        """Given a full AGI context string, use the LLM to extract topics, then trigger curiosity. Lateralness controls how lateral the topic should be."""
        prompt = (
            "Given the following context, list the main topics or areas already known. "
            "Return only a comma-separated list of topics.\n\n"
            f"{context}"
        )
        topics_str = call_llm(prompt)
        recent_topics = [t.strip() for t in topics_str.split(',') if t.strip()]
        return cls.trigger(recent_topics, lateralness=lateralness)

# --- Example usage ---
if __name__ == "__main__":
    # Simulate recent topics from other modules
    recent_topics = [
        "machine learning", "neural networks", "python programming",
        "artificial intelligence", "data science", "deep learning",
        "natural language processing", "reinforcement learning",
        "computer vision", "robotics"
    ]
    fact, prompt = CuriosityTrigger.trigger(recent_topics, lateralness=1.0)
    print(prompt)
    print(fact)

    # Example: using full AGI context
    agi_context = """The AGI has recently studied neural networks, deep learning, reinforcement learning,\ncomputer vision, natural language processing, robotics, and data science. It has also explored\nthe basics of quantum computing and ethical AI."""
    curiosity_prompt, article = CuriosityTrigger.from_context(agi_context, lateralness=0.5)
    print("\n[From Context]", curiosity_prompt)
    print(article[:500]) 
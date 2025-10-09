import unittest
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger


class TestCuriosityTrigger(unittest.TestCase):
    def setUp(self):
        self.recent_topics = [
            "machine learning", "neural networks", "python programming",
            "artificial intelligence", "data science", "deep learning",
            "natural language processing", "reinforcement learning",
            "computer vision", "robotics"
        ]

    def test_trigger_returns_article_and_prompt(self):
        article, prompt = CuriosityTrigger.trigger(self.recent_topics)
        self.assertIsInstance(article, str)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(article), 100)
        self.assertIn("curiosity topic", prompt.lower())
        print("\nPrompt:", prompt)
        print("\nArticle (first 500 chars):\n", article[:500])


if __name__ == "__main__":
    unittest.main()

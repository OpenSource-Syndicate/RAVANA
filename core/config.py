import os

class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    FEED_URLS = [
        "http://rss.cnn.com/rss/cnn_latest.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reddit.com/r/worldnews/.rss",
        "https://techcrunch.com/feed/",
        "https://www.npr.org/rss/rss.php?id=1001",
    ] 
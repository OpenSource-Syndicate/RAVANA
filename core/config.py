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

    # Autonomous Loop Settings
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
    LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))
    ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 60))

    # Emotional Intelligence Settings
    POSITIVE_MOODS = ['Confident', 'Curious', 'Reflective']
    NEGATIVE_MOODS = ['Frustrated', 'Stuck', 'Low Energy']

    # Background Task Intervals (in seconds)
    DATA_COLLECTION_INTERVAL = int(os.environ.get("DATA_COLLECTION_INTERVAL", 3600))
    EVENT_DETECTION_INTERVAL = int(os.environ.get("EVENT_DETECTION_INTERVAL", 600)) 
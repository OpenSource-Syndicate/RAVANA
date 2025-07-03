import feedparser
import sqlite3
import time
from collections import Counter
import string
import os

# Define the project root and ensure the database is always at the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DB_FILE = os.path.join(PROJECT_ROOT, 'trends.db')

# --------- SETUP DATABASE -----------
def setup_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT,
            published TEXT,
            source TEXT,
            timestamp INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# --------- SAVE NEW ARTICLES -----------
def save_article(title, link, published, source):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO articles (title, link, published, source, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (title, link, published, source, int(time.time())))
    conn.commit()
    conn.close()

# --------- FETCH RSS FEEDS -----------
def fetch_feeds(feed_urls):
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            published = entry.get('published', 'Unknown')
            save_article(title, link, published, url)

# --------- ANALYZE TRENDS -----------
def analyze_trends(last_hours=24):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    cutoff_time = int(time.time()) - (last_hours * 3600)
    c.execute('SELECT title FROM articles WHERE timestamp >= ?', (cutoff_time,))
    rows = c.fetchall()
    conn.close()

    all_text = " ".join([row[0] for row in rows]).lower()

    # Basic cleaning
    translator = str.maketrans('', '', string.punctuation)
    all_text = all_text.translate(translator)

    words = all_text.split()
    blacklist = set(['the', 'and', 'to', 'a', 'in', 'of', 'for', 'on', 'is', 'at', 'with', 'by', 'from'])
    filtered_words = [word for word in words if word not in blacklist and len(word) > 2]

    word_counts = Counter(filtered_words)
    trending = word_counts.most_common(10)

    print("\n🔥 Top Trending Words (Last", last_hours, "hours):")
    for word, count in trending:
        print(f"{word}: {count} mentions")

# --------- MAIN LOOP -----------
def main():
    setup_db()

    with open('feeds.txt', 'r') as f:
        feed_urls = [line.strip() for line in f if line.strip()]

    while True:
        print("\n🔎 Fetching new feed data...")
        fetch_feeds(feed_urls)
        analyze_trends(last_hours=24)

        print("\n⏳ Sleeping for 15 minutes...\n")
        time.sleep(900)  # 15 minutes

if __name__ == "__main__":
    main()
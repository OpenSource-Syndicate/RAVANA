from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import pipeline
import logging

# Placeholder for a more sophisticated data structure
class Document:
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
        self.embedding = None
        self.cluster_id = None
        self.sentiment = None
        self.is_relevant = True

    def to_dict(self):
        """Return a JSON-serializable dictionary representation."""
        d = self.__dict__.copy()
        if isinstance(d.get('embedding'), np.ndarray):
            d['embedding'] = d['embedding'].tolist()
        return d

class Event:
    def __init__(self, event_id: int, keywords: List[str], summary: str):
        self.event_id = int(event_id)  # Ensure event_id is Python int
        self.keywords = keywords
        self.summary = summary
        self.doc_count = 0
        
    def to_dict(self):
        """Return a JSON-serializable dictionary representation."""
        return {
            "event_id": self.event_id,
            "keywords": self.keywords,
            "summary": self.summary,
            "doc_count": int(self.doc_count)  # Ensure doc_count is Python int
        }

# --- Model Loading ---
# It's better to load models once and reuse them.
embedding_model = None
sentiment_classifier = None

# Get a logger instance
logger = logging.getLogger(__name__)

def load_models():
    """Load all required models."""
    global embedding_model, sentiment_classifier
    
    logger.info("Loading embedding model...")
    # Using a lightweight model for embedding.
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully.")
    
    logger.info("Loading sentiment analysis pipeline...")
    # Using a default sentiment analysis model.
    sentiment_classifier = pipeline('sentiment-analysis')
    logger.info("Sentiment analysis pipeline loaded successfully.")

# --- Core Functions ---

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    if not embedding_model:
        raise ValueError("Embedding model not loaded.")
    return embedding_model.encode(texts, show_progress_bar=False)

def cluster_documents(embeddings: np.ndarray, distance_threshold=0.5) -> np.ndarray:
    """Cluster documents based on their embeddings."""
    # Using Agglomerative Clustering. It's good for finding clusters without knowing the number beforehand.
    clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=distance_threshold)
    clustering_model.fit(embeddings)
    return clustering_model.labels_

def filter_content(documents: List[Document]):
    """Filter documents based on sentiment or other criteria."""
    if not sentiment_classifier:
        raise ValueError("Sentiment classifier not loaded.")
    
    texts_to_analyze = [doc.text for doc in documents]
    sentiments = sentiment_classifier(texts_to_analyze)
    
    for doc, sentiment in zip(documents, sentiments):
        doc.sentiment = sentiment
        # Example filtering: discard negative content. This can be customized.
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
            doc.is_relevant = False

def generate_event_alerts(clustered_docs: Dict[int, List[Document]], min_cluster_size=5) -> List[Event]:
    """Generate alerts for significant event clusters."""
    alerts = []
    for cluster_id, docs in clustered_docs.items():
        if cluster_id == -1 or len(docs) < min_cluster_size:
            continue # Skip noise or small clusters

        # Simple summary: just take the text of the first doc for now.
        # A better approach would be to use a summarization model.
        summary = docs[0].text 
        
        # Simple keywords: for now, just an empty list.
        # A better approach would be to use TF-IDF or a keyword extraction model.
        keywords = [] 
        
        event = Event(event_id=cluster_id, summary=summary, keywords=keywords)
        event.doc_count = len(docs)
        alerts.append(event)
        
    return alerts

def process_data_for_events(texts: List[str]) -> Dict[str, Any]:
    """
    Main function to process a batch of texts to detect events.
    
    Args:
        texts: A list of strings (documents) to analyze.
        
    Returns:
        A dictionary containing detected events and processed documents.
    """
    if not embedding_model or not sentiment_classifier:
        # This is a safeguard, but models should be loaded at startup.
        logger.warning("Models not pre-loaded. Loading them now. This may take a moment.")
        load_models()

    # 1. Create Document objects
    documents = [Document(text) for text in texts]
    
    # 2. Sentiment & Keyword Filtering (initial pass)
    filter_content(documents)
    
    relevant_documents = [doc for doc in documents if doc.is_relevant]
    if not relevant_documents:
        return {"events": [], "documents": documents, "message": "No relevant documents after filtering."}
        
    relevant_texts = [doc.text for doc in relevant_documents]

    # 3. Event Tracking & Filtering Topic Detection
    # Generate embeddings
    embeddings = get_embeddings(relevant_texts)
    for doc, emb in zip(relevant_documents, embeddings):
        doc.embedding = emb
        
    # Cluster documents
    cluster_labels = cluster_documents(embeddings)
    
    clustered_docs = {}
    for doc, label in zip(relevant_documents, cluster_labels):
        doc.cluster_id = int(label)
        if label not in clustered_docs:
            clustered_docs[label] = []
        clustered_docs[label].append(doc)

    # 4. Alert Generation
    events = generate_event_alerts(clustered_docs)
    
    # Here you would typically save the events to a database or send a notification.
    # For now, we just return them.
    
    return {
        "events": [event.__dict__ for event in events],
        "documents": [doc.to_dict() for doc in documents]
    }

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        "Massive solar flare expected to hit Earth tomorrow.",
        "New study shows coffee can improve memory.",
        "Scientists are amazed by the recent solar activity.",
        "The government announced new tax cuts for small businesses.",
        "Another report on solar flares causing potential power outages.",
        "This is some offensive content that should be filtered out.",
        "Local sports team wins the championship.",
        "Experts warn about the impact of the incoming solar storm.",
        "I really hate this, it's terrible.",
        "Researchers find a link between caffeine and alertness."
    ]
    
    results = process_data_for_events(sample_data)
    
    print("--- Detected Events ---")
    for event in results['events']:
        print(f"Event ID: {event['event_id']}, Docs: {event['doc_count']}, Summary: {event['summary']}")
        
    print("\n--- Processed Documents ---")
    for doc in results['documents']:
        print(f"Text: {doc['text'][:30]}... | Relevant: {doc['is_relevant']} | Cluster: {doc['cluster_id']} | Sentiment: {doc['sentiment']}")
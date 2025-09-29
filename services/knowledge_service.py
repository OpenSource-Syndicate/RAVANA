import hashlib
import logging
import numpy as np
import pickle
import os
from datetime import datetime
from typing import List, Dict, Any
from modules.knowledge_compression.main import compress_knowledge
from sqlmodel import Session, select
from database.models import Summary
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, engine, embedding_model=None):
        self.engine = engine
        self.embedding_model = embedding_model or SentenceTransformer(
            'all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index for semantic search
        self.faiss_index = None
        self.id_map = []  # Maps FAISS index positions to database IDs
        self.index_file = "knowledge_index.faiss"
        self.id_map_file = "knowledge_id_map.pkl"

        self._initialize_semantic_search()

    def _initialize_semantic_search(self):
        """
        Load or create a FAISS index for semantic vector search.
        If faiss isn't installed or something goes wrong, we fall back
        to disabling semantic search (faiss_index remains None).
        """
        try:
            import faiss
        except ModuleNotFoundError:
            logger.warning(
                "Faiss library not found. Semantic search disabled.")
            self.faiss_index = None
            self.id_map = []
            return

        # Try loading existing index + id_map from disk
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.id_map_file):
                logger.info("Loading existing FAISS index and ID map")
                self.faiss_index = faiss.read_index(self.index_file)
                with open(self.id_map_file, "rb") as f:
                    self.id_map = pickle.load(f)
            else:
                logger.info("No existing FAISS index found. Creating new one.")
                # flat L2 index; change to IndexFlatIP for cosine/IP if you prefer
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                self.id_map = []

            # OPTIONAL: on very first run you may want to pull _all_ summaries
            # and add them into the new index so later searches see everything.
            if len(self.id_map) == 0:
                with Session(self.engine) as session:
                    all_summaries = session.exec(select(Summary)).all()
                if all_summaries:
                    texts = [s.summary_text for s in all_summaries]
                    embeddings = self.embedding_model.encode(
                        texts, convert_to_numpy=True)
                    # ensure dtype float32
                    embeddings = np.array(embeddings, dtype=np.float32)
                    self.faiss_index.add(embeddings)
                    self.id_map = [s.id for s in all_summaries]
                    # persist to disk
                    faiss.write_index(self.faiss_index, self.index_file)
                    with open(self.id_map_file, "wb") as f:
                        pickle.dump(self.id_map, f)

            logger.info("Semantic search index initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize semantic search: {e}", exc_info=True)
            self.faiss_index = None
            self.id_map = []

    def add_knowledge(self, content: str, source: str = "unknown", category: str = "misc") -> dict:
        """
        Add new knowledge by summarizing content and saving it with metadata.
        Returns the summary dict with metadata.
        """
        if not content or not content.strip():
            raise ValueError("No content provided to add_knowledge")

        # Create content hash for deduplication
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        # Check if we already have this content
        with Session(self.engine) as session:
            existing = session.exec(
                select(Summary).where(Summary.content_hash == content_hash)
            ).first()
            if existing:
                logger.info(
                    f"Content already exists in knowledge base (hash: {content_hash[:8]}...)")
                return {
                    "timestamp": existing.timestamp,
                    "summary": existing.summary_text,
                    "source": existing.source,
                    "category": existing.category,
                    "duplicate": True
                }

        try:
            # Compress/summarize the content
            summary_data = compress_knowledge(
                {"text": content, "source": source, "category": category})

            # Save to database
            with Session(self.engine) as session:
                summary_obj = Summary(
                    timestamp=summary_data["timestamp"],
                    summary_text=summary_data["summary"],
                    source=source,
                    category=category,
                    content_hash=content_hash
                )
                session.add(summary_obj)
                session.commit()
                session.refresh(summary_obj)

            # Update FAISS index with the new summary
            try:
                if self.faiss_index is not None:
                    emb = self.embedding_model.encode(
                        summary_data["summary"], convert_to_numpy=True
                    ).astype(np.float32)
                    # faiss expects shape (n, d)
                    self.faiss_index.add(np.expand_dims(emb, axis=0))
                    self.id_map.append(summary_obj.id)
                    # persist index + id_map
                    import faiss
                    faiss.write_index(self.faiss_index, self.index_file)
                    with open(self.id_map_file, "wb") as f:
                        pickle.dump(self.id_map, f)
            except Exception as e:
                logger.warning(
                    f"Failed to update FAISS index for new knowledge: {e}")

            logger.info(
                f"Added new knowledge: {category}/{source} (hash: {content_hash[:8]}...)")
            return {
                "timestamp": summary_data["timestamp"],
                "summary": summary_data["summary"],
                "source": source,
                "category": category,
                "duplicate": False,
                "id": summary_obj.id
            }

        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            raise

    def get_knowledge_by_category(self, category: str, limit: int = 10) -> List[dict]:
        """Retrieve knowledge entries by category."""
        with Session(self.engine) as session:
            stmt = select(Summary).where(
                Summary.category == category).limit(limit)
            results = session.exec(stmt).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "summary": r.summary_text,
                    "source": r.source,
                    "category": r.category
                }
                for r in results
            ]

    def get_recent_knowledge(self, hours: int = 24, limit: int = 20) -> List[dict]:
        """Get recent knowledge entries."""
        cutoff_time = datetime.utcnow().replace(microsecond=0).isoformat()
        with Session(self.engine) as session:
            stmt = (select(Summary)
                    .where(Summary.timestamp >= cutoff_time)
                    .order_by(Summary.timestamp.desc())
                    .limit(limit))
            results = session.exec(stmt).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "summary": r.summary_text,
                    "source": r.source,
                    "category": r.category
                }
                for r in results
            ]

    def search_knowledge(self, query: str, limit: int = 10) -> List[dict]:
        """Simple text search in knowledge summaries."""
        with Session(self.engine) as session:
            # Simple LIKE search - could be enhanced with full-text search
            stmt = (select(Summary)
                    .where(Summary.summary_text.contains(query))
                    .limit(limit))
            results = session.exec(stmt).all()
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "summary": r.summary_text,
                    "source": r.source,
                    "category": r.category,
                    "relevance_score": self._calculate_relevance(query, r.summary_text)
                }
                for r in results
            ]

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Simple relevance scoring based on keyword matches."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words:
            return 0.0
        matches = len(query_words.intersection(text_words))
        return matches / len(query_words)

    def compress_and_save_knowledge(self):
        """Enhanced version with better logging and error handling."""
        try:
            # Get recent interactions from database instead of placeholder
            with Session(self.engine) as session:
                # Get recent summaries to avoid re-compressing the same data
                recent_summaries = session.exec(
                    select(Summary).order_by(
                        Summary.timestamp.desc()).limit(50)
                ).all()

                if not recent_summaries:
                    logs = {"recent_interactions": [
                        "No recent data available"]}
                else:
                    logs = {
                        "recent_interactions": [
                            f"{s.category}/{s.source}: {s.summary_text[:200]}..."
                            for s in recent_summaries
                        ]
                    }

            summary = compress_knowledge(logs)

            with Session(self.engine) as session:
                summary_obj = Summary(
                    timestamp=summary["timestamp"],
                    summary_text=summary["summary"],
                    source="system",
                    category="compression",
                    content_hash=hashlib.sha256(
                        summary["summary"].encode()).hexdigest()
                )
                session.add(summary_obj)
                session.commit()

            logger.info("Successfully compressed and saved knowledge")
            return summary

        except Exception as e:
            logger.error(f"Failed to compress knowledge: {e}", exc_info=True)
            raise

    # Methods for graceful shutdown compatibility
    async def prepare_shutdown(self) -> bool:
        """
        Prepare knowledge service for shutdown.
        """
        logger.info("Knowledge service preparing for shutdown")
        return True

    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown the knowledge service with timeout.
        Ensures FAISS index is properly saved.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        logger.info("Shutting down Knowledge Service...")
        try:
            # Attempt to save FAISS index and id_map during shutdown
            if self.faiss_index is not None:
                import faiss
                faiss.write_index(self.faiss_index, self.index_file)
                with open(self.id_map_file, "wb") as f:
                    pickle.dump(self.id_map, f)
                logger.info("FAISS index and ID map saved during shutdown")
            return True
        except Exception as e:
            logger.error(f"Error during Knowledge Service shutdown: {e}")
            return False

    def get_shutdown_metrics(self) -> Dict[str, Any]:
        """
        Get shutdown-related metrics for this component.

        Returns:
            Dict containing shutdown metrics
        """
        return {
            "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "id_map_size": len(self.id_map),
            "index_file": self.index_file,
            "id_map_file": self.id_map_file
        }

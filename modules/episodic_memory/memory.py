from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn # For running the server
import re
from datetime import datetime, timedelta
import os

from .llm import call_llm
import sqlite3
import json # For storing embeddings as JSON strings
import numpy as np
from sentence_transformers import SentenceTransformer # For generating embeddings
import logging
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="MemoryDB API",
    description="An API to extract, store, and retrieve memories from conversations.",
    version="0.1.0"
)

# The embedding model will be set by the main AGI system before starting the server.
embedding_model: Optional[SentenceTransformer] = None

DATABASE_NAME = 'memory.db'

PROMPT_FOR_EXTRACTING_MEMORIES_CONVO = """
You are a memory extractor. You need to extract memories for the conversation and return them as a list of distinct facts or statements. 
Return only the memories — no commentary, no formatting. Do not store everything into memory.
Just extract key facts, recurring user preferences, major goals, and meaningful statements that help build context later.
Example: 
- User likes French cuisine.
- User is planning a trip to Paris.
- User's favorite color is blue.
- Memories should be concise (15-30 words max)
- Prefer long-term facts over transient context
- Include dates/times where relevant
"""

PROMPT_FOR_CONSOLIDATION = """
You are a memory consolidation system. Analyze these memories and:
1. Remove duplicates and near-duplicates
2. Merge related facts into comprehensive statements
3. Identify and flag outdated information (older than 1 year)
4. Resolve contradictions (favor newer information)
5. Preserve all key information in a condensed format

Return ONLY a JSON list of consolidated memories with structure:
[{
  "text": "Consolidated memory text",
  "source_ids": [1, 2, 3],
  "status": "active"|"outdated"
}]
"""

# Pydantic Models for API requests and responses
class ConversationRequest(BaseModel):
    user_input: str = Field(..., example="Hi, I'm planning a trip to Paris.")
    ai_output: str = Field(..., example="Sounds great!")

class MemoriesList(BaseModel):
    memories: List[str] = Field(..., example=["User is planning a trip to Paris.", "User likes French cuisine."])
    type: Optional[str] = Field('long-term', example='episodic')

class QueryRequest(BaseModel):
    query_text: str = Field(..., example="What are the user's travel plans?")
    top_n: Optional[int] = Field(5, example=3)
    similarity_threshold: Optional[float] = Field(0.7, example=0.65)

class MemoryRecord(BaseModel):
    id: int
    text: str
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    type: str = 'long-term'

class ConsolidateRequest(BaseModel):
    min_similarity: Optional[float] = Field(0.75, example=0.8)
    max_age_days: Optional[int] = Field(365, example=90)

class RelevantMemory(BaseModel):
    id: int
    text: str
    similarity: float

class RelevantMemoriesResponse(BaseModel):
    relevant_memories: List[RelevantMemory]

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Any] = None

# ChromaDB setup
CHROMA_COLLECTION = 'memories'
chroma_client = chromadb.Client(Settings(persist_directory="chroma_db"))
chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

def init_db():
    """Initializes the SQLite database and creates the memories table if it doesn't exist."""
    try:
        # Ensure the database directory exists
        db_dir = os.path.dirname(os.path.abspath(DATABASE_NAME))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE, -- Ensure memories are unique
                embedding TEXT NOT NULL,    -- Store embedding as JSON string
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME,
                access_count INTEGER DEFAULT 0,
                type TEXT DEFAULT 'long-term'
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"Database '{DATABASE_NAME}' initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

def migrate_db():
    """Add new columns to database schema if needed"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Add last_accessed and access_count if missing
    cursor.execute("PRAGMA table_info(memories)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'last_accessed' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN last_accessed DATETIME")
    if 'access_count' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
    if 'type' not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN type TEXT DEFAULT 'long-term'")
    
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    """Initializes the database when the FastAPI application starts."""
    init_db()
    # The embedding model is now set globally from main.py before startup
    global embedding_model
    embedding_model = app.embedding_model if hasattr(app, "embedding_model") else None
    
    if not embedding_model:
        logging.warning("Embedding model could not be loaded or was not passed from the main application.")
    # Add new columns if missing
    migrate_db()


def get_embedding(text):
    """Generates an embedding for the given text."""
    if embedding_model and text:
        try:
            return embedding_model.encode(text)
        except Exception as e:
            logging.error(f"Error generating embedding for '{text}': {e}")
            return None
    return None

@app.post("/extract_memories/", response_model=MemoriesList, tags=["Memories"])
async def extract_memories_api(request: ConversationRequest):
    """Extracts memories from a given conversation text."""
    # If called directly, request might be a dict. Convert to ConversationRequest.
    if isinstance(request, dict):
        request = ConversationRequest(**request)

    try:
        conversation = f"User: {request.user_input}\nAI: {request.ai_output}"
        prompt = PROMPT_FOR_EXTRACTING_MEMORIES_CONVO + "\nConversation:\n" + conversation
        logging.info("Sending memory extraction request to LLM...")
        llm_response = call_llm(prompt)
        if llm_response:
            memories_list = [m.strip().lstrip('-* ') for m in llm_response.split('\n') if m.strip().lstrip('-* ')]
            logging.info(f"Extracted memories from LLM: {memories_list}")
            return MemoriesList(memories=memories_list)
        logging.warning("LLM did not return a response for memory extraction.")
        return MemoriesList(memories=[])
    except Exception as e:
        logging.error(f"Error in extract_memories_api: {e}")
        return MemoriesList(memories=[])

@app.post("/save_memories/", response_model=StatusResponse, tags=["Memories"])
async def save_memories_api(memories_request: MemoriesList):
    """Saves a list of memories to the database."""
    memories_to_save = memories_request.memories
    memory_type = memories_request.type or 'long-term'
    return save_memories(memories_to_save, memory_type=memory_type)

def save_memories(memories_to_save, memory_type='long-term'):
    """Saves memories with enhanced deduplication and consolidation"""
    if not embedding_model:
        logging.error("Embedding model not available. Cannot save memories.")
        return StatusResponse(status="error", message="Embedding model not available")

    if isinstance(memories_to_save, str):
        memories_list = [m.strip() for m in memories_to_save.split('\n') if m.strip()]
    elif isinstance(memories_to_save, list):
        memories_list = memories_to_save
    else:
        logging.warning(f"save_memories received an unexpected type: {type(memories_to_save)}. Attempting to process as a single item list.")
        memories_list = [str(memories_to_save)]

    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        saved_count = 0
        updated_count = 0
        similarity_threshold = 0.75  # Lowered threshold for better memory capture

        # First, get all existing memories and their embeddings
        cursor.execute("SELECT id, text, embedding FROM memories")
        existing_memories = cursor.fetchall()
        existing_embeddings = []
        for id, text, emb in existing_memories:
            try:
                emb_array = np.array(json.loads(emb))
                existing_embeddings.append((id, text, emb_array))
            except Exception as e:
                logging.error(f"Failed to parse embedding for memory {id}: {e}")
                continue

        for memory_text in memories_list:
            if not memory_text.strip():
                continue

            # Generate enhanced embedding
            new_embedding = get_embedding(memory_text)
            if new_embedding is None:
                logging.warning(f"Could not generate embedding for '{memory_text}'. Skipping.")
                continue

            # Save to ChromaDB
            try:
                chroma_collection.add(
                    documents=[memory_text],
                    embeddings=[new_embedding.tolist()],
                    metadatas=[{"type": memory_type}],
                    ids=[str(hash(memory_text))]
                )
            except Exception as e:
                logging.error(f"Failed to add memory to ChromaDB: {e}")

            # Find similar memories with enhanced similarity
            most_similar_id = None
            highest_sim = 0
            best_match = None
            
            for mem_id, mem_text, stored_emb in existing_embeddings:
                try:
                    similarity = enhanced_similarity(
                        new_embedding, 
                        stored_emb, 
                        memory_text, 
                        mem_text
                    )
                    
                    if similarity > highest_sim and similarity >= similarity_threshold:
                        highest_sim = similarity
                        most_similar_id = mem_id
                        best_match = mem_text
                except Exception as e:
                    logging.error(f"Failed to compute similarity for memory {mem_id}: {e}")
                    continue

            # Handle consolidation cases
            if most_similar_id is not None:
                if should_consolidate(memory_text, best_match):
                    consolidated = consolidate_pair(memory_text, best_match)
                    if consolidated:
                        try:
                            # Update with consolidated version
                            new_emb = get_embedding(consolidated)
                            if new_emb is not None:
                                cursor.execute("""
                                    UPDATE memories 
                                    SET text = ?, embedding = ?, timestamp = CURRENT_TIMESTAMP
                                    WHERE id = ?
                                """, (consolidated, json.dumps(new_emb.tolist()), most_similar_id))
                                updated_count += 1
                                logging.info(f"Consolidated memory: {best_match} + {memory_text} → {consolidated}")
                        except Exception as e:
                            logging.error(f"Consolidation failed: {e}")
                            # Fallback to saving as new memory
                            try:
                                cursor.execute("INSERT INTO memories (text, embedding, type) VALUES (?, ?, ?)", 
                                           (memory_text, json.dumps(new_embedding.tolist()), memory_type))
                                saved_count += 1
                            except Exception as e:
                                logging.error(f"Failed to save memory after consolidation failure: {e}")
                else:
                    # Keep both memories if they're different enough
                    try:
                        cursor.execute("INSERT INTO memories (text, embedding, type) VALUES (?, ?, ?)", 
                                   (memory_text, json.dumps(new_embedding.tolist()), memory_type))
                        saved_count += 1
                    except sqlite3.IntegrityError:
                        logging.info(f"Identical memory '{memory_text}' already exists. Skipping.")
                    except Exception as e:
                        logging.error(f"Failed to save memory: {e}")
            else:
                try:
                    cursor.execute("INSERT INTO memories (text, embedding, type) VALUES (?, ?, ?)", 
                               (memory_text, json.dumps(new_embedding.tolist()), memory_type))
                    saved_count += 1
                    logging.info(f"Saved new memory: '{memory_text}' (type={memory_type})")
                except sqlite3.IntegrityError:
                    logging.info(f"Identical memory '{memory_text}' already exists. Skipping.")
                except Exception as e:
                    logging.error(f"Failed to save memory '{memory_text}': {e}")

        conn.commit()
        
        total_changes = saved_count + updated_count
        if total_changes > 0:
            message = f"Successfully saved {saved_count} new memories and updated {updated_count} existing memories."
            logging.info(message)
            return StatusResponse(status="success", message=message)
        elif not memories_list:
            return StatusResponse(status="no_action", message="No valid memories provided to save.")
        else:
            return StatusResponse(status="no_new_memories_saved", 
                                message="No new memories were saved. They might be similar to existing ones, already exist, or embeddings failed.")
    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"Error in save_memories: {e}")
        return StatusResponse(status="error", message=f"Failed to save memories: {str(e)}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")


def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/get_relevant_memories/", response_model=RelevantMemoriesResponse, tags=["Memories"])
async def get_relevant_memories_api(request: QueryRequest):
    """Retrieves relevant memories based on a query string. Uses LLM if no relevant memories are found."""
    global embedding_model
    # If called directly, request might be a dict. Convert to QueryRequest.
    if isinstance(request, dict):
        request = QueryRequest(**request)

    # Ensure the embedding model and DB are initialized for direct calls.
    if not embedding_model and hasattr(app, "embedding_model"):
        logging.info("Initializing memory module for direct call...")
        embedding_model = app.embedding_model
        init_db()
        migrate_db()
        
    query = request.query_text
    top_n = request.top_n
    similarity_threshold = request.similarity_threshold
    
    if not embedding_model:
        logging.error("Embedding model not available. Cannot retrieve relevant memories.")
        return RelevantMemoriesResponse(relevant_memories=[])

    query_embedding = get_embedding(query)
    if query_embedding is None:
        logging.warning(f"Could not generate embedding for query '{query}'.")
        return RelevantMemoriesResponse(relevant_memories=[])

    # Use ChromaDB ANN search
    try:
        results = chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n,
            include=['documents', 'metadatas', 'distances']
        )
        relevant_memories = []
        for i, doc in enumerate(results['documents'][0]):
            sim = 1.0 - results['distances'][0][i]  # Chroma returns L2 distance by default
            if sim >= similarity_threshold:
                relevant_memories.append(RelevantMemory(
                    id=int(results['ids'][0][i]),
                    text=doc,
                    similarity=sim
                ))
        return RelevantMemoriesResponse(relevant_memories=relevant_memories)
    except Exception as e:
        logging.error(f"ChromaDB query failed: {e}. Falling back to SQLite.")
        # Fallback to SQLite search (existing logic)
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        try:
            # Get all memories with their embeddings
            cursor.execute("SELECT id, text, embedding, last_accessed, access_count FROM memories")
            all_memories = cursor.fetchall()
            
            if not all_memories:
                logging.info("No memories found in the database.")
                return RelevantMemoriesResponse(relevant_memories=[])

            relevant_memories = []
            for mem_id, text, stored_embedding_json, last_accessed, access_count in all_memories:
                stored_embedding = np.array(json.loads(stored_embedding_json))
                similarity = cosine_similarity(query_embedding, stored_embedding)
                
                # Apply recency and frequency boost
                recency_boost = 1.0
                if last_accessed:
                    days_since_access = (datetime.now() - datetime.fromisoformat(last_accessed)).days
                    recency_boost = max(0.5, 1.0 - (days_since_access / 30))  # Decay over 30 days
                
                frequency_boost = min(1.5, 1.0 + (access_count / 10))  # Cap at 1.5x boost
                
                adjusted_similarity = similarity * recency_boost * frequency_boost
                
                if adjusted_similarity >= similarity_threshold:
                    relevant_memories.append({
                        'id': mem_id, 
                        'text': text, 
                        'similarity': adjusted_similarity,
                        'base_similarity': similarity,
                        'recency_boost': recency_boost,
                        'frequency_boost': frequency_boost
                    })
            
            # Sort by adjusted similarity in descending order and take top_n
            relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
            top_memories = relevant_memories[:top_n]
            
            # Update access info for relevant memories
            for memory in top_memories:
                cursor.execute("""
                    UPDATE memories 
                    SET last_accessed = CURRENT_TIMESTAMP, 
                        access_count = access_count + 1 
                    WHERE id = ?
                """, (memory['id'],))
            
            conn.commit()
            
            # If no relevant memories found, use LLM to try to answer from all memories
            if not top_memories:
                # Compose a context of all memories
                all_texts = [row[1] for row in all_memories]  # Get just the text
                if all_texts:
                    context = "\n".join(f"- {t}" for t in all_texts)
                    llm_prompt = (
                        f"You are a memory retrieval assistant. Given the following stored memories and a user query, "
                        f"return the most relevant memory or fact (if any) that answers the query.\n"
                        f"Memories:\n{context}\n\nQuery: {query}\n"
                        f"If none of the memories are relevant, reply with an empty string."
                    )
                    llm_response = call_llm(llm_prompt)
                    if llm_response and llm_response.strip():
                        # Try to match the LLM response to a memory
                        for mem_id, text, _, _, _ in all_memories:
                            if llm_response.strip() in text or text in llm_response.strip():
                                return RelevantMemoriesResponse(relevant_memories=[{
                                    'id': mem_id, 
                                    'text': text, 
                                    'similarity': 1.0,
                                    'base_similarity': 1.0,
                                    'recency_boost': 1.0,
                                    'frequency_boost': 1.0
                                }])
                        # If not found, return as a synthetic memory
                        return RelevantMemoriesResponse(relevant_memories=[{
                            'id': -1, 
                            'text': llm_response.strip(), 
                            'similarity': 1.0,
                            'base_similarity': 1.0,
                            'recency_boost': 1.0,
                            'frequency_boost': 1.0
                        }])
            
            logging.info(f"Query: '{query}', Found {len(top_memories)} relevant memories (threshold={similarity_threshold})")
            return RelevantMemoriesResponse(relevant_memories=top_memories)
        except Exception as e:
            logging.error(f"Error in get_relevant_memories: {e}")
            return RelevantMemoriesResponse(relevant_memories=[])
        finally:
            conn.close()

@app.post("/consolidate_memories/", response_model=StatusResponse, tags=["Memories"])
async def consolidate_memories_api(request: ConsolidateRequest):
    """Run full memory consolidation using LLM"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Get all memories
    cursor.execute("SELECT id, text FROM memories")
    all_memories = cursor.fetchall()
    
    if not all_memories:
        return StatusResponse(status="skipped", message="No memories to consolidate")
    
    # Prepare prompt input
    memories_list = [f"{id}: {text}" for id, text in all_memories]
    prompt = PROMPT_FOR_CONSOLIDATION + "\nMemories:\n" + "\n".join(memories_list)
    
    # Get LLM consolidation plan
    try:
        response = call_llm(prompt)
        consolidation_plan = parse_consolidation_response(response)
        
        # Process consolidation plan
        deleted_ids = set()
        new_memories = []
        
        for item in consolidation_plan:
            # Process outdated memories
            if item.get("status") == "outdated":
                for mem_id in item["source_ids"]:
                    cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                    deleted_ids.add(mem_id)
                continue
                
            # Create new consolidated memory
            if "source_ids" in item and "text" in item:
                new_memory = item["text"]
                new_embedding = get_embedding(new_memory)
                
                if new_embedding is not None:
                    cursor.execute("""
                        INSERT INTO memories (text, embedding) 
                        VALUES (?, ?)
                    """, (new_memory, json.dumps(new_embedding.tolist())))
                    
                    # Remove source memories
                    for mem_id in item["source_ids"]:
                        cursor.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                        deleted_ids.add(mem_id)
                    new_memories.append(new_memory)
        
        conn.commit()
        return StatusResponse(
            status="success",
            message=f"Consolidated {len(consolidation_plan)} groups, "
                    f"deleted {len(deleted_ids)} memories, "
                    f"added {len(new_memories)} consolidated memories"
        )
    except Exception as e:
        logging.error(f"Consolidation failed: {str(e)}")
        return StatusResponse(status="error", message=str(e))
    finally:
        conn.close()

@app.get("/list_memories/", response_model=List[MemoryRecord], tags=["Memories"])
async def list_memories_api(limit: int = 100, min_accesses: int = 0):
    """List memories with usage statistics"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, timestamp, last_accessed, access_count
        FROM memories
        WHERE access_count >= ?
        ORDER BY last_accessed DESC
        LIMIT ?
    """, (min_accesses, limit))
    
    memories = [
        MemoryRecord(
            id=row[0],
            text=row[1],
            created_at=row[2],
            last_accessed=row[3],
            access_count=row[4]
        )
        for row in cursor.fetchall()
    ]
    conn.close()
    return memories

def enhanced_similarity(vec1, vec2, text1, text2):
    """Computes weighted similarity considering both semantic and lexical features"""
    # Semantic similarity (70% weight)
    semantic_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Lexical similarity (30% weight)
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    lexical_sim = len(set1 & set2) / max(len(set1 | set2), 1)
    
    return 0.7 * semantic_sim + 0.3 * lexical_sim

def should_consolidate(new_memory, existing_memory):
    """Determine if two memories should be consolidated"""
    if not new_memory or not existing_memory:
        return False

    # Get embeddings
    new_emb = get_embedding(new_memory)
    existing_emb = get_embedding(existing_memory)
    
    if not new_emb is not None and existing_emb is not None:
        # Always consolidate if similarity > 85%
        if enhanced_similarity(new_emb, existing_emb, new_memory, existing_memory) > 0.85:
            return True
            
    # Consolidate complementary information
    keywords = ["but", "however", "except", "while", "though"]
    return not any(kw in new_memory.lower() or kw in existing_memory.lower() for kw in keywords)

def consolidate_pair(memory1, memory2):
    """Use LLM to consolidate two related memories"""
    prompt = f"""
    Consolidate these related memories into a single comprehensive statement:
    1. {memory1}
    2. {memory2}
    
    Rules:
    - Preserve all key information
    - Maintain factual accuracy
    - Keep concise (max 30 words)
    - Resolve any contradictions (favor newer information)
    
    Return ONLY the consolidated memory text with NO additional formatting.
    """
    
    try:
        response = call_llm(prompt)
        if response and len(response) < 150:  # Sanity check length
            return response.strip()
    except Exception as e:
        logging.error(f"Consolidation failed: {e}")
    return None

def parse_consolidation_response(response):
    """Parse LLM consolidation response into structured data"""
    try:
        # Extract JSON from response
        json_str = re.search(r'\[.*\]', response, re.DOTALL)
        if json_str:
            return json.loads(json_str.group(0))
    except json.JSONDecodeError:
        logging.error("Invalid JSON in consolidation response")
    return []

@app.get("/health", response_model=StatusResponse, tags=["System"])
async def health_check():
    """Performs a health check of the API and its dependencies."""
    db_ok = False
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.execute("SELECT 1 FROM memories LIMIT 1") # Simple query to check DB connection and table
        conn.close()
        db_ok = True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        db_ok = False

    embedding_model_ok = embedding_model is not None

    # Add consolidation health check
    try:
        test_consolidation = parse_consolidation_response('[{"test": "valid"}]')
        consolidation_ok = bool(test_consolidation)
    except Exception:
        consolidation_ok = False

    if db_ok and embedding_model_ok and consolidation_ok:
        return StatusResponse(status="ok", message="API is healthy.")
    else:
        details = {
            "database_status": "ok" if db_ok else "error",
            "embedding_model_status": "ok" if embedding_model_ok else "error",
            "consolidation_system": "ok" if consolidation_ok else "error"
        }
        raise HTTPException(status_code=503, detail=details)


if __name__ == "__main__":
    # This block is now for running the Uvicorn server directly for development.
    # In production, you might use a process manager like Gunicorn with Uvicorn workers.
    logging.info("Starting Uvicorn server for MemoryDB API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", timeout_keep_alive=120)

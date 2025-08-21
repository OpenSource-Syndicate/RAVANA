from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn # For running the server
import re
from datetime import datetime, timedelta
import os
import uuid
import asyncio
from core.llm import call_llm
import json # For storing embeddings as JSON strings
import numpy as np
from sentence_transformers import SentenceTransformer # For generating embeddings
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="MemoryDB API",
    description="An API to extract, store, and retrieve memories from conversations.",
    version="0.2.0"
)

# The embedding model will be set by the main AGI system.
embedding_model: Optional[SentenceTransformer] = None

PROMPT_FOR_EXTRACTING_MEMORIES_CONVO = """
You are a memory extraction module for an AGI. Your task is to analyze a conversation and identify key pieces of information to be stored in the AGI's long-term memory. These memories help the AGI build a consistent understanding of its interactions and the world.

Focus on extracting:
- **Key facts:** "The user's company is named 'Innovate Inc'."
- **User preferences:** "The user prefers concise and direct answers."
- **Major goals or plans:** "The user is working on a project to analyze climate data."
- **Core beliefs or perspectives:** "The user believes AI should be developed ethically."

Guidelines:
- Each memory should be a single, self-contained statement.
- Keep memories concise (under 20 words is ideal).
- Prefer information that is likely to be relevant long-term.
- Do not store transitory conversational details (e.g., "The user said hello").
- Output the memories as a JSON object containing a list of strings.

Example Output:
{
  "memories": [
    "User is planning a trip to Paris.",
    "User's favorite color is blue."
  ]
}

Conversation to analyze:
"""

PROMPT_FOR_CONSOLIDATION = """
You are a memory consolidation module for an AGI. Your task is to refine a list of existing memories to make them more efficient and coherent.

Analyze the following list of memories and perform these actions:
1.  **Merge:** Combine related memories into a single, more comprehensive statement. For example, "User likes coffee" and "User drinks espresso in the morning" can be merged into "User is a coffee drinker, preferring espresso in the mornings."
2.  **Deduplicate:** Identify and remove memories that state the same fact in slightly different ways. Keep the most detailed version.
3.  **Generalize:** If there are many specific but related facts, create a more general summary memory. For example, memories of buying apples, bananas, and oranges could be generalized to "User frequently buys fruit."

Rules:
-   Do not lose critical information.
-   Do not merge unrelated facts.
-   Return ONLY a JSON object with two keys: "consolidated" and "to_delete".
    -   "consolidated": A list of new or updated memory strings.
    -   "to_delete": A list of the IDs of the original memories that have been consolidated or are redundant.

Example Input Memories:
[
  {"id": "mem_1", "text": "The user is a fan of sci-fi movies."},
  {"id": "mem_2", "text": "The user recently watched 'Dune'."},
  {"id": "mem_3", "text": "The user enjoys science fiction films."},
  {"id": "mem_4", "text": "The user's cat is named 'Leo'."}
]

Example Output:
{
  "consolidated": [
    "The user is a fan of sci-fi movies and recently watched 'Dune'.",
    "The user's cat is named 'Leo'."
  ],
  "to_delete": ["mem_1", "mem_2", "mem_3"]
}

Memories to process:
"""

# Pydantic Models for API requests and responses
class ConversationRequest(BaseModel):
    user_input: str = Field(..., example="Hi, I'm planning a trip to Paris.")
    ai_output: str = Field(..., example="Sounds great!")

class MemoriesList(BaseModel):
    memories: List[str]
    type: Optional[str] = Field('long-term', example='episodic')

class QueryRequest(BaseModel):
    query_text: str
    top_n: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

class MemoryRecord(BaseModel):
    id: str
    text: str
    created_at: str
    last_accessed: Optional[str] = None
    access_count: int = 0
    type: str

class ConsolidateRequest(BaseModel):
    memory_ids: Optional[List[str]] = None
    max_memories_to_process: int = 50

class RelevantMemory(BaseModel):
    id: str
    text: str
    similarity: float

class RelevantMemoriesResponse(BaseModel):
    relevant_memories: List[RelevantMemory]

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Any] = None

# ChromaDB setup
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION = 'memories'
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR, is_persistent=True))
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=sentence_transformer_ef
)

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup."""
    global embedding_model
    # The embedding model might be passed from the main app for consistency
    embedding_model = app.embedding_model if hasattr(app, "embedding_model") else None
    if not embedding_model:
        logging.info("Using default SentenceTransformer model for embeddings.")
        # In a microservice context, the model would be loaded here.
        # For integrated use, it's passed from main.py
    else:
        logging.info("Embedding model loaded from main application.")
    logging.info("ChromaDB client initialized and collection is ready.")

def get_embedding(text):
    """Generates an embedding for the given text using the globally set model."""
    if embedding_model and text:
        try:
            return embedding_model.encode(text).tolist()
        except Exception as e:
            logging.error(f"Error generating embedding for '{text}': {e}")
    return None

def parse_llm_json_response(response_text: str) -> Optional[Dict]:
    """Safely parses a JSON string from an LLM response."""
    try:
        # Find the JSON object within the response text
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        logging.warning("No JSON object found in LLM response.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM response: {response_text}")
        return None

@app.post("/extract_memories/", response_model=MemoriesList, tags=["Memories"])
async def extract_memories_api(request: ConversationRequest):
    """Extracts memories from a given conversation, expecting a JSON response."""
    if isinstance(request, dict):
        request = ConversationRequest(**request)

    try:
        conversation = f"User: {request.user_input}\nAI: {request.ai_output}"
        prompt = PROMPT_FOR_EXTRACTING_MEMORIES_CONVO + "\nConversation:\n" + conversation
        logging.info("Sending memory extraction request to LLM...")

        llm_response = await asyncio.to_thread(call_llm, prompt)
        if not llm_response:
            logging.warning("LLM did not return a response for memory extraction.")
            return MemoriesList(memories=[])

        parsed_json = parse_llm_json_response(llm_response)
        if parsed_json and "memories" in parsed_json and isinstance(parsed_json["memories"], list):
            logging.info(f"Extracted memories from LLM: {parsed_json['memories']}")
            return MemoriesList(memories=parsed_json["memories"])
        else:
            logging.error("Failed to parse memories from LLM response or key 'memories' is missing.")
            # Fallback to simple line splitting if JSON parsing fails
            memories_list = [m.strip().lstrip('-* ') for m in llm_response.split('\n') if m.strip().lstrip('-* ')]
            return MemoriesList(memories=memories_list)

    except Exception as e:
        logging.error(f"Error in extract_memories_api: {e}", exc_info=True)
        return MemoriesList(memories=[])

@app.post("/save_memories/", response_model=StatusResponse, tags=["Memories"])
def save_memories_api(memories_request: MemoriesList):
    """Saves a list of memories to the database."""
    return save_memories(memories_request.memories, memories_request.type)

def save_memories(memories_to_save: List[str], memory_type: str = 'long-term'):
    """Saves memories to ChromaDB."""
    if not memories_to_save:
        return StatusResponse(status="ok", message="No memories to save.")

    ids = []
    metadatas = []
    
    for text in memories_to_save:
        if not text.strip():
            continue
        # Use a UUID for the ID to ensure uniqueness.
        mem_id = str(uuid.uuid4())
        ids.append(mem_id)
        metadatas.append({
            "text": text,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,
            "type": memory_type
        })
    
    if not ids:
        return StatusResponse(status="ok", message="No valid memories to save.")

    try:
        # Note: ChromaDB's embedding function will handle embedding generation automatically.
        chroma_collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=[meta["text"] for meta in metadatas] # Pass documents for embedding
        )
        logging.info(f"Successfully saved {len(ids)} memories to ChromaDB.")
        return StatusResponse(status="ok", message=f"Saved {len(ids)} new memories.")
    except Exception as e:
        logging.error(f"Failed to save memories to ChromaDB: {e}", exc_info=True)
        return StatusResponse(status="error", message=f"Failed to save memories: {e}")

@app.post("/get_relevant_memories/", response_model=RelevantMemoriesResponse, tags=["Memories"])
async def get_relevant_memories_api(request: QueryRequest):
    """Queries ChromaDB for memories relevant to the query text."""
    if isinstance(request, dict):
        request = QueryRequest(**request)
        
    try:
        if not request.query_text:
            return RelevantMemoriesResponse(relevant_memories=[])

        results = chroma_collection.query(
            query_texts=[request.query_text],
            n_results=request.top_n
        )

        relevant_memories = []
        ids_to_update = []
        metadatas_to_update = []

        if results and results['ids'][0]:
            for i, mem_id in enumerate(results['ids'][0]):
                dist = results['distances'][0][i]
                similarity = 1 - dist # Convert distance to similarity
                
                if similarity >= request.similarity_threshold:
                    metadata = results['metadatas'][0][i]
                    relevant_memories.append(
                        RelevantMemory(id=mem_id, text=metadata.get("text", ""), similarity=similarity)
                    )
                    
                    # Prepare metadata update
                    ids_to_update.append(mem_id)
                    metadata['last_accessed'] = datetime.utcnow().isoformat()
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    metadatas_to_update.append(metadata)

        # Update access metadata for retrieved memories
        if ids_to_update:
            chroma_collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
            logging.info(f"Updated access metadata for {len(ids_to_update)} memories.")
            
        return RelevantMemoriesResponse(relevant_memories=relevant_memories)

    except Exception as e:
        logging.error(f"Error querying ChromaDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consolidate_memories/", response_model=StatusResponse, tags=["Memories"])
async def consolidate_memories_api(request: ConsolidateRequest):
    """Consolidates memories using an LLM to merge, deduplicate, and generalize."""
    try:
        if request.memory_ids:
            memories_data = chroma_collection.get(ids=request.memory_ids, include=["metadatas"])
        else:
            # Fetch a batch of the least recently accessed memories for consolidation
            memories_data = chroma_collection.get(
                limit=request.max_memories_to_process,
                # TODO: ChromaDB does not support sorting by metadata field yet.
                # This will fetch a random batch instead.
                include=["metadatas"]
            )
        
        if not memories_data or not memories_data['ids']:
            return StatusResponse(status="ok", message="No memories found to consolidate.")

        memories_to_process = [
            {"id": mem_id, "text": memories_data['metadatas'][i].get("text", "")}
            for i, mem_id in enumerate(memories_data['ids'])
        ]

        prompt = PROMPT_FOR_CONSOLIDATION + "\n" + json.dumps(memories_to_process, indent=2)
        llm_response_str = await asyncio.to_thread(call_llm, prompt)

        if not llm_response_str:
            return StatusResponse(status="error", message="LLM failed to provide a consolidation plan.")

        consolidation_plan = parse_llm_json_response(llm_response_str)
        if not consolidation_plan or "consolidated" not in consolidation_plan or "to_delete" not in consolidation_plan:
            return StatusResponse(status="error", message="Could not parse consolidation plan from LLM response.")

        # Save new consolidated memories
        if consolidation_plan["consolidated"]:
            save_memories(consolidation_plan["consolidated"], memory_type='long-term-consolidated')

        # Delete old memories
        if consolidation_plan["to_delete"]:
            to_delete_ids = consolidation_plan["to_delete"]
            if to_delete_ids:
                # Ensure uniqueness of IDs before deleting
                unique_to_delete_ids = list(set(to_delete_ids))
                if unique_to_delete_ids:
                    try:
                        chroma_collection.delete(ids=unique_to_delete_ids)
                        logging.info(f"Deleted {len(unique_to_delete_ids)} old memories.")
                    except Exception as e:
                        logging.error(f"Error deleting memories from ChromaDB: {e}", exc_info=True)
                        # Optionally, decide if you want to raise an exception or just log the error
                        # For now, we'll just log it and continue

        return StatusResponse(
            status="ok",
            message=f"Consolidation process completed. Added {len(consolidation_plan['consolidated'])} and attempted to delete {len(to_delete_ids)} memories.",
            details=consolidation_plan
        )

    except Exception as e:
        logging.error(f"Error during consolidation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during consolidation: {e}")

@app.get("/list_memories/", response_model=List[MemoryRecord], tags=["Memories"])
async def list_memories_api(limit: int = 100):
    """Lists all memories currently stored in ChromaDB."""
    try:
        results = chroma_collection.get(limit=limit, include=["metadatas"])
        
        memory_records = []
        if results and results['ids']:
            for i, mem_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                memory_records.append(MemoryRecord(
                    id=mem_id,
                    text=metadata.get("text", ""),
                    created_at=metadata.get("created_at", ""),
                    last_accessed=metadata.get("last_accessed"),
                    access_count=metadata.get("access_count", 0),
                    type=metadata.get("type", "unknown")
                ))
        return memory_records
    except Exception as e:
        logging.error(f"Error listing memories: {e}", exc_info=True)
        return []

@app.get("/health", response_model=StatusResponse, tags=["System"])
async def health_check():
    """Performs a health check of the service."""
    try:
        # Check ChromaDB connection
        count = chroma_collection.count()
        return StatusResponse(status="ok", message="Service is healthy.", details={"memory_count": count})
    except Exception as e:
        logging.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

# This allows running the memory service independently for debugging or as a microservice
if __name__ == "__main__":
    # For standalone execution, we need to load a model.
    # In the integrated AGI, the model is passed via app state.
    app.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    uvicorn.run(app, host="0.0.0.0", port=8001)

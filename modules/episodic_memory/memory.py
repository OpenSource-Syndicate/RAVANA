from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn # For running the server
import re
from datetime import datetime, timedelta
import os
import uuid
import asyncio
from core.llm import call_llm
from core.config import Config
import json # For storing embeddings as JSON strings
import numpy as np
from sentence_transformers import SentenceTransformer # For generating embeddings
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tempfile
from pathlib import Path

# Import new multi-modal components
try:
    from .models import (
        MemoryRecord as NewMemoryRecord, ContentType, MemoryType, 
        SearchRequest, SearchResponse, ConversationRequest as NewConversationRequest,
        MemoriesList as NewMemoriesList, ProcessingResult, BatchProcessRequest,
        BatchProcessResult, CrossModalSearchRequest, HealthCheckResponse
    )
    from .multi_modal_service import MultiModalMemoryService
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Multi-modal components not available: {e}")
    MULTIMODAL_AVAILABLE = False

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
```
User is planning a trip to Paris.
User's favorite color is blue.
```

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
```
The user is a fan of sci-fi movies and recently watched 'Dune'.",
The user's cat is named 'Leo'."
```

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
    
    def dict(self):
        """Pydantic v1 style serialization (backward compatibility)"""
        return {
            "id": self.id,
            "text": self.text,
            "similarity": self.similarity
        }
    
    def model_dump(self):
        """Pydantic v2 style serialization"""
        return {
            "id": self.id,
            "text": self.text,
            "similarity": self.similarity
        }
    
    def to_dict(self):
        """Custom serialization method"""
        return {
            "id": self.id,
            "text": self.text,
            "similarity": self.similarity
        }
    
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

# Multi-modal service initialization
multimodal_service: Optional[MultiModalMemoryService] = None

def get_database_url() -> str:
    """Get PostgreSQL database URL from environment or use default."""
    return os.getenv(
        "POSTGRES_URL", 
        "postgresql://postgres:password@localhost:5432/ravana_memory"
    )

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup."""
    global embedding_model, multimodal_service
    
    # The embedding model might be passed from the main app for consistency
    embedding_model = app.embedding_model if hasattr(app, "embedding_model") else None
    if not embedding_model:
        logging.info("Using default SentenceTransformer model for embeddings.")
        # In a microservice context, the model would be loaded here.
        # For integrated use, it's passed from main.py
    else:
        logging.info("Embedding model loaded from main application.")
    
    # Initialize ChromaDB collection with error handling
    try:
        logging.info("Initializing ChromaDB client and collection...")
        global chroma_client, chroma_collection, sentence_transformer_ef
        
        # Initialize embedding function
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize ChromaDB client with better error handling
        chroma_client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIR, 
            is_persistent=True
        ))
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=sentence_transformer_ef
        )
        
        logging.info("ChromaDB client initialized and collection is ready.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB: {e}")
        # Fallback to in-memory storage
        try:
            chroma_client = chromadb.Client()
            chroma_collection = chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                embedding_function=sentence_transformer_ef
            )
            logging.warning("Using in-memory ChromaDB storage due to persistence error")
        except Exception as fallback_error:
            logging.error(f"Fallback ChromaDB initialization also failed: {fallback_error}")
            raise Exception(f"Failed to initialize memory storage: {e}")
    
    # Initialize multi-modal service if available
    if MULTIMODAL_AVAILABLE:
        try:
            database_url = get_database_url()
            multimodal_service = MultiModalMemoryService(
                database_url=database_url,
                text_model_name=Config.EMBEDDING_MODEL
            )
            await multimodal_service.initialize()
            logging.info("Multi-modal memory service initialized successfully")
        except Exception as e:
            logging.warning(f"Multi-modal service initialization failed: {e}")
            multimodal_service = None
    else:
        logging.info("Multi-modal components not available, using legacy mode")

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
        # Handle empty or None responses
        if not response_text or not response_text.strip():
            logging.warning("Empty response received from LLM")
            return None
            
        response_text = response_text.strip()
        
        # Strategy 1: Try to parse the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Look for JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON from code block: {e}")
                
        # Strategy 3: Look for any JSON-like structure
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                # Fix common JSON issues
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse extracted JSON structure: {e}")
                
        # Strategy 4: Handle common LLM response patterns
        # Remove common prefixes/suffixes
        cleaned_response = re.sub(r'^[^{]*', '', response_text)  # Remove everything before first {
        cleaned_response = re.sub(r'[^}]*$', '', cleaned_response)  # Remove everything after last }
        
        if cleaned_response:
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse cleaned response: {e}")
                
        logging.warning("No valid JSON object found in LLM response.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing LLM response: {e}")
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
            # Improved fallback to simple line splitting if JSON parsing fails
            # First try to extract any list-like structure
            memories_list = []
            
            # Look for markdown lists
            markdown_items = re.findall(r'^\s*[*\-+]\s+(.+)$', llm_response, re.MULTILINE)
            if markdown_items:
                memories_list.extend([item.strip() for item in markdown_items if item.strip()])
            
            # Look for numbered lists
            if not memories_list:
                numbered_items = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', llm_response, re.MULTILINE)
                if numbered_items:
                    memories_list.extend([item.strip() for item in numbered_items if item.strip()])
            
            # Fallback to line splitting if no lists found
            if not memories_list:
                memories_list = [m.strip().lstrip('-* ') for m in llm_response.split('\n') if m.strip().lstrip('-* ')]
            
            # Filter out empty items and common artifacts
            memories_list = [m for m in memories_list if m and not m.startswith('#') and not m.startswith('```')]
            
            logging.info(f"Extracted {len(memories_list)} memories using fallback method")
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

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if ChromaDB is responsive
        collection_count = len(chroma_client.list_collections())
        
        # Check if embedding model is working
        test_embedding = sentence_transformer_ef(["test"])
        
        return HealthCheckResponse(
            status="healthy",
            details={
                "collections": collection_count,
                "embedding_model": "ok",
                "multimodal_service": "available" if multimodal_service else "unavailable"
            }
        )
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            details={
                "error": str(e)
            }
        )

# ===== NEW MULTI-MODAL ENDPOINTS =====

@app.post("/memories/audio/", response_model=ProcessingResult, tags=["Multi-Modal"])
async def process_audio_memory(
    audio_file: UploadFile = File(...),
    context: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Process audio file with Whisper transcription and store as memory."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.gettempdir()) / "ravana_audio"
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{uuid.uuid4()}_{audio_file.filename}"
        
        with open(file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Process audio
        memory_record = await multimodal_service.process_audio_memory(
            audio_path=str(file_path),
            context=context,
            tags=tag_list
        )
        
        # Clean up temp file
        file_path.unlink(missing_ok=True)
        
        return ProcessingResult(
            memory_record=memory_record,
            processing_time_ms=0,  # Would be calculated
            success=True
        )
        
    except Exception as e:
        logging.error(f"Audio processing failed: {e}")
        # Clean up temp file on error
        if 'file_path' in locals():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/image/", response_model=ProcessingResult, tags=["Multi-Modal"])
async def process_image_memory(
    image_file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Process image file and store as memory."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.gettempdir()) / "ravana_images"
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{uuid.uuid4()}_{image_file.filename}"
        
        with open(file_path, "wb") as f:
            content = await image_file.read()
            f.write(content)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Process image
        memory_record = await multimodal_service.process_image_memory(
            image_path=str(file_path),
            description=description,
            tags=tag_list
        )
        
        # Clean up temp file
        file_path.unlink(missing_ok=True)
        
        return ProcessingResult(
            memory_record=memory_record,
            processing_time_ms=0,  # Would be calculated
            success=True
        )
        
    except Exception as e:
        logging.error(f"Image processing failed: {e}")
        # Clean up temp file on error
        if 'file_path' in locals():
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/advanced/", response_model=SearchResponse, tags=["Search"])
async def advanced_search(request: SearchRequest):
    """Perform advanced search with multiple modes."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        return await multimodal_service.search_memories(request)
    except Exception as e:
        logging.error(f"Advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/cross-modal/", response_model=List[NewMemoryRecord], tags=["Search"])
async def cross_modal_search(request: CrossModalSearchRequest):
    """Perform cross-modal search across different content types."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        search_results = await multimodal_service.search_engine.cross_modal_search(request)
        return [result.memory_record for result in search_results]
    except Exception as e:
        logging.error(f"Cross-modal search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{memory_id}/similar", response_model=List[NewMemoryRecord], tags=["Search"])
async def find_similar_memories(
    memory_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.7
):
    """Find memories similar to a given memory."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        memory_uuid = uuid.UUID(memory_id)
        similar_memories = await multimodal_service.find_similar_memories(
            memory_uuid, limit, similarity_threshold
        )
        return similar_memories
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory ID format")
    except Exception as e:
        logging.error(f"Similar memories search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/process/", response_model=BatchProcessResult, tags=["Batch"])
async def batch_process_files(request: BatchProcessRequest):
    """Process multiple files in batch."""
    if not multimodal_service:
        raise HTTPException(status_code=501, detail="Multi-modal service not available")
    
    try:
        return await multimodal_service.batch_process_files(request)
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics/", response_model=Dict[str, Any], tags=["Statistics"])
async def get_memory_statistics():
    """Get comprehensive memory statistics."""
    try:
        # Get ChromaDB stats
        chroma_count = chroma_collection.count()
        
        stats = {
            "chroma_memory_count": chroma_count,
            "service_mode": "legacy" if not multimodal_service else "multimodal"
        }
        
        # Get multi-modal stats if available
        if multimodal_service:
            multimodal_stats = await multimodal_service.get_memory_statistics()
            stats["multimodal_stats"] = multimodal_stats
        
        return stats
        
    except Exception as e:
        logging.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown."""
    global multimodal_service, chroma_client, chroma_collection
    
    logger.info("📋 FastAPI Memory Service shutdown initiated...")
    
    try:
        # Close multi-modal service if available
        if multimodal_service:
            logger.info("Closing multi-modal memory service...")
            await asyncio.wait_for(
                multimodal_service.close(),
                timeout=Config.MEMORY_SERVICE_SHUTDOWN_TIMEOUT
            )
            logger.info("Multi-modal service closed successfully")
            multimodal_service = None
    except asyncio.TimeoutError:
        logger.warning("Multi-modal service shutdown exceeded timeout")
    except Exception as e:
        logger.error(f"Error closing multi-modal service: {e}")
    
    try:
        # Persist ChromaDB if enabled
        if Config.CHROMADB_PERSIST_ON_SHUTDOWN and chroma_collection:
            logger.info("Persisting ChromaDB collections...")
            # ChromaDB automatically persists with persistent client
            # But we can ensure any pending operations are completed
            memory_count = chroma_collection.count()
            logger.info(f"ChromaDB persistence confirmed - {memory_count} memories stored")
    except Exception as e:
        logger.error(f"Error persisting ChromaDB: {e}")
    
    try:
        # Clean up temporary files if enabled
        if Config.TEMP_FILE_CLEANUP_ENABLED:
            logger.info("Cleaning up temporary files...")
            await _cleanup_temp_files()
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
    
    logger.info("✅ FastAPI Memory Service shutdown completed")


async def _cleanup_temp_files():
    """Clean up temporary files created by the memory service."""
    try:
        temp_dirs = [
            Path(tempfile.gettempdir()) / "ravana_audio",
            Path(tempfile.gettempdir()) / "ravana_images"
        ]
        
        total_cleaned = 0
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                file_count = 0
                for file_path in temp_dir.iterdir():
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            file_count += 1
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {file_path}: {e}")
                
                total_cleaned += file_count
                
                # Try to remove empty directory
                try:
                    if not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                        logger.info(f"Removed empty temp directory: {temp_dir}")
                except OSError:
                    pass  # Directory not empty
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} temporary files")
        
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

# This allows running the memory service independently for debugging or as a microservice
if __name__ == "__main__":
    # For standalone execution, we need to load a model.
    # In the integrated AGI, the model is passed via app state.
    app.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Set up environment for multi-modal service
    if not os.getenv("POSTGRES_URL"):
        logging.warning("POSTGRES_URL not set, multi-modal features may not work")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

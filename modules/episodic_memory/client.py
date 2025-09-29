import requests
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import mimetypes

BASE_URL = "http://localhost:8001"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def make_request(method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Helper function to make HTTP requests with retries and better error handling"""
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=headers, timeout=timeout)
            else:
                response = requests.post(
                    url, headers=headers, json=json_data, timeout=timeout)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(
                    f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"Failed to connect to server after {MAX_RETRIES} attempts: {e}")
                return None
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                print(
                    f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Request timed out after {MAX_RETRIES} attempts: {e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"Response content: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"Response content: {e.response.text}")
            return None
    return None


def extract_memories(user_input: str, ai_output: str) -> Optional[Dict[str, Any]]:
    """Calls the /extract_memories/ endpoint with separate user and AI messages."""
    payload = {"user_input": user_input, "ai_output": ai_output}
    return make_request("post", "extract_memories/", json_data=payload)


def save_memories(memories_list: List[str], memory_type: str = 'long-term') -> Optional[Dict[str, Any]]:
    """Saves a list of memories to the server with a specified type."""
    data = {"memories": memories_list, "type": memory_type}
    return make_request("POST", "/save_memories/", json_data=data)


def get_relevant_memories(query_text: str, top_n: int = 5, similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Calls the /get_relevant_memories/ endpoint."""
    payload = {
        "query_text": query_text,
        "top_n": top_n,
        "similarity_threshold": similarity_threshold
    }
    return make_request("post", "get_relevant_memories/", json_data=payload)


def health_check() -> Optional[Dict[str, Any]]:
    """Calls the /health endpoint."""
    return make_request("get", "health", timeout=10)

# ===== NEW MULTI-MODAL CLIENT FUNCTIONS =====


def upload_audio_memory(audio_file_path: str,
                        context: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Upload and process an audio file as memory."""
    if not Path(audio_file_path).exists():
        print(f"Audio file not found: {audio_file_path}")
        return None

    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio_file': (
                Path(audio_file_path).name, audio_file, 'audio/wav')}
            data = {}

            if context:
                data['context'] = context
            if tags:
                data['tags'] = ','.join(tags)

            url = f"{BASE_URL}/memories/audio/"
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        print(f"Audio upload failed: {e}")
        return None


def upload_image_memory(image_file_path: str,
                        description: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Upload and process an image file as memory."""
    if not Path(image_file_path).exists():
        print(f"Image file not found: {image_file_path}")
        return None

    try:
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(image_file_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default

        with open(image_file_path, 'rb') as image_file:
            files = {'image_file': (
                Path(image_file_path).name, image_file, mime_type)}
            data = {}

            if description:
                data['description'] = description
            if tags:
                data['tags'] = ','.join(tags)

            url = f"{BASE_URL}/memories/image/"
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        print(f"Image upload failed: {e}")
        return None


def advanced_search(query: str,
                    search_mode: str = "hybrid",
                    content_types: Optional[List[str]] = None,
                    memory_types: Optional[List[str]] = None,
                    limit: int = 10,
                    similarity_threshold: float = 0.7,
                    tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Perform advanced search with multiple modes and filters."""
    payload = {
        "query": query,
        "search_mode": search_mode,
        "limit": limit,
        "similarity_threshold": similarity_threshold
    }

    if content_types:
        payload["content_types"] = content_types
    if memory_types:
        payload["memory_types"] = memory_types
    if tags:
        payload["tags"] = tags

    return make_request("post", "search/advanced/", json_data=payload)


def cross_modal_search(query_content: str,
                       query_type: str,
                       target_types: List[str],
                       limit: int = 10,
                       similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Perform cross-modal search across different content types."""
    payload = {
        "query_content": query_content,
        "query_type": query_type,
        "target_types": target_types,
        "limit": limit,
        "similarity_threshold": similarity_threshold
    }

    return make_request("post", "search/cross-modal/", json_data=payload)


def find_similar_memories(memory_id: str,
                          limit: int = 10,
                          similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Find memories similar to a given memory."""
    url = f"memories/{memory_id}/similar?limit={limit}&similarity_threshold={similarity_threshold}"
    return make_request("get", url)


def batch_process_files(file_paths: List[str],
                        content_types: Optional[List[str]] = None,
                        parallel_processing: bool = True,
                        max_workers: int = 4) -> Optional[Dict[str, Any]]:
    """Process multiple files in batch."""
    payload = {
        "file_paths": file_paths,
        "parallel_processing": parallel_processing,
        "max_workers": max_workers
    }

    if content_types:
        payload["content_types"] = content_types

    return make_request("post", "batch/process/", json_data=payload)


def get_memory_statistics() -> Optional[Dict[str, Any]]:
    """Get comprehensive memory system statistics."""
    return make_request("get", "statistics/")


def process_text_memory(text: str,
                        memory_type: str = "episodic",
                        tags: Optional[List[str]] = None,
                        emotional_valence: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Process and store text memory (legacy compatibility)."""
    # For backward compatibility, use the existing save_memories function
    return save_memories([text], memory_type)

# Enhanced search functions


def vector_search(query: str,
                  content_types: Optional[List[str]] = None,
                  limit: int = 10,
                  similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Perform pure vector similarity search."""
    return advanced_search(
        query=query,
        search_mode="vector",
        content_types=content_types,
        limit=limit,
        similarity_threshold=similarity_threshold
    )


def text_search(query: str,
                content_types: Optional[List[str]] = None,
                limit: int = 10) -> Optional[Dict[str, Any]]:
    """Perform pure text search."""
    return advanced_search(
        query=query,
        search_mode="text",
        content_types=content_types,
        limit=limit
    )


def hybrid_search(query: str,
                  content_types: Optional[List[str]] = None,
                  limit: int = 10,
                  similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Perform hybrid search combining vector and text search."""
    return advanced_search(
        query=query,
        search_mode="hybrid",
        content_types=content_types,
        limit=limit,
        similarity_threshold=similarity_threshold
    )

# Content type helpers


class ContentType:
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class MemoryType:
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CONSOLIDATED = "consolidated"
    WORKING = "working"


class SearchMode:
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"
    CROSS_MODAL = "cross_modal"


if __name__ == "__main__":
    print("--- Enhanced MemoryDB Client Test Script ---")

    # 0. Health Check
    print("\n0. Checking API health...")
    health = health_check()
    if health and health.get("status") == "ok":
        print(f"API Health: {health['status']} - {health.get('message', '')}")
        if 'details' in health:
            details = health['details']
            if 'multimodal_service' in details:
                print(
                    f"Multi-modal service: {details['multimodal_service'].get('status', 'unknown')}")
    else:
        print(f"Health check failed or API not healthy: {health}")
        print("Please ensure the FastAPI server (memory.py) is running.")
        exit()

    # 1. Test legacy functionality
    print("\n1. Testing legacy memory extraction and storage...")
    user1 = "I'm planning a vacation to Hawaii next month. And then Japan"
    ai1 = "That sounds wonderful!"
    extracted_data1 = extract_memories(user1, ai1)
    memories1 = []
    if extracted_data1 and 'memories' in extracted_data1:
        memories1 = extracted_data1['memories']
        print(f"Extracted memories: {memories1}")

    if memories1:
        save_response = save_memories(memories1)
        print(f"Save response: {save_response}")

    # 2. Test advanced search
    print("\n2. Testing advanced search capabilities...")

    # Hybrid search
    print("\n2a. Hybrid search for vacation plans:")
    hybrid_result = hybrid_search(
        query="vacation plans",
        content_types=[ContentType.TEXT],
        limit=5
    )
    if hybrid_result and 'results' in hybrid_result:
        print(f"Found {len(hybrid_result['results'])} results")
        for result in hybrid_result['results'][:3]:
            memory = result['memory_record']
            score = result['similarity_score']
            print(f"  - Score: {score:.3f} | {memory['content_text'][:50]}...")

    # Vector search
    print("\n2b. Vector search for travel:")
    vector_result = vector_search(
        query="travel destinations",
        limit=3,
        similarity_threshold=0.5
    )
    if vector_result and 'results' in vector_result:
        print(f"Found {len(vector_result['results'])} vector results")

    # 3. Test file upload (if files exist)
    print("\n3. Testing multi-modal file processing...")

    # Test audio upload (create a dummy file for demo)
    print("\n3a. Audio memory processing:")
    try:
        import tempfile
        import os

        # Create a dummy audio file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(b"dummy audio data for testing")
            temp_audio_path = temp_audio.name

        # Note: This will fail unless you have actual audio processing setup
        audio_result = upload_audio_memory(
            temp_audio_path,
            context="Test audio for demonstration",
            tags=["test", "demo"]
        )

        if audio_result:
            print(
                f"Audio processing result: {audio_result.get('success', False)}")
        else:
            print("Audio processing not available (expected in demo mode)")

        # Clean up
        os.unlink(temp_audio_path)

    except Exception as e:
        print(f"Audio test skipped: {e}")

    # Test image upload
    print("\n3b. Image memory processing:")
    try:
        from PIL import Image
        import numpy as np

        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
            # Create simple test image
            img_array = np.random.randint(
                0, 256, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(temp_img.name, 'JPEG')
            temp_img_path = temp_img.name

        image_result = upload_image_memory(
            temp_img_path,
            description="Test image for demonstration",
            tags=["test", "generated"]
        )

        if image_result:
            print(
                f"Image processing result: {image_result.get('success', False)}")
        else:
            print("Image processing not available (expected in demo mode)")

        # Clean up
        os.unlink(temp_img_path)

    except ImportError:
        print("PIL not available, skipping image test")
    except Exception as e:
        print(f"Image test skipped: {e}")

    # 4. Test statistics
    print("\n4. Getting memory statistics...")
    stats = get_memory_statistics()
    if stats:
        print(f"Service mode: {stats.get('service_mode', 'unknown')}")
        print(f"ChromaDB memories: {stats.get('chroma_memory_count', 0)}")
        if 'multimodal_stats' in stats:
            mm_stats = stats['multimodal_stats']
            print(f"Total memories: {mm_stats.get('total_memories', 0)}")

    # 5. Test cross-modal search (conceptual)
    print("\n5. Testing cross-modal search concepts...")
    cross_modal_result = cross_modal_search(
        query_content="vacation planning",
        query_type=ContentType.TEXT,
        target_types=[ContentType.TEXT, ContentType.IMAGE],
        limit=5
    )

    if cross_modal_result:
        print(f"Cross-modal search returned {len(cross_modal_result)} results")
    else:
        print("Cross-modal search not available (expected in demo mode)")

    print("\n--- Enhanced MemoryDB Client Test Completed ---")
    print("\nNote: Some multi-modal features require:")
    print("- PostgreSQL with pgvector extension")
    print("- Audio/image processing dependencies")
    print("- Proper configuration of the multi-modal service")

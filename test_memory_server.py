#!/usr/bin/env python3
"""
Test Memory Server
This script tests the episodic memory server functionality in isolation.
"""

import os
import sys
import time
import logging
import argparse
import requests
import threading
import json
import traceback
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_server_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MemoryServerTest")

# Add modules directory to path
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.append(MODULES_DIR)

def start_memory_server():
    """Start the memory server in a separate process."""
    logger.info("Starting memory server...")
    
    try:
        # Add episodic_memory directory to path
        episodic_memory_dir = os.path.join(MODULES_DIR, "episodic_memory")
        sys.path.append(episodic_memory_dir)
        
        # Import memory app
        from memory import app
        import uvicorn
        
        # Start server in a separate thread
        def run_server():
            logger.info("Memory server thread starting...")
            try:
                uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
            except Exception as e:
                logger.error(f"Error in memory server: {e}")
                logger.debug(traceback.format_exc())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        logger.info("Waiting for memory server to start...")
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://127.0.0.1:8000/health")
                if response.status_code == 200:
                    logger.info("Memory server started successfully")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        
        logger.error("Memory server failed to start within timeout")
        return False
    except Exception as e:
        logger.error(f"Error starting memory server: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_extract_memories():
    """Test the extract_memories endpoint."""
    logger.info("Testing extract_memories endpoint...")
    
    try:
        test_input = "Today I learned about artificial intelligence and how it can be used to solve complex problems."
        
        response = requests.post(
            "http://127.0.0.1:8000/extract_memories/",
            json={"user_input": test_input, "ai_output": ""}
        )
        
        if response.status_code == 200:
            memories = response.json().get("memories", [])
            logger.info(f"Successfully extracted {len(memories)} memories")
            logger.debug(f"Extracted memories: {json.dumps(memories, indent=2)}")
            return True
        else:
            logger.error(f"Failed to extract memories: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing extract_memories: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_save_memories():
    """Test the save_memories endpoint."""
    logger.info("Testing save_memories endpoint...")
    
    try:
        test_memories = [
            {
                "text": "I learned about artificial intelligence.",
                "importance": 7,
                "metadata": {
                    "source": "test",
                    "timestamp": time.time()
                }
            }
        ]
        
        response = requests.post(
            "http://127.0.0.1:8000/save_memories/",
            json={"memories": test_memories}
        )
        
        if response.status_code == 200:
            logger.info("Successfully saved memories")
            return True
        else:
            logger.error(f"Failed to save memories: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing save_memories: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_get_relevant_memories():
    """Test the get_relevant_memories endpoint."""
    logger.info("Testing get_relevant_memories endpoint...")
    
    try:
        # First save a memory to ensure there's something to retrieve
        test_save_memories()
        
        query = "artificial intelligence"
        
        response = requests.post(
            "http://127.0.0.1:8000/get_relevant_memories/",
            json={"query_text": query, "top_n": 5}
        )
        
        if response.status_code == 200:
            memories = response.json().get("relevant_memories", [])
            logger.info(f"Successfully retrieved {len(memories)} relevant memories")
            logger.debug(f"Retrieved memories: {json.dumps(memories, indent=2)}")
            return True
        else:
            logger.error(f"Failed to get relevant memories: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing get_relevant_memories: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_memory_flow():
    """Test the complete memory flow: extract, save, retrieve."""
    logger.info("Testing complete memory flow...")
    
    try:
        # 1. Extract memories
        test_input = "The future of AI involves developing systems that can learn and adapt to new situations autonomously."
        
        extract_response = requests.post(
            "http://127.0.0.1:8000/extract_memories/",
            json={"user_input": test_input, "ai_output": ""}
        )
        
        if extract_response.status_code != 200:
            logger.error(f"Failed to extract memories: {extract_response.status_code} {extract_response.text}")
            return False
        
        memories = extract_response.json().get("memories", [])
        logger.info(f"Extracted {len(memories)} memories")
        
        # 2. Save memories
        if memories:
            save_response = requests.post(
                "http://127.0.0.1:8000/save_memories/",
                json={"memories": memories}
            )
            
            if save_response.status_code != 200:
                logger.error(f"Failed to save memories: {save_response.status_code} {save_response.text}")
                return False
            
            logger.info("Saved memories successfully")
        else:
            logger.warning("No memories to save")
        
        # 3. Retrieve memories
        query = "AI systems learning"
        
        retrieve_response = requests.post(
            "http://127.0.0.1:8000/get_relevant_memories/",
            json={"query_text": query, "top_n": 5}
        )
        
        if retrieve_response.status_code != 200:
            logger.error(f"Failed to retrieve memories: {retrieve_response.status_code} {retrieve_response.text}")
            return False
        
        relevant_memories = retrieve_response.json().get("relevant_memories", [])
        logger.info(f"Retrieved {len(relevant_memories)} relevant memories")
        logger.debug(f"Retrieved memories: {json.dumps(relevant_memories, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing memory flow: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_memory_performance():
    """Test the performance of the memory server with multiple requests."""
    logger.info("Testing memory server performance...")
    
    try:
        # Generate some test inputs
        test_inputs = [
            "Artificial intelligence is transforming industries across the globe.",
            "Machine learning models can identify patterns in large datasets.",
            "Neural networks are inspired by the human brain's structure.",
            "Deep learning has revolutionized computer vision and natural language processing.",
            "Reinforcement learning allows agents to learn through trial and error.",
            "Supervised learning uses labeled data to train models.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Transfer learning applies knowledge from one domain to another.",
            "Generative AI can create new content like images and text.",
            "Ethical AI focuses on fairness, transparency, and accountability."
        ]
        
        start_time = time.time()
        success_count = 0
        
        for i, test_input in enumerate(test_inputs):
            logger.info(f"Processing test input {i+1}/{len(test_inputs)}")
            
            # Extract memories
            extract_response = requests.post(
                "http://127.0.0.1:8000/extract_memories/",
                json={"user_input": test_input, "ai_output": ""}
            )
            
            if extract_response.status_code == 200:
                memories = extract_response.json().get("memories", [])
                
                # Save memories
                if memories:
                    save_response = requests.post(
                        "http://127.0.0.1:8000/save_memories/",
                        json={"memories": memories}
                    )
                    
                    if save_response.status_code == 200:
                        success_count += 1
            
            # Small delay between requests
            time.sleep(0.5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Performance test completed in {duration:.2f} seconds")
        logger.info(f"Successfully processed {success_count}/{len(test_inputs)} inputs")
        
        return success_count == len(test_inputs)
    except Exception as e:
        logger.error(f"Error testing memory performance: {e}")
        logger.debug(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Memory Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test", choices=["all", "extract", "save", "retrieve", "flow", "performance"], 
                       default="all", help="Test to run")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Set handler levels too
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Start the memory server
    if not start_memory_server():
        logger.error("Failed to start memory server, exiting")
        return 1
    
    # Wait for server to fully initialize
    logger.info("Waiting for server to fully initialize...")
    time.sleep(5)
    
    # Run the specified test
    success = True
    if args.test == "extract" or args.test == "all":
        extract_success = test_extract_memories()
        logger.info(f"Extract memories test: {'PASSED' if extract_success else 'FAILED'}")
        success = success and extract_success
    
    if args.test == "save" or args.test == "all":
        save_success = test_save_memories()
        logger.info(f"Save memories test: {'PASSED' if save_success else 'FAILED'}")
        success = success and save_success
    
    if args.test == "retrieve" or args.test == "all":
        retrieve_success = test_get_relevant_memories()
        logger.info(f"Retrieve memories test: {'PASSED' if retrieve_success else 'FAILED'}")
        success = success and retrieve_success
    
    if args.test == "flow" or args.test == "all":
        flow_success = test_memory_flow()
        logger.info(f"Memory flow test: {'PASSED' if flow_success else 'FAILED'}")
        success = success and flow_success
    
    if args.test == "performance" or args.test == "all":
        perf_success = test_memory_performance()
        logger.info(f"Performance test: {'PASSED' if perf_success else 'FAILED'}")
        success = success and perf_success
    
    if success:
        logger.info("All tests PASSED")
        print("\n✅ Memory server is working correctly!")
        return 0
    else:
        logger.error("Some tests FAILED")
        print("\n❌ Memory server has issues. Check the log for details.")
        return 1

if __name__ == "__main__":
    # Handle keyboard interrupt gracefully
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        print("\nTest interrupted by user")
        sys.exit(1) 
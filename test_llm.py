#!/usr/bin/env python3
"""
Test LLM Module
This script tests the LLM functionality to diagnose any issues with API connections.
"""

import os
import sys
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLMTest")

# Add modules directory to path
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.append(MODULES_DIR)

# Add agent_self_reflection directory to path
agent_self_reflection_dir = os.path.join(MODULES_DIR, "agent_self_reflection")
sys.path.append(agent_self_reflection_dir)

def test_config():
    """Test if the config.json file exists and is valid."""
    config_path = os.path.join(agent_self_reflection_dir, "config.json")
    logger.info(f"Testing config file: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Config file loaded successfully. Providers: {list(config.keys())}")
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return False

def test_imports():
    """Test if all required imports are available."""
    logger.info("Testing imports...")
    
    try:
        # Import required modules
        from agent_self_reflection.llm import call_llm
        import requests
        from openai import OpenAI
        
        try:
            from google import genai
            logger.info("Google Generative AI module imported successfully")
        except ImportError:
            logger.warning("Google Generative AI module not available. Some functionality may be limited.")
        
        logger.info("All required imports are available")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing imports: {e}")
        return False

def test_llm_call():
    """Test if the LLM call works."""
    logger.info("Testing LLM call...")
    
    try:
        # Import the LLM module
        from agent_self_reflection.llm import call_llm
        
        # Test prompt
        prompt = "What is the capital of France? Keep your answer short."
        
        # Try to call the LLM
        logger.info(f"Calling LLM with prompt: {prompt}")
        start_time = time.time()
        response = call_llm(prompt)
        end_time = time.time()
        
        if response:
            logger.info(f"LLM call successful in {end_time - start_time:.2f} seconds")
            logger.info(f"Response: {response}")
            return True
        else:
            logger.error("LLM call returned empty response")
            return False
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return False

def test_providers():
    """Test each provider individually."""
    logger.info("Testing individual providers...")
    
    try:
        # Import the provider functions
        from agent_self_reflection.llm import (
            call_zuki, call_electronhub, call_zanity, call_a4f, call_gemini
        )
        
        # Test prompt
        prompt = "What is the capital of France? Keep your answer short."
        
        # Test each provider
        providers = [
            ("Zuki", call_zuki),
            ("ElectronHub", call_electronhub),
            ("Zanity", call_zanity),
            ("A4F", call_a4f),
            ("Gemini", call_gemini)
        ]
        
        results = {}
        for name, func in providers:
            logger.info(f"Testing provider: {name}")
            try:
                start_time = time.time()
                if name == "A4F":
                    response = func(prompt)
                else:
                    response = func(prompt, None)
                end_time = time.time()
                
                success = response is not None and len(response) > 0
                results[name] = {
                    "success": success,
                    "time": end_time - start_time,
                    "response": response[:100] + "..." if response and len(response) > 100 else response
                }
                
                if success:
                    logger.info(f"{name}: Success in {end_time - start_time:.2f} seconds")
                else:
                    logger.error(f"{name}: Failed or empty response")
            except Exception as e:
                logger.error(f"{name}: Error - {e}")
                results[name] = {"success": False, "error": str(e)}
        
        # Check if at least one provider is working
        working_providers = [name for name, result in results.items() if result.get("success", False)]
        if working_providers:
            logger.info(f"Working providers: {working_providers}")
            return True
        else:
            logger.error("No working providers found")
            return False
    except Exception as e:
        logger.error(f"Error testing providers: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting LLM module tests")
    
    # Test config
    config_ok = test_config()
    logger.info(f"Config test: {'PASSED' if config_ok else 'FAILED'}")
    
    # Test imports
    imports_ok = test_imports()
    logger.info(f"Import test: {'PASSED' if imports_ok else 'FAILED'}")
    
    # Test providers
    providers_ok = test_providers()
    logger.info(f"Provider test: {'PASSED' if providers_ok else 'FAILED'}")
    
    # Test LLM call
    llm_ok = test_llm_call()
    logger.info(f"LLM call test: {'PASSED' if llm_ok else 'FAILED'}")
    
    # Overall result
    if config_ok and imports_ok and (providers_ok or llm_ok):
        logger.info("All critical tests PASSED")
        print("\n✅ LLM module is working correctly!")
        return 0
    else:
        logger.error("Some tests FAILED")
        print("\n❌ LLM module has issues. Check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
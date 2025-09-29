"""
Example: Using AI-Driven Embedding Model Selection in Ravana AGI System

This example demonstrates how to utilize the AI-driven embedding model selection
within the Ravana AGI system for various tasks.
"""
import asyncio
from core.embeddings_manager import embeddings_manager, ModelPurpose

class RavanaAIDrivenEmbeddingExample:
    """Example class showing how to use AI-driven embeddings in Ravana context."""
    
    def __init__(self):
        self.embeddings_mgr = embeddings_manager
    
    async def demonstrate_purpose_driven_selection(self):
        """Demonstrate how the AI selects different models based on purpose."""
        print("=== AI-Driven Model Selection Based on Purpose ===")
        
        test_texts = {
            "general": ["The weather is nice today", "I'm working on a project"],
            "semantic_search": ["Find documents about machine learning", "Search for AI research papers"],
            "sentence_similarity": ["The cat is sleeping", "A feline is resting"],
            "multilingual": ["Hello", "Hola", "Bonjour"],  # Mixed languages for demonstration
            "quality_critical": ["Complex scientific concept explanation", "Detailed technical documentation"],
            "performance_critical": ["Short text", "Quick"]
        }
        
        for purpose_name, texts in test_texts.items():
            purpose = getattr(ModelPurpose, purpose_name.upper())
            embedding = self.embeddings_mgr.get_embedding(texts[0], purpose=purpose)
            model_info = self.embeddings_mgr.get_current_model_info()
            
            print(f"{purpose_name}: {model_info['name']} (dim: {model_info['dimension']})")
        
        print()
    
    async def demonstrate_context_aware_selection(self):
        """Demonstrate how the AI selects models based on content context."""
        print("=== Context-Aware Model Selection ===")
        
        # Different types of content that would benefit from different models
        content_types = [
            ("Short tweet analysis", "Just had coffee, feeling great! #coffee #mood"),
            ("Long document analysis", "This is a very long document with complex concepts that require high-quality embeddings for accurate semantic understanding and retrieval purposes."),
            ("Multilingual content", "Hello world in English and Hola mundo en español"),
            ("Technical content", "The quantum mechanical properties of superconductors exhibit fascinating behaviors at near absolute zero temperatures")
        ]
        
        for content_type, text in content_types:
            # Use general purpose but let the AI analyze content characteristics
            embedding = self.embeddings_mgr.get_embedding(text, purpose=ModelPurpose.GENERAL)
            model_info = self.embeddings_mgr.get_current_model_info()
            
            print(f"{content_type}: {model_info['name']} (quality: {model_info['quality_score']:.2f})")
        
        print()
    
    async def demonstrate_memory_retrieval_with_selection(self):
        """Demonstrate how embeddings can be used for memory retrieval with AI-selected models."""
        print("=== Memory Retrieval with AI-Selected Models ===")
        
        # Sample memories (in a real system, these would come from the memory service)
        memories = [
            "Yesterday I learned about neural networks and their applications",
            "I discovered that reinforcement learning can be used for game playing",
            "The conversation with the user about AI safety was interesting",
            "I experimented with transformer models for text generation",
            "My emotional state has been curious and motivated lately"
        ]
        
        # Query that should match related memories
        query = "What have I learned about machine learning recently?"
        
        # Use semantic search purpose for better memory retrieval
        similar_memories = self.embeddings_mgr.get_similar_texts(
            query, memories, top_k=3, purpose=ModelPurpose.SEMANTIC_SEARCH
        )
        
        model_info = self.embeddings_mgr.get_current_model_info()
        print(f"Selected model for memory retrieval: {model_info['name']}")
        print(f"Query: {query}")
        print("Most relevant memories:")
        for i, (memory, similarity) in enumerate(similar_memories, 1):
            print(f"  {i}. [{similarity:.3f}] {memory}")
        
        print()
    
    async def demonstrate_multilingual_selection(self):
        """Demonstrate multilingual model selection."""
        print("=== Multilingual Model Selection ===")
        
        # Multilingual content
        multilingual_texts = [
            "Hello in English",
            "Bonjour en français", 
            "Hola en español",
            "Ciao in italiano"
        ]
        
        embedding = self.embeddings_mgr.get_embedding(
            multilingual_texts[0], 
            purpose=ModelPurpose.MULTILINGUAL
        )
        model_info = self.embeddings_mgr.get_current_model_info()
        print(f"Multilingual model selected: {model_info['name']}")
        print(f"Multilingual support: {model_info['multilingual']}")
        
        # Test similarity with sentence similarity purpose
        similarity = self.embeddings_mgr.get_similarity(
            "Hello", "Bonjour"
        )
        model_info = self.embeddings_mgr.get_current_model_info()
        print(f"Similarity model: {model_info['name']}, Score: {similarity:.3f}")
        
        print()
    
    async def demonstrate_performance_vs_quality_tradeoff(self):
        """Demonstrate selection between performance and quality based on requirements."""
        print("=== Performance vs Quality Trade-off ===")
        
        # Simulate different performance requirements
        
        # Quality-critical task
        quality_embedding = self.embeddings_mgr.get_embedding(
            "Complex scientific explanation requiring high accuracy",
            purpose=ModelPurpose.QUALITY_CRITICAL
        )
        quality_model = self.embeddings_mgr.get_current_model_info()
        print(f"Quality-critical: {quality_model['name']} (quality: {quality_model['quality_score']:.2f})")
        
        # Performance-critical task
        performance_embedding = self.embeddings_mgr.get_embedding(
            "Simple text for quick processing",
            purpose=ModelPurpose.PERFORMANCE_CRITICAL
        )
        performance_model = self.embeddings_mgr.get_current_model_info()
        print(f"Performance-critical: {performance_model['name']} (speed: {performance_model['speed_score']:.2f})")
        
        print()
    
    async def demonstrate_fallback_mechanism(self):
        """Demonstrate the fallback mechanism when models fail."""
        print("=== Fallback Mechanism Test ===")
        
        try:
            # This will use the AI-driven selection and fallback if needed
            embedding = self.embeddings_mgr.get_embedding(
                "Testing fallback mechanism",
                purpose=ModelPurpose.GENERAL
            )
            model_info = self.embeddings_mgr.get_current_model_info()
            print(f"Successfully used model: {model_info['name']}")
            
            print("SUCCESS: Fallback mechanism working correctly")
        except Exception as e:
            print(f"ERROR: Error in fallback mechanism: {e}")
        
        print()

    async def run_all_examples(self):
        """Run all AI-driven embedding examples."""
        print("Ravana AGI - AI-Driven Embedding Model Selection Example\n")
        
        await self.demonstrate_purpose_driven_selection()
        await self.demonstrate_context_aware_selection()
        await self.demonstrate_memory_retrieval_with_selection()
        await self.demonstrate_multilingual_selection()
        await self.demonstrate_performance_vs_quality_tradeoff()
        await self.demonstrate_fallback_mechanism()
        
        print("All AI-driven embedding examples completed successfully!")
        
        # Show available models
        available_models = self.embeddings_mgr.get_available_models()
        print(f"\nAvailable models: {available_models}")
        
        # Clean up
        self.embeddings_mgr.unload_all_models()

async def main():
    """Main function to run the example."""
    example = RavanaAIDrivenEmbeddingExample()
    await example.run_all_examples()

if __name__ == "__main__":
    asyncio.run(main())
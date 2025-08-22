-- Multi-Modal Memory System Database Schema
-- Requires PostgreSQL with pgvector extension

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main memory records table
CREATE TABLE memory_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_type VARCHAR(50) NOT NULL, 
    content_text TEXT,
    content_metadata JSONB DEFAULT '{}',
    file_path VARCHAR(1000),
    
    -- Multi-modal embeddings
    text_embedding vector(384),      -- SentenceTransformers (all-MiniLM-L6-v2)
    image_embedding vector(512),     -- CLIP embeddings
    audio_embedding vector(512),     -- Audio feature embeddings
    unified_embedding vector(1024),  -- Cross-modal unified embeddings
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    memory_type VARCHAR(50) DEFAULT 'episodic',
    emotional_valence FLOAT,
    confidence_score FLOAT DEFAULT 1.0,
    
    -- Search optimization
    search_vector tsvector,
    tags TEXT[] DEFAULT '{}',
    
    CONSTRAINT valid_content_type CHECK (content_type IN ('text', 'audio', 'image', 'video'))
);

-- Audio-specific metadata table
CREATE TABLE audio_memories (
    memory_id UUID PRIMARY KEY REFERENCES memory_records(id) ON DELETE CASCADE,
    transcript TEXT,
    language_code VARCHAR(10),
    confidence_scores JSONB DEFAULT '{}',
    duration_seconds FLOAT,
    audio_features JSONB DEFAULT '{}',
    sample_rate INTEGER,
    channels INTEGER
);

-- Image-specific metadata table
CREATE TABLE image_memories (
    memory_id UUID PRIMARY KEY REFERENCES memory_records(id) ON DELETE CASCADE,
    width INTEGER,
    height INTEGER,
    object_detections JSONB DEFAULT '{}',
    scene_description TEXT,
    image_hash VARCHAR(64),
    color_palette JSONB DEFAULT '{}',
    image_features JSONB DEFAULT '{}'
);

-- Video-specific metadata table  
CREATE TABLE video_memories (
    memory_id UUID PRIMARY KEY REFERENCES memory_records(id) ON DELETE CASCADE,
    duration_seconds FLOAT,
    frame_rate FLOAT,
    width INTEGER,
    height INTEGER,
    video_features JSONB DEFAULT '{}',
    thumbnail_path VARCHAR(1000)
);

-- Memory consolidation tracking
CREATE TABLE memory_consolidations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_memory_ids UUID[] NOT NULL,
    consolidated_memory_id UUID REFERENCES memory_records(id) ON DELETE CASCADE,
    consolidation_type VARCHAR(50) DEFAULT 'llm_based',
    consolidation_prompt TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector indexes for similarity search
CREATE INDEX idx_text_embedding ON memory_records USING ivfflat (text_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_image_embedding ON memory_records USING ivfflat (image_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_audio_embedding ON memory_records USING ivfflat (audio_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_unified_embedding ON memory_records USING ivfflat (unified_embedding vector_cosine_ops) WITH (lists = 100);

-- Standard indexes for performance
CREATE INDEX idx_memory_records_content_type ON memory_records(content_type);
CREATE INDEX idx_memory_records_created_at ON memory_records(created_at);
CREATE INDEX idx_memory_records_last_accessed ON memory_records(last_accessed);
CREATE INDEX idx_memory_records_memory_type ON memory_records(memory_type);
CREATE INDEX idx_memory_records_tags ON memory_records USING GIN(tags);

-- Full-text search index
CREATE INDEX idx_memory_records_search_vector ON memory_records USING GIN(search_vector);

-- Trigger to automatically update search_vector
CREATE OR REPLACE FUNCTION update_search_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_search_vector 
    BEFORE INSERT OR UPDATE ON memory_records 
    FOR EACH ROW 
    EXECUTE FUNCTION update_search_vector();

-- Function to calculate embedding similarity
CREATE OR REPLACE FUNCTION embedding_similarity(embedding1 vector, embedding2 vector) 
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (embedding1 <=> embedding2);
END;
$$ LANGUAGE plpgsql;

-- View for memory statistics
CREATE VIEW memory_stats AS
SELECT 
    content_type,
    memory_type,
    COUNT(*) as count,
    AVG(confidence_score) as avg_confidence,
    MIN(created_at) as oldest_memory,
    MAX(created_at) as newest_memory,
    AVG(access_count) as avg_access_count
FROM memory_records 
GROUP BY content_type, memory_type;

-- View for recent memories
CREATE VIEW recent_memories AS
SELECT 
    id,
    content_type,
    content_text,
    confidence_score,
    created_at,
    access_count
FROM memory_records 
ORDER BY created_at DESC 
LIMIT 100;

-- Sample data for testing (optional)
INSERT INTO memory_records (content_type, content_text, memory_type, confidence_score)
VALUES 
    ('text', 'This is a sample text memory for testing the multi-modal system.', 'episodic', 1.0),
    ('text', 'User prefers detailed technical explanations over simplified overviews.', 'semantic', 0.9);
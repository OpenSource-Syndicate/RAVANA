-- Very Long-Term Memory System Database Schema Extension
-- This file extends the existing RAVANA database with tables for very long-term memory

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types for better type safety
CREATE TYPE memory_type AS ENUM (
    'strategic_knowledge',
    'architectural_insight', 
    'evolution_pattern',
    'meta_learning_rule',
    'critical_failure',
    'successful_improvement',
    'failed_experiment',
    'code_pattern',
    'behavioral_pattern'
);

CREATE TYPE pattern_type AS ENUM (
    'temporal',
    'causal',
    'behavioral',
    'structural',
    'performance'
);

CREATE TYPE consolidation_type AS ENUM (
    'daily',
    'weekly',
    'monthly',
    'quarterly',
    'event_triggered'
);

-- Very Long-Term Memory Records Table
CREATE TABLE IF NOT EXISTS very_long_term_memories (
    memory_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    memory_type memory_type NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    promoted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    strategic_value REAL DEFAULT 0.5 CHECK (strategic_value >= 0.0 AND strategic_value <= 1.0),
    compressed_content JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    source_session VARCHAR(255) DEFAULT 'unknown',
    related_memories JSONB DEFAULT '[]',
    retention_category VARCHAR(100) DEFAULT 'permanent',
    
    -- Indexes for performance
    CONSTRAINT valid_compressed_content CHECK (jsonb_typeof(compressed_content) = 'object'),
    CONSTRAINT valid_metadata CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT valid_related_memories CHECK (jsonb_typeof(related_memories) = 'array')
);

-- Memory Patterns Table
CREATE TABLE IF NOT EXISTS memory_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    pattern_type pattern_type NOT NULL,
    pattern_description TEXT NOT NULL,
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    pattern_data JSONB NOT NULL,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    supporting_memories JSONB DEFAULT '[]',
    validation_count INTEGER DEFAULT 0,
    last_validated TIMESTAMP WITH TIME ZONE,
    source_memory_id VARCHAR(255) REFERENCES very_long_term_memories(memory_id),
    
    CONSTRAINT valid_pattern_data CHECK (jsonb_typeof(pattern_data) = 'object'),
    CONSTRAINT valid_supporting_memories CHECK (jsonb_typeof(supporting_memories) = 'array')
);

-- Memory Consolidations Table
CREATE TABLE IF NOT EXISTS memory_consolidations (
    consolidation_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    consolidation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    consolidation_type consolidation_type NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    patterns_extracted INTEGER DEFAULT 0,
    compression_ratio REAL DEFAULT 1.0,
    consolidation_results JSONB DEFAULT '{}',
    processing_time_seconds REAL DEFAULT 0.0,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    CONSTRAINT valid_consolidation_results CHECK (jsonb_typeof(consolidation_results) = 'object')
);

-- Strategic Knowledge Table
CREATE TABLE IF NOT EXISTS strategic_knowledge (
    knowledge_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    knowledge_domain VARCHAR(255) NOT NULL,
    knowledge_summary TEXT NOT NULL,
    confidence_level REAL NOT NULL CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_patterns JSONB DEFAULT '[]',
    knowledge_structure JSONB DEFAULT '{}',
    validation_score REAL DEFAULT 0.5 CHECK (validation_score >= 0.0 AND validation_score <= 1.0),
    application_count INTEGER DEFAULT 0,
    
    CONSTRAINT valid_source_patterns CHECK (jsonb_typeof(source_patterns) = 'array'),
    CONSTRAINT valid_knowledge_structure CHECK (jsonb_typeof(knowledge_structure) = 'object')
);

-- Consolidation Metrics Table
CREATE TABLE IF NOT EXISTS consolidation_metrics (
    metric_id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    consolidation_id VARCHAR(255) NOT NULL REFERENCES memory_consolidations(consolidation_id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Junction table for pattern-strategic knowledge relationships
CREATE TABLE IF NOT EXISTS pattern_strategic_knowledge (
    pattern_id VARCHAR(255) REFERENCES memory_patterns(pattern_id),
    knowledge_id VARCHAR(255) REFERENCES strategic_knowledge(knowledge_id),
    contribution_weight REAL DEFAULT 1.0,
    PRIMARY KEY (pattern_id, knowledge_id)
);

-- Junction table for consolidation-pattern relationships  
CREATE TABLE IF NOT EXISTS consolidation_patterns (
    consolidation_id VARCHAR(255) REFERENCES memory_consolidations(consolidation_id),
    pattern_id VARCHAR(255) REFERENCES memory_patterns(pattern_id),
    extraction_confidence REAL DEFAULT 1.0,
    PRIMARY KEY (consolidation_id, pattern_id)
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_vltm_memory_type ON very_long_term_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_vltm_created_at ON very_long_term_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_vltm_last_accessed ON very_long_term_memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_vltm_importance_score ON very_long_term_memories(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_vltm_strategic_value ON very_long_term_memories(strategic_value DESC);
CREATE INDEX IF NOT EXISTS idx_vltm_source_session ON very_long_term_memories(source_session);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON memory_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON memory_patterns(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_discovered_at ON memory_patterns(discovered_at);
CREATE INDEX IF NOT EXISTS idx_patterns_source_memory ON memory_patterns(source_memory_id);

CREATE INDEX IF NOT EXISTS idx_consolidations_date ON memory_consolidations(consolidation_date);
CREATE INDEX IF NOT EXISTS idx_consolidations_type ON memory_consolidations(consolidation_type);
CREATE INDEX IF NOT EXISTS idx_consolidations_success ON memory_consolidations(success);

CREATE INDEX IF NOT EXISTS idx_strategic_domain ON strategic_knowledge(knowledge_domain);
CREATE INDEX IF NOT EXISTS idx_strategic_confidence ON strategic_knowledge(confidence_level DESC);
CREATE INDEX IF NOT EXISTS idx_strategic_updated ON strategic_knowledge(last_updated);
CREATE INDEX IF NOT EXISTS idx_strategic_application_count ON strategic_knowledge(application_count DESC);

-- JSONB Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_vltm_metadata_gin ON very_long_term_memories USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_vltm_content_gin ON very_long_term_memories USING GIN(compressed_content);
CREATE INDEX IF NOT EXISTS idx_patterns_data_gin ON memory_patterns USING GIN(pattern_data);
CREATE INDEX IF NOT EXISTS idx_strategic_structure_gin ON strategic_knowledge USING GIN(knowledge_structure);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_patterns_description_text ON memory_patterns USING GIN(to_tsvector('english', pattern_description));
CREATE INDEX IF NOT EXISTS idx_strategic_summary_text ON strategic_knowledge USING GIN(to_tsvector('english', knowledge_summary));

-- Create views for common queries
CREATE OR REPLACE VIEW vltm_summary AS
SELECT 
    memory_type,
    COUNT(*) as memory_count,
    AVG(importance_score) as avg_importance,
    AVG(strategic_value) as avg_strategic_value,
    MAX(last_accessed) as most_recent_access
FROM very_long_term_memories 
GROUP BY memory_type;

CREATE OR REPLACE VIEW pattern_summary AS
SELECT 
    pattern_type,
    COUNT(*) as pattern_count,
    AVG(confidence_score) as avg_confidence,
    AVG(validation_count) as avg_validations
FROM memory_patterns 
GROUP BY pattern_type;

CREATE OR REPLACE VIEW consolidation_performance AS
SELECT 
    consolidation_type,
    COUNT(*) as total_consolidations,
    AVG(memories_processed) as avg_memories_processed,
    AVG(patterns_extracted) as avg_patterns_extracted,
    AVG(processing_time_seconds) as avg_processing_time,
    AVG(compression_ratio) as avg_compression_ratio
FROM memory_consolidations 
WHERE success = TRUE
GROUP BY consolidation_type;

-- Functions for maintenance and optimization
CREATE OR REPLACE FUNCTION update_memory_access(memory_id_param VARCHAR(255))
RETURNS VOID AS $$
BEGIN
    UPDATE very_long_term_memories 
    SET 
        last_accessed = NOW(),
        access_count = access_count + 1
    WHERE memory_id = memory_id_param;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_old_metrics()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM consolidation_metrics 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_memory_statistics()
RETURNS TABLE(
    total_memories BIGINT,
    total_patterns BIGINT,
    total_strategic_knowledge BIGINT,
    avg_importance NUMERIC,
    oldest_memory TIMESTAMP WITH TIME ZONE,
    newest_memory TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM very_long_term_memories),
        (SELECT COUNT(*) FROM memory_patterns),
        (SELECT COUNT(*) FROM strategic_knowledge),
        (SELECT AVG(importance_score) FROM very_long_term_memories),
        (SELECT MIN(created_at) FROM very_long_term_memories),
        (SELECT MAX(created_at) FROM very_long_term_memories);
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic updates
CREATE OR REPLACE FUNCTION update_strategic_knowledge_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER strategic_knowledge_update_trigger
    BEFORE UPDATE ON strategic_knowledge
    FOR EACH ROW
    EXECUTE FUNCTION update_strategic_knowledge_timestamp();

-- Comments for documentation
COMMENT ON TABLE very_long_term_memories IS 'Core very long-term memory records with compressed content and metadata';
COMMENT ON TABLE memory_patterns IS 'Patterns extracted from memory analysis with confidence scores';
COMMENT ON TABLE memory_consolidations IS 'History of memory consolidation processes and their results';
COMMENT ON TABLE strategic_knowledge IS 'High-level strategic knowledge derived from memory patterns';
COMMENT ON TABLE consolidation_metrics IS 'Performance metrics for consolidation operations';

COMMENT ON COLUMN very_long_term_memories.compressed_content IS 'JSONB compressed representation of memory content';
COMMENT ON COLUMN very_long_term_memories.importance_score IS 'Calculated importance score (0.0-1.0)';
COMMENT ON COLUMN very_long_term_memories.strategic_value IS 'Strategic value for long-term planning (0.0-1.0)';
COMMENT ON COLUMN memory_patterns.confidence_score IS 'Confidence in pattern validity (0.0-1.0)';
COMMENT ON COLUMN strategic_knowledge.application_count IS 'Number of times this knowledge was applied';
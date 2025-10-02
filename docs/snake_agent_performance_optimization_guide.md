# Snake Agent Performance Optimization Guide

## Overview
This document details the performance improvements made to the Snake Agent and provides best practices for achieving peak performance.

## Key Performance Improvements

### 1. Thread and Process Management Optimization
- **Increased thread counts**: Raised `SNAKE_MAX_THREADS` from 12 to 16 for better concurrent analysis
- **Increased process counts**: Raised `SNAKE_MAX_PROCESSES` from 6 to 8 for enhanced experimentation
- **Analysis threads**: Increased `SNAKE_ANALYSIS_THREADS` from 4 to 6 to improve code analysis throughput

### 2. File Monitoring Efficiency
- **Faster monitoring**: Reduced `SNAKE_MONITOR_INTERVAL` from 1.0s to 0.5s for more responsive file change detection
- **Enhanced peek prioritizer**: Enabled `SNAKE_USE_PEEK_PRIORITIZER` to efficiently select important files for analysis

### 3. Enhanced Error Handling and Recovery
- **Fixed VLTM initialization errors**: Resolved the `existing_memory_service` parameter issue in `MemoryIntegrationManager`
- **Added retrieval engine**: Properly initialized `AdvancedRetrievalEngine` for complete VLTM functionality
- **Improved auto-recovery**: Enhanced `SNAKE_AUTO_RECOVERY` mechanism for better resilience

### 4. Resource Management
- **Larger queues**: Increased `SNAKE_MAX_QUEUE_SIZE` from 2000 to 5000 for better task buffering
- **Extended timeouts**: Raised `SNAKE_TASK_TIMEOUT` from 600s to 900s (15 minutes) for complex tasks
- **More frequent heartbeats**: Reduced `SNAKE_HEARTBEAT_INTERVAL` from 5s to 2s for better monitoring

### 5. Memory and Storage Optimization
- **Efficient memory consolidation**: Improved algorithms for consolidating long-term memories
- **Better storage backend**: Enhanced storage backend initialization and management
- **Increased retention**: Extended `SNAKE_LOG_RETENTION_DAYS` from 60 to 90 days

## Configuration Files

### Environment Variables
The optimized configuration is provided in three formats:

1. **Unix/Linux/Mac**: `snake_agent_optimized_config.sh`
2. **Windows**: `snake_agent_optimized_config.bat`
3. **Python**: `optimized_snake_config.py`

### Loading the Configuration

#### On Unix/Linux/Mac:
```bash
source snake_agent_optimized_config.sh
```

#### On Windows (Command Prompt):
```cmd
snake_agent_optimized_config.bat
```

#### In Python code:
```python
import optimized_snake_config
```

## Best Practices

### 1. Performance Tuning Guidelines
- **Thread counts**: Set `SNAKE_MAX_THREADS` to 1-2x your CPU core count
- **Process counts**: Set `SNAKE_MAX_PROCESSES` to 0.5-1x your CPU core count
- **Queue sizes**: Adjust `SNAKE_MAX_QUEUE_SIZE` based on your workload patterns
- **Monitoring interval**: Lower values (0.5s) for responsive systems, higher values (2s) for efficiency

### 2. Resource Management
- **Memory usage**: Monitor memory consumption and adjust thread/process counts accordingly
- **Disk space**: Ensure adequate space for VLTM storage and logging (several GB recommended for production)
- **CPU utilization**: Set concurrency parameters to avoid saturating system resources

### 3. Monitoring and Metrics
- **Enable performance monitoring**: Use `SNAKE_PERF_MONITORING=True` to track agent metrics
- **Regular health checks**: Implement monitoring for sustained operations
- **Alerting**: Set up alerts for performance degradation or errors

### 4. Maintenance
- **Regular cleanup**: Ensure `SNAKE_CLEANUP_INTERVAL` is set appropriately for your system
- **Log rotation**: Monitor log file sizes and retention
- **Memory consolidation**: Allow sufficient time for memory consolidation cycles

## Performance Testing Framework

### Running Performance Tests
```bash
python -m pytest tests/test_snake_agent_performance.py -v
```

### Running Load Tests
```bash
python tests/load_test_framework.py --quick
```

### Comprehensive Performance Analysis
```bash
python core/snake_config_analyzer.py
```

## Monitoring Tools

### Real-time Performance Dashboard
```python
from core.snake_performance_monitor import MetricsDashboard, PerformanceAnalyzer, MetricsCollector
from core.snake_agent_enhanced import EnhancedSnakeAgent

# Initialize and start dashboard
collector = MetricsCollector(agent, collection_interval=2.0)
analyzer = PerformanceAnalyzer(collector)
dashboard = MetricsDashboard(analyzer)

await collector.start_collection()
await dashboard.start_dashboard()
```

### Metrics Export
Performance metrics can be exported to JSON format for analysis:
```python
from core.snake_performance_monitor import MetricsExporter

exporter = MetricsExporter(collector)
exporter.export_to_json()
exporter.generate_summary_report()
```

## Troubleshooting Common Issues

### Slow Initialization
- Check network connectivity for LLM providers
- Verify VLTM storage backend connection
- Ensure sufficient system resources

### High Memory Usage
- Reduce thread/process counts
- Increase cleanup frequency
- Review memory consolidation settings

### Low Throughput
- Increase queue sizes
- Optimize file monitoring interval
- Check for system resource constraints

## Future Improvements

### Planned Enhancements
- Further optimization of LLM usage with caching and request batching
- Advanced peek prioritizer with ML-based file importance scoring
- Enhanced self-evaluation capabilities with deeper performance analysis
- More sophisticated memory consolidation algorithms

### Performance Benchmarks
- Baseline performance metrics established
- Continuous monitoring of performance regressions
- Automated performance testing in CI/CD pipeline
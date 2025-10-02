# Snake Agent Testing and Optimization Guide

This guide explains how to test, monitor, and optimize the Snake Agent's performance in the RAVANA system.

## Table of Contents
1. [Overview](#overview)
2. [Running Performance Tests](#running-performance-tests)
3. [Monitoring Snake Agent Performance](#monitoring-snake-agent-performance)
4. [Configuration Optimization](#configuration-optimization)
5. [Threading and Multiprocessing Optimization](#threading-and-multiprocessing-optimization)
6. [Error Handling and Auto-Recovery](#error-handling-and-auto-recovery)
7. [UV Dependency Management](#uv-dependency-management)
8. [Best Practices](#best-practices)

## Overview

The Snake Agent is an autonomous code analysis and improvement system that runs continuously in the background to monitor, analyze, and enhance the RAVANA codebase. This guide provides instructions on how to test its performance, monitor its operation, and optimize its capabilities.

## Running Performance Tests

### Running the Comprehensive Test Suite

To run the comprehensive Snake Agent performance test suite:

```bash
python test_snake_agent_performance.py
```

This test suite includes:
- Initialization tests for both basic and enhanced snake agents
- Configuration validation tests
- Threading manager performance tests
- Process manager performance tests
- Resource usage measurements
- Detailed performance analysis with varying thread counts

### Running Individual Test Components

For specific tests, you can run:

```bash
# Run with pytest for more detailed output
python -m pytest tests/test_enhanced_snake_agent.py -v

# Run the simple core functionality test
python tests/simple_snake_test.py
```

## Monitoring Snake Agent Performance

### Real-time Performance Monitoring

To monitor the Snake Agent in real-time, use the performance monitoring script:

```bash
python snake_performance_monitor.py --short-test
```

This will initialize a Snake Agent instance and collect performance metrics over a short period.

### Analyzing Historical Performance Data

To analyze previously collected performance data:

```bash
python snake_performance_monitor.py --analyze-file snake_performance.json
```

### Key Metrics to Monitor

- **Processing Rate**: Tasks processed per minute
- **Error Rate**: Percentage of failed tasks
- **Resource Usage**: CPU and memory consumption
- **Queue Sizes**: Thread and process queue backlogs
- **Active Workers**: Number of active threads and processes
- **Task Completion**: Files analyzed, experiments completed, improvements applied

## Configuration Optimization

### Environment Variables for Performance Tuning

The Snake Agent can be tuned using environment variables in the `core/config.py` file:

```python
# Enhanced threading configuration
SNAKE_MAX_THREADS = int(os.environ.get("SNAKE_MAX_THREADS", "12"))
SNAKE_MAX_PROCESSES = int(os.environ.get("SNAKE_MAX_PROCESSES", "6"))
SNAKE_ANALYSIS_THREADS = int(os.environ.get("SNAKE_ANALYSIS_THREADS", "4"))

# Performance monitoring
SNAKE_PERF_MONITORING = bool(os.environ.get("SNAKE_PERF_MONITORING", "True").lower() in ["true", "1", "yes"])

# Task and queue settings
SNAKE_TASK_TIMEOUT = float(os.environ.get("SNAKE_TASK_TIMEOUT", "600.0"))  # 10 minutes
SNAKE_MAX_QUEUE_SIZE = int(os.environ.get("SNAKE_MAX_QUEUE_SIZE", "2000"))
SNAKE_MONITOR_INTERVAL = float(os.environ.get("SNAKE_MONITOR_INTERVAL", "1.0"))  # 1 second
```

### Recommended Performance Configurations

For high-performance environments:
```bash
export SNAKE_MAX_THREADS=16
export SNAKE_MAX_PROCESSES=8
export SNAKE_ANALYSIS_THREADS=6
export SNAKE_MAX_QUEUE_SIZE=5000
export SNAKE_PERF_MONITORING=True
```

For resource-constrained environments:
```bash
export SNAKE_MAX_THREADS=4
export SNAKE_MAX_PROCESSES=2
export SNAKE_ANALYSIS_THREADS=2
export SNAKE_MAX_QUEUE_SIZE=1000
export SNAKE_PERF_MONITORING=False
```

## Threading and Multiprocessing Optimization

### Understanding the Architecture

The Snake Agent uses a hybrid threading and multiprocessing architecture:

**Threading Components:**
- File Monitor Threads: Monitor file system changes
- Indexing Threads: Perform code analysis and indexing (parallelized)
- Study Threads: Perform deep analysis (serialized)
- Communication Threads: Handle communication with the main system

**Process Components:**
- Experiment Processes: Run experiments safely in isolated processes
- Analysis Processes: Perform deep analysis on CPU-intensive tasks
- Improvement Processes: Handle improvement implementations

### Optimizing Thread Counts

The optimal thread configuration depends on your system:

- **Analysis Threads**: Should be set to 25-50% of max threads, depending on workload
- **Study Threads**: Usually kept at 1 since some analysis tasks need to be serialized
- **File Monitor Threads**: Usually 1 is sufficient
- **Communication Threads**: Usually 1 is sufficient

### Queue Size Optimization

Queue sizes should be large enough to handle bursts of activity but not so large that they consume excessive memory:

- For heavy workloads: 2000-5000 items per queue
- For light workloads: 500-1000 items per queue

## Error Handling and Auto-Recovery

### Built-in Error Handling

The Snake Agent includes several error handling mechanisms:

1. **Component-Level Isolation**: Errors in one component don't affect others
2. **Exponential Backoff**: Failed components retry with increasing delays
3. **Graceful Degradation**: System continues operating even if some components fail
4. **Detailed Logging**: All errors are logged with full context

### Auto-Recovery Features

The system includes auto-recovery capabilities:

- **Thread Restart**: Failed threads are automatically restarted
- **Process Restart**: Failed processes are automatically restarted
- **State Recovery**: Agent state is persisted and restored on restart
- **Connection Recovery**: Lost connections are automatically reestablished

### Monitoring for Errors

Watch for these key error indicators:
```python
# Check the agent status for error counts
status = await snake_agent.get_status()
error_metrics = {
    "thread_errors": sum(thread.get("error_count", 0) for thread in status.get("threading_status", {}).values()),
    "process_errors": sum(process.get("tasks_failed", 0) for process in status.get("process_status", {}).values()),
}
```

## UV Dependency Management

### Setting up UV

RAVANA uses UV for fast Python package management. To set it up:

1. **Install UV** (if not already installed):
   ```bash
   # Install using the official installation script
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or with Homebrew on macOS
   brew install uv
   # Or with pip
   pip install uv
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies using UV**:
   ```bash
   # Install all dependencies quickly
   uv pip install -r requirements.txt
   # Or install from pyproject.toml
   uv pip install -e .
   ```

### Benefits of Using UV

- **Faster installation**: UV is significantly faster than pip
- **Dependency resolution**: Better and faster dependency resolution
- **Virtual environment management**: Built-in venv management
- **Consistent environments**: More reproducible builds

### Updating Dependencies with UV

```bash
# Update all dependencies to the latest compatible versions
uv pip sync requirements.txt

# Install a specific package
uv pip install package_name

# Update pyproject.toml dependencies
uv pip compile pyproject.toml --output-file requirements.txt
```

## Best Practices

### Performance Optimization

1. **Monitor Resource Usage**: Regularly check CPU and memory usage to determine optimal thread/process counts
2. **Adjust Queue Sizes**: Tune queue sizes based on your workload patterns
3. **Use Appropriate Thread Counts**: Don't set thread counts higher than your CPU can effectively handle
4. **Enable Performance Monitoring**: Keep performance monitoring enabled in production for ongoing optimization
5. **Regular Testing**: Run performance tests regularly as the codebase evolves

### Operational Best Practices

1. **Logging and Monitoring**: Keep detailed logs and monitor system metrics continuously
2. **State Persistence**: Ensure state is properly persisted for recovery
3. **Error Handling**: Implement proper error handling and alerting
4. **Regular Maintenance**: Perform regular cleanup and optimization tasks
5. **Backup and Recovery**: Implement proper backup and recovery procedures

### Development Best Practices

1. **Testing**: Write comprehensive tests for new Snake Agent functionality
2. **Performance Testing**: Always test performance impact of new features
3. **Error Handling**: Implement comprehensive error handling in new code
4. **Documentation**: Document new features and configuration options
5. **Code Quality**: Maintain high code quality standards for self-improvement code

## Troubleshooting

### Common Issues and Solutions

1. **High Memory Usage**:
   - Reduce thread and queue counts
   - Check for memory leaks in custom analysis code
   - Implement proper resource cleanup

2. **Low Processing Rate**:
   - Increase thread counts (but not beyond CPU capacity)
   - Optimize analysis algorithms
   - Check for I/O bottlenecks

3. **High Error Rates**:
   - Review logs for root causes
   - Improve error handling in analysis code
   - Validate input data more thoroughly

### Performance Tuning Steps

1. Start with conservative settings (4-8 threads, 2-4 processes)
2. Monitor performance under load
3. Gradually increase thread/process counts while monitoring metrics
4. Stop increasing when performance plateaus or degrades
5. Monitor resource usage to ensure you're not overloading the system
6. Repeat the process periodically as workload changes

This guide provides a comprehensive overview of how to test, monitor, and optimize the Snake Agent for peak performance. Regular monitoring and tuning will ensure the agent continues to run efficiently as the RAVANA system evolves.
#!/usr/bin/env python3
"""
Debug Main.py
This script runs the main.py file with extensive debugging and tracing to identify where it's getting stuck.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import threading
import signal
import traceback
import faulthandler
import psutil

# Enable faulthandler to debug segfaults and deadlocks
faulthandler.enable()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DebugMain")

def trace_imports():
    """Add import tracing to Python."""
    import builtins
    original_import = builtins.__import__
    
    def import_tracer(name, *args, **kwargs):
        logger.debug(f"Importing: {name}")
        try:
            result = original_import(name, *args, **kwargs)
            logger.debug(f"Successfully imported: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to import {name}: {e}")
            raise
    
    builtins.__import__ = import_tracer
    logger.debug("Import tracing enabled")

def trace_threads():
    """Log information about all running threads periodically."""
    import threading
    
    def log_threads():
        while True:
            threads = threading.enumerate()
            logger.debug(f"Active threads ({len(threads)}):")
            for thread in threads:
                logger.debug(f"  - {thread.name} (daemon: {thread.daemon})")
            time.sleep(10)
    
    thread = threading.Thread(target=log_threads, name="ThreadTracer", daemon=True)
    thread.start()
    logger.debug("Thread tracing enabled")

def monitor_process(pid, interval=5):
    """Monitor a process and its children for resource usage."""
    try:
        proc = psutil.Process(pid)
        while True:
            try:
                # Get process info
                cpu_percent = proc.cpu_percent(interval=1)
                memory_mb = proc.memory_info().rss / 1024 / 1024
                status = proc.status()
                
                # Get thread info
                num_threads = proc.num_threads()
                
                # Get child processes
                children = proc.children(recursive=True)
                
                logger.debug(f"Process {pid} - Status: {status}, CPU: {cpu_percent}%, "
                            f"Memory: {memory_mb:.1f}MB, Threads: {num_threads}, "
                            f"Children: {len(children)}")
                
                # Check if any child processes are zombies
                for child in children:
                    try:
                        child_status = child.status()
                        if child_status == psutil.STATUS_ZOMBIE:
                            logger.warning(f"Zombie child process detected: {child.pid}")
                    except psutil.NoSuchProcess:
                        pass
                
                time.sleep(interval)
            except psutil.NoSuchProcess:
                logger.debug(f"Process {pid} no longer exists")
                break
            except Exception as e:
                logger.error(f"Error monitoring process {pid}: {e}")
                time.sleep(interval)
    except Exception as e:
        logger.error(f"Failed to monitor process {pid}: {e}")

def run_with_python_trace():
    """Run main.py with Python's trace module to get detailed execution flow."""
    logger.info("Running main.py with Python's trace module")
    
    cmd = [
        sys.executable, 
        "-m", "trace", 
        "--trace",
        "--ignore-dir", os.path.dirname(os.__file__),  # Ignore standard library
        "main.py", 
        "--auto",
        "--debug"
    ]
    
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Start monitoring the process
        monitor_thread = threading.Thread(
            target=monitor_process,
            args=(process.pid, 5),
            daemon=True
        )
        monitor_thread.start()
        
        # Read and log output
        while True:
            output = process.stdout.readline()
            if output:
                print(f"STDOUT: {output.strip()}")
            
            error = process.stderr.readline()
            if error:
                print(f"STDERR: {error.strip()}")
            
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Get any remaining output
        remaining_output, remaining_errors = process.communicate()
        if remaining_output:
            print(f"STDOUT: {remaining_output.strip()}")
        if remaining_errors:
            print(f"STDERR: {remaining_errors.strip()}")
        
        return process.returncode
    except Exception as e:
        logger.error(f"Error running with trace: {e}")
        logger.debug(traceback.format_exc())
        return 1

def run_with_pdb():
    """Run main.py with Python debugger."""
    logger.info("Running main.py with Python debugger (pdb)")
    
    cmd = [
        sys.executable, 
        "-m", "pdb", 
        "main.py", 
        "--auto",
        "--debug"
    ]
    
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            env=env
        )
        
        process.wait()
        return process.returncode
    except Exception as e:
        logger.error(f"Error running with pdb: {e}")
        logger.debug(traceback.format_exc())
        return 1

def run_with_strace():
    """Run main.py with strace (Linux only) or similar tool to trace system calls."""
    if os.name != 'posix':
        logger.warning("strace is only available on Linux/Unix systems")
        return 1
    
    logger.info("Running main.py with strace")
    
    try:
        # Check if strace is available
        strace_check = subprocess.run(["which", "strace"], capture_output=True, text=True)
        if strace_check.returncode != 0:
            logger.warning("strace not found, skipping system call tracing")
            return 1
        
        cmd = [
            "strace", 
            "-f",  # Follow forks
            "-o", "strace_output.txt",  # Output file
            sys.executable, 
            "main.py", 
            "--auto",
            "--debug"
        ]
        
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd)
        process.wait()
        
        logger.info(f"strace output saved to strace_output.txt")
        return process.returncode
    except Exception as e:
        logger.error(f"Error running with strace: {e}")
        logger.debug(traceback.format_exc())
        return 1

def main():
    parser = argparse.ArgumentParser(description="Debug Main.py")
    parser.add_argument("--mode", choices=["trace", "pdb", "strace"], default="trace",
                       help="Debug mode: trace (execution flow), pdb (interactive debugger), or strace (system calls)")
    parser.add_argument("--trace-imports", action="store_true", help="Enable import tracing")
    parser.add_argument("--trace-threads", action="store_true", help="Enable thread tracing")
    args = parser.parse_args()
    
    logger.info(f"Starting debug_main.py with mode: {args.mode}")
    
    # Enable import tracing if requested
    if args.trace_imports:
        trace_imports()
    
    # Enable thread tracing if requested
    if args.trace_threads:
        trace_threads()
    
    # Run in the selected mode
    if args.mode == "trace":
        return run_with_python_trace()
    elif args.mode == "pdb":
        return run_with_pdb()
    elif args.mode == "strace":
        return run_with_strace()
    else:
        logger.error(f"Unknown debug mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
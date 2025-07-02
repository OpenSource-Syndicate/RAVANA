#!/usr/bin/env python3
"""
Run AGI System in Autonomous Mode
This script runs the AGI system in autonomous mode, where it generates and processes situations without user input.
Supports continuous 24/7 operation with automatic restart capability.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import threading
import signal
import psutil
import traceback
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_agi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutonomousAGI")

# Global variables for continuous operation
should_restart = True
restart_count = 0
max_restarts = 10  # Maximum number of restarts within restart_window
restart_window = 3600  # 1 hour window for counting restarts
restart_timestamps = []

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.debug("Starting dependency check...")
    try:
        import requests
        logger.debug("[OK] requests module imported")
        
        import sentence_transformers
        logger.debug(f"[OK] sentence_transformers module imported (version: {sentence_transformers.__version__})")
        
        import wikipedia
        logger.debug("[OK] wikipedia module imported")
        
        import uvicorn
        logger.debug("[OK] uvicorn module imported")
        
        import openai
        logger.debug(f"[OK] openai module imported (version: {openai.__version__})")
        
        try:
            from google import genai
            logger.debug("[OK] google.genai module imported")
        except ImportError:
            logger.warning("[FAIL] google.genai module not available - will attempt to install")
            raise ImportError("google.genai not found")
            
        logger.info("All required dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing missing dependencies...")
        try:
            logger.debug(f"Running pip install for missing packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "sentence-transformers", 
                                  "wikipedia", "uvicorn", "openai", "google-generativeai"])
            logger.info("Dependencies installed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return False

def kill_process_after_timeout(process, timeout):
    """Kill the process after the specified timeout."""
    logger.debug(f"Starting timeout monitor thread for {timeout} seconds")
    time.sleep(timeout)
    if process and process.poll() is None:
        logger.warning(f"Process did not complete within {timeout} seconds, terminating...")
        try:
            # Log process info before terminating
            proc = psutil.Process(process.pid)
            logger.debug(f"Process info before termination: CPU {proc.cpu_percent()}%, Memory {proc.memory_info().rss / 1024 / 1024:.1f}MB")
            
            # Try graceful termination first
            process.terminate()
            logger.debug("Sent SIGTERM to process, waiting 5 seconds for graceful shutdown")
            time.sleep(5)
            
            if process.poll() is None:
                logger.warning("Process did not terminate gracefully, sending SIGKILL")
                process.kill()
                logger.debug("Sent SIGKILL to process")
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")

def monitor_process_resources(process, interval=10):
    """Monitor and log resource usage of the process."""
    logger.debug(f"Starting resource monitor thread with interval {interval} seconds")
    try:
        proc = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                # Get CPU and memory usage
                cpu_percent = proc.cpu_percent(interval=1)
                memory_mb = proc.memory_info().rss / 1024 / 1024
                
                # Get child processes
                children = proc.children(recursive=True)
                child_count = len(children)
                
                # Log resource usage
                logger.debug(f"Process resources: CPU {cpu_percent}%, Memory {memory_mb:.1f}MB, Children: {child_count}")
                
                # Check for any zombie processes
                zombies = [p for p in children if p.status() == psutil.STATUS_ZOMBIE]
                if zombies:
                    logger.warning(f"Found {len(zombies)} zombie child processes")
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error monitoring process resources: {e}")
                time.sleep(interval)
    except Exception as e:
        logger.error(f"Failed to monitor process resources: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")

def check_port_in_use(port):
    """Check if a port is already in use."""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking port {port}: {e}")
        return False

def kill_processes_on_port(port):
    """Kill any processes using the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        logger.warning(f"Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                        psutil.Process(proc.info['pid']).terminate()
                        time.sleep(1)
                        if psutil.pid_exists(proc.info['pid']):
                            psutil.Process(proc.info['pid']).kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                continue
    except Exception as e:
        logger.error(f"Error killing processes on port {port}: {e}")

def heartbeat_checker(process, interval=60, max_no_output=300):
    """Check if the process is still producing output and responding."""
    global should_restart
    
    logger.debug(f"Starting heartbeat checker with interval {interval} seconds")
    last_log_size = 0
    last_output_time = time.time()
    log_file = "agi_system.log"
    
    while process.poll() is None:
        try:
            time.sleep(interval)
            
            # Check if log file has grown
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                if current_size > last_log_size:
                    last_log_size = current_size
                    last_output_time = time.time()
                    logger.debug("Heartbeat detected: log file has grown")
                else:
                    time_since_output = time.time() - last_output_time
                    logger.debug(f"No new log output for {time_since_output:.1f} seconds")
                    
                    # If no output for too long, consider the process hung
                    if time_since_output > max_no_output:
                        logger.warning(f"No output for {time_since_output:.1f} seconds, process may be hung")
                        
                        # Try to get stack trace if on Unix-like system
                        if os.name == 'posix':
                            try:
                                os.kill(process.pid, signal.SIGUSR1)
                                logger.debug("Sent SIGUSR1 to process to request stack trace")
                            except Exception as e:
                                logger.debug(f"Failed to send signal: {e}")
                        
                        # If we're in continuous mode, terminate the process so it can restart
                        logger.warning("Terminating hung process to allow restart")
                        process.terminate()
                        should_restart = True
                        return
            else:
                logger.warning(f"Log file {log_file} not found")
                
        except Exception as e:
            logger.error(f"Error in heartbeat checker: {e}")
            logger.debug(traceback.format_exc())

def can_restart():
    """Check if we should restart based on restart limits."""
    global restart_count, restart_timestamps
    
    # Remove timestamps older than the window
    current_time = time.time()
    restart_timestamps = [t for t in restart_timestamps if current_time - t <= restart_window]
    
    # Check if we've exceeded the maximum restarts in the window
    if len(restart_timestamps) >= max_restarts:
        logger.error(f"Exceeded maximum restarts ({max_restarts}) within {restart_window/3600} hour window")
        return False
    
    # Add current timestamp and increment count
    restart_timestamps.append(current_time)
    restart_count += 1
    
    logger.info(f"Restarting AGI system (restart {restart_count}, {len(restart_timestamps)} in current window)")
    return True

def run_agi_process(args):
    """Run the AGI system process with the specified arguments."""
    global should_restart
    
    # Check if memory server port is already in use
    if check_port_in_use(8000):
        logger.warning("Port 8000 is already in use. Attempting to kill existing processes.")
        kill_processes_on_port(8000)
        time.sleep(2)  # Give processes time to terminate
    
    # Prepare command with appropriate flags
    cmd = [sys.executable, "main.py", "--auto"]
    if args.debug or args.ddebug:
        cmd.append("--debug")
    if args.ddebug:
        cmd.append("--verbose")
        
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    # Start the AGI system process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=dict(os.environ)  # Copy current environment
    )
    
    logger.debug(f"Process started with PID: {process.pid}")
    
    # Set up timeout for process startup if not in continuous mode
    if args.timeout > 0 and not args.continuous:
        logger.debug(f"Setting up timeout monitor for {args.timeout} seconds")
        timeout_thread = threading.Thread(
            target=kill_process_after_timeout,
            args=(process, args.timeout),
            name="TimeoutMonitor"
        )
        timeout_thread.daemon = True
        timeout_thread.start()
    
    # Set up resource monitoring
    resource_thread = threading.Thread(
        target=monitor_process_resources,
        args=(process, args.monitor_interval),
        name="ResourceMonitor"
    )
    resource_thread.daemon = True
    resource_thread.start()
    
    # Set up heartbeat checker for continuous mode
    if args.continuous:
        heartbeat_thread = threading.Thread(
            target=heartbeat_checker,
            args=(process, 60, args.heartbeat_timeout),
            name="HeartbeatChecker"
        )
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    
    # Set up the duration timer if specified
    end_time = time.time() + args.duration if args.duration > 0 else None
    if end_time:
        logger.debug(f"Process will run until: {time.ctime(end_time)}")
    
    # Track when we last saw output
    last_output_time = time.time()
    
    # Monitor the output
    logger.debug("Starting output monitoring loop")
    while True:
        # Check if we've reached the duration limit
        if end_time and time.time() >= end_time:
            logger.info(f"Reached specified duration of {args.duration} seconds")
            should_restart = args.continuous  # Restart if in continuous mode
            break
        
        # Read output from the process
        output = process.stdout.readline()
        if output:
            print(output.strip())
            last_output_time = time.time()
            
            # Log specific patterns that might indicate issues
            if "error" in output.lower() or "exception" in output.lower():
                logger.warning(f"Possible error in output: {output.strip()}")
        
        # Read errors from the process
        error = process.stderr.readline()
        if error:
            logger.error(f"Process stderr: {error.strip()}")
            last_output_time = time.time()
            
            # Check for common error patterns
            if "import" in error.lower() and "error" in error.lower():
                logger.critical(f"Import error detected: {error.strip()}")
            elif "memory" in error.lower():
                logger.critical(f"Possible memory issue: {error.strip()}")
        
        # Check if the process has exited
        if process.poll() is not None:
            logger.info(f"AGI system process has exited with code {process.returncode}")
            should_restart = args.continuous  # Restart if in continuous mode
            break
        
        # Check for long periods without output (possible hang)
        time_since_output = time.time() - last_output_time
        if time_since_output > 60 and not args.continuous:  # No output for 1 minute (only log if not in continuous mode)
            logger.warning(f"No output for {time_since_output:.1f} seconds, process may be hanging")
            
            # Try to get stack trace if on Unix-like system
            if os.name == 'posix':
                try:
                    import signal
                    os.kill(process.pid, signal.SIGUSR1)
                    logger.debug("Sent SIGUSR1 to process to request stack trace")
                except Exception as e:
                    logger.debug(f"Failed to send signal: {e}")
        
        # Sleep briefly to avoid high CPU usage
        time.sleep(0.1)
    
    logger.debug("Exited monitoring loop, collecting remaining output")
    
    # Get any remaining output
    try:
        remaining_output, remaining_errors = process.communicate(timeout=5)
        if remaining_output:
            print(remaining_output.strip())
        if remaining_errors:
            logger.error(f"Final stderr output: {remaining_errors.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning("Timed out waiting for final output")
    
    # Terminate the process if it's still running
    if process.poll() is None:
        logger.info("Terminating AGI system process")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.debug(f"Process terminated with exit code: {process.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("AGI system process did not terminate gracefully, killing")
            process.kill()
            try:
                process.wait(timeout=5)
                logger.debug(f"Process killed with exit code: {process.returncode}")
            except subprocess.TimeoutExpired:
                logger.critical("Failed to kill process, it may be left running")
    
    return process.returncode

def main():
    global should_restart
    
    parser = argparse.ArgumentParser(description="Run AGI System in Autonomous Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--ddebug", action="store_true", help="Enable very verbose debug logging")
    parser.add_argument("--duration", type=int, default=0, help="Duration to run in seconds (0 = indefinitely)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for process startup in seconds (default: 300)")
    parser.add_argument("--skip-dependency-check", action="store_true", help="Skip dependency check")
    parser.add_argument("--monitor-interval", type=int, default=10, help="Interval for resource monitoring in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous 24/7 mode with automatic restart")
    parser.add_argument("--heartbeat-timeout", type=int, default=300, help="Maximum time without output before restarting (seconds)")
    parser.add_argument("--restart-delay", type=int, default=30, help="Delay between restarts in seconds")
    args = parser.parse_args()
    
    if args.debug or args.ddebug:
        logger.setLevel(logging.DEBUG)
        if args.ddebug:
            # Set even more verbose logging
            os.environ["PYTHONVERBOSE"] = "2"
            logger.debug("Extra verbose debugging enabled")
    
    logger.debug(f"Starting script with arguments: {args}")
    
    # Check dependencies unless skipped
    if not args.skip_dependency_check:
        logger.debug("Performing dependency check")
        if not check_dependencies():
            logger.error("Dependency check failed. Please install required packages manually.")
            return
    else:
        logger.debug("Dependency check skipped")
    
    # Print continuous mode banner if enabled
    if args.continuous:
        logger.info("=" * 80)
        logger.info("Starting AGI system in CONTINUOUS 24/7 MODE with automatic restart")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
    else:
        logger.info("Starting AGI system in autonomous mode")
    
    # Main execution loop with restart capability
    try:
        while True:
            try:
                # Run the AGI process
                exit_code = run_agi_process(args)
                
                # Check if we should restart
                if should_restart and args.continuous:
                    if can_restart():
                        logger.info(f"Waiting {args.restart_delay} seconds before restarting...")
                        time.sleep(args.restart_delay)
                        logger.info(f"Restarting AGI system...")
                        # Continue the loop to restart
                    else:
                        logger.error("Too many restart attempts, giving up")
                        break
                else:
                    # Normal exit
                    break
                    
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                should_restart = False
                break
            except Exception as e:
                logger.error(f"Error running AGI system: {e}")
                logger.debug(f"Detailed error: {traceback.format_exc()}")
                
                if args.continuous and can_restart():
                    logger.info(f"Waiting {args.restart_delay} seconds before restarting...")
                    time.sleep(args.restart_delay)
                    logger.info(f"Restarting AGI system after error...")
                    # Continue the loop to restart
                else:
                    break
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt in main loop, stopping")
    
    if args.continuous:
        logger.info("=" * 80)
        logger.info("AGI system continuous operation ended")
        logger.info(f"Total restarts: {restart_count}")
        logger.info("=" * 80)
    else:
        logger.info("Autonomous AGI run completed")

if __name__ == "__main__":
    main() 
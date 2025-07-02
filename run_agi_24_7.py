#!/usr/bin/env python3
"""
Run AGI System in 24/7 Continuous Mode
This is a simple wrapper script that runs the AGI system in continuous mode with optimal settings for 24/7 operation.
"""

import os
import sys
import subprocess
import logging
import time
import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agi_24_7.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AGI_24_7")

def print_banner():
    """Print a banner with the current time."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    AGI SYSTEM - 24/7 CONTINUOUS OPERATION                    ║
║                                                                              ║
║  Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                      ║
║                                                                              ║
║  Press Ctrl+C to stop                                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    logger.info("AGI 24/7 continuous operation started")

def main():
    parser = argparse.ArgumentParser(description="Run AGI System in 24/7 Continuous Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--heartbeat-timeout", type=int, default=600, 
                      help="Maximum time without output before restarting (seconds, default: 600)")
    parser.add_argument("--restart-delay", type=int, default=60, 
                      help="Delay between restarts in seconds (default: 60)")
    parser.add_argument("--monitor-interval", type=int, default=30, 
                      help="Interval for resource monitoring in seconds (default: 30)")
    args = parser.parse_args()
    
    print_banner()
    
    # Build the command to run the autonomous script in continuous mode
    cmd = [
        sys.executable, 
        "run_autonomous.py", 
        "--continuous",
        "--heartbeat-timeout", str(args.heartbeat_timeout),
        "--restart-delay", str(args.restart_delay),
        "--monitor-interval", str(args.monitor_interval),
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the process and wait for it to complete
        process = subprocess.Popen(cmd)
        process.wait()
        
        # If the process exits, log it
        logger.info(f"AGI system exited with code: {process.returncode}")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping AGI system")
        print("\nStopping AGI system. Please wait...")
        
        # Try to terminate the process gracefully
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
                logger.info("AGI system stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("AGI system did not stop gracefully, forcing termination")
                process.kill()
    
    except Exception as e:
        logger.error(f"Error running AGI system: {e}")
    
    finally:
        # Print final message
        end_time = datetime.datetime.now()
        runtime = end_time - datetime.datetime.strptime(
            logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None)), 
            '%Y-%m-%d %H:%M:%S'
        )
        
        print("\n" + "=" * 80)
        print(f"AGI 24/7 operation ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {runtime}")
        print("=" * 80)
        logger.info(f"AGI 24/7 operation ended. Total runtime: {runtime}")

if __name__ == "__main__":
    main() 
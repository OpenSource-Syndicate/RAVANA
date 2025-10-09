"""
Process Management for Snake Agent
This module provides functionality to safely shutdown and restart RAVANA processes
when implementing changes.
"""

import os
import signal
import psutil
import time
import subprocess
import sys
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessManager:
    """
    Manages RAVANA processes, allowing for safe shutdown and restart
    when implementing changes.
    """
    
    def __init__(self):
        self.ravana_process_name = "python"
        self.ravana_script_name = "main.py"
        self.ravana_processes = []
        
    def find_ravana_processes(self) -> List[psutil.Process]:
        """
        Find all running RAVANA processes.
        
        Returns:
            List of process objects that appear to be RAVANA
        """
        ravana_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                # Check if this process is related to RAVANA
                if self._is_ravana_process(proc):
                    ravana_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process disappeared or we don't have access, skip it
                continue
        
        self.ravana_processes = ravana_processes
        logger.info(f"Found {len(ravana_processes)} RAVANA processes")
        return ravana_processes
    
    def _is_ravana_process(self, proc: psutil.Process) -> bool:
        """
        Determine if a process is a RAVANA instance.
        
        Args:
            proc: Process to check
            
        Returns:
            True if the process appears to be RAVANA
        """
        try:
            # Check the command line and process name
            cmdline = proc.cmdline()
            
            # Check if main.py is in the command line
            if any(self.ravana_script_name in arg for arg in cmdline):
                return True
            
            # Check if the process name contains python and it's running main.py
            if "python" in proc.name().lower():
                # Look for main.py in the command line
                for arg in cmdline:
                    if self.ravana_script_name in arg:
                        return True
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        
        return False
    
    def shutdown_ravana_gracefully(self, timeout: int = 30) -> bool:
        """
        Attempt to shutdown RAVANA processes gracefully using SIGTERM.
        
        Args:
            timeout: Time to wait for graceful shutdown before force killing
            
        Returns:
            True if all processes were successfully stopped
        """
        ravana_processes = self.find_ravana_processes()
        
        if not ravana_processes:
            logger.info("No RAVANA processes found to shutdown")
            return True
        
        logger.info(f"Attempting to shutdown {len(ravana_processes)} RAVANA processes gracefully...")
        
        # First, send SIGTERM to all RAVANA processes
        for proc in ravana_processes:
            try:
                proc.terminate()  # Equivalent to SIGTERM on Unix, os.kill on Windows
                logger.info(f"Sent SIGTERM to process {proc.pid}")
            except psutil.NoSuchProcess:
                logger.info(f"Process {proc.pid} already terminated")
            except Exception as e:
                logger.warning(f"Could not terminate process {proc.pid}: {e}")
        
        # Wait for processes to terminate gracefully
        gone, alive = psutil.wait_procs(ravana_processes, timeout=timeout)
        
        if alive:
            logger.warning(f"{len(alive)} processes did not terminate gracefully, force killing...")
            
            # Force kill any remaining processes
            for proc in alive:
                try:
                    proc.kill()  # Equivalent to SIGKILL on Unix
                    logger.info(f"Force killed process {proc.pid}")
                except psutil.NoSuchProcess:
                    logger.info(f"Process {proc.pid} already terminated")
                except Exception as e:
                    logger.error(f"Could not force kill process {proc.pid}: {e}")
                    return False
        
        # Wait a bit more to ensure processes are gone
        time.sleep(1)
        
        # Verify all processes are gone
        remaining_processes = self.find_ravana_processes()
        if remaining_processes:
            logger.error(f"Failed to shutdown {len(remaining_processes)} RAVANA processes")
            return False
        
        logger.info("All RAVANA processes shut down successfully")
        return True
    
    def restart_ravana(self, script_args: Optional[List[str]] = None) -> Optional[subprocess.Popen]:
        """
        Restart RAVANA by starting a new process.
        
        Args:
            script_args: Optional arguments to pass to the script
            
        Returns:
            Process object if restart was successful, None otherwise
        """
        logger.info("Attempting to restart RAVANA...")
        
        # Find the main.py file in the project
        main_py_path = self._find_main_script()
        if not main_py_path:
            logger.error("Could not find main.py script")
            return None
        
        # Prepare command to run the script
        cmd = [sys.executable, str(main_py_path)]
        if script_args:
            cmd.extend(script_args)
        
        try:
            # Start the process
            logger.info(f"Starting RAVANA with command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            
            # Check if the process is still running
            if process.poll() is not None:
                # Process has already terminated, get the error
                _, stderr = process.communicate()
                logger.error(f"RAVANA failed to start: {stderr.decode()}")
                return None
            
            logger.info(f"RAVANA restarted successfully with PID {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to restart RAVANA: {e}")
            return None
    
    def _find_main_script(self) -> Optional[Path]:
        """
        Find the main.py script in the project.
        
        Returns:
            Path to main.py if found, None otherwise
        """
        # Look for main.py in the current directory and common locations
        possible_paths = [
            Path("main.py"),
            Path(".") / "main.py",
            Path("..") / "main.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.resolve()
        
        # Look in project root
        project_root = Path(__file__).parent.parent  # Go up from core/ to project root
        main_path = project_root / "main.py"
        if main_path.exists():
            return main_path.resolve()
        
        return None
    
    def is_ravana_running(self) -> bool:
        """
        Check if RAVANA is currently running.
        
        Returns:
            True if RAVANA processes are detected
        """
        ravana_processes = self.find_ravana_processes()
        return len(ravana_processes) > 0
    
    def wait_for_startup(self, timeout: int = 60, check_interval: int = 1) -> bool:
        """
        Wait for RAVANA to fully start up by checking for expected processes.
        
        Args:
            timeout: Maximum time to wait for startup
            check_interval: Interval between checks
            
        Returns:
            True if RAVANA appears to be running after timeout period
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ravana_running():
                logger.info("RAVANA appears to be running")
                # Wait a bit more to ensure it's fully started
                time.sleep(3)
                return True
            
            time.sleep(check_interval)
        
        logger.error(f"RAVANA did not appear to start within {timeout} seconds")
        return False


class ProcessController:
    """
    Higher-level controller that coordinates process management with change implementation.
    """
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.shutdown_timeout = 30
        self.startup_timeout = 60
    
    def shutdown_and_kill_ravana(self) -> bool:
        """
        Shutdown RAVANA processes. If graceful shutdown fails, force kill.
        
        Returns:
            True if all processes were successfully terminated
        """
        logger.info("Shutting down RAVANA before applying changes...")
        
        # Try graceful shutdown first
        success = self.process_manager.shutdown_ravana_gracefully(timeout=self.shutdown_timeout)
        
        if not success:
            logger.warning("Graceful shutdown failed, force killing processes...")
            # Find and force-kill any remaining processes
            processes = self.process_manager.find_ravana_processes()
            for proc in processes:
                try:
                    proc.kill()
                    logger.info(f"Force killed process {proc.pid}")
                except Exception as e:
                    logger.error(f"Could not force kill process {proc.pid}: {e}")
                    return False
        
        # Verify processes are gone
        if self.process_manager.is_ravana_running():
            logger.error("Some RAVANA processes remain after shutdown attempt")
            return False
        
        logger.info("All RAVANA processes terminated successfully")
        return True
    
    def start_ravana_with_args(self, args: Optional[List[str]] = None) -> bool:
        """
        Start RAVANA with specified arguments and wait for it to start.
        
        Args:
            args: Arguments to pass to the RAVANA script
            
        Returns:
            True if RAVANA started successfully
        """
        logger.info("Starting RAVANA after applying changes...")
        
        process = self.process_manager.restart_ravana(script_args=args)
        if not process:
            logger.error("Failed to start RAVANA")
            return False
        
        # Wait for RAVANA to fully start
        started = self.process_manager.wait_for_startup(timeout=self.startup_timeout)
        if not started:
            logger.error("RAVANA did not start properly")
            return False
        
        logger.info("RAVANA started successfully after changes")
        return True
    
    def restart_ravana(self, args: Optional[List[str]] = None) -> bool:
        """
        Complete restart cycle: shutdown existing processes and start new ones.
        
        Args:
            args: Arguments to pass to the RAVANA script
            
        Returns:
            True if restart was successful
        """
        # Shutdown existing processes
        if not self.shutdown_and_kill_ravana():
            logger.error("Could not shutdown existing RAVANA processes")
            return False
        
        # Start new processes
        return self.start_ravana_with_args(args)


# Example usage:
if __name__ == "__main__":
    controller = ProcessController()
    
    # Check if RAVANA is running
    if controller.process_manager.is_ravana_running():
        print("RAVANA is currently running")
        
        # Shutdown RAVANA
        success = controller.shutdown_and_kill_ravana()
        if success:
            print("RAVANA shut down successfully")
        else:
            print("Failed to shutdown RAVANA")
    else:
        print("RAVANA is not currently running")
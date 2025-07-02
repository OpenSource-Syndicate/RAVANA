#!/usr/bin/env python3
"""
Check AGI System Status
This script checks the status of the running AGI system, showing uptime, memory usage, and recent log entries.
"""

import os
import sys
import time
import datetime
import argparse
import psutil
import subprocess
import re
from tabulate import tabulate

def find_agi_processes():
    """Find all running AGI system processes."""
    agi_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and ('main.py' in ' '.join(cmdline) and '--auto' in cmdline):
                # This is an AGI process
                create_time = datetime.datetime.fromtimestamp(proc.info['create_time'])
                uptime = datetime.datetime.now() - create_time
                
                # Get memory usage
                process = psutil.Process(proc.pid)
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=0.5)
                
                # Get child processes
                children = process.children(recursive=True)
                child_count = len(children)
                
                agi_processes.append({
                    'pid': proc.pid,
                    'name': 'AGI System',
                    'uptime': str(uptime).split('.')[0],  # Remove microseconds
                    'memory_mb': f"{memory_mb:.1f}",
                    'cpu_percent': f"{cpu_percent:.1f}",
                    'child_count': child_count,
                    'process': process
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return agi_processes

def get_recent_logs(log_file='agi_system.log', lines=10):
    """Get recent log entries from the AGI system log file."""
    if not os.path.exists(log_file):
        return ["Log file not found: " + log_file]
    
    try:
        # Use tail command on Unix-like systems
        if os.name == 'posix':
            result = subprocess.run(['tail', '-n', str(lines), log_file], 
                                   capture_output=True, text=True)
            return result.stdout.splitlines()
        else:
            # On Windows, read the file and get the last lines
            with open(log_file, 'r', encoding='utf-8') as f:
                return f.readlines()[-lines:]
    except Exception as e:
        return [f"Error reading log file: {e}"]

def format_memory_usage():
    """Format system memory usage information."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    memory_info = [
        ["Total Memory", f"{memory.total / (1024**3):.2f} GB"],
        ["Available Memory", f"{memory.available / (1024**3):.2f} GB"],
        ["Used Memory", f"{memory.used / (1024**3):.2f} GB ({memory.percent}%)"],
        ["Swap Total", f"{swap.total / (1024**3):.2f} GB"],
        ["Swap Used", f"{swap.used / (1024**3):.2f} GB ({swap.percent}%)"]
    ]
    
    return tabulate(memory_info, tablefmt="simple")

def format_disk_usage():
    """Format disk usage information for the current disk."""
    disk = psutil.disk_usage('/')
    
    disk_info = [
        ["Total Disk Space", f"{disk.total / (1024**3):.2f} GB"],
        ["Used Disk Space", f"{disk.used / (1024**3):.2f} GB ({disk.percent}%)"],
        ["Free Disk Space", f"{disk.free / (1024**3):.2f} GB"]
    ]
    
    return tabulate(disk_info, tablefmt="simple")

def print_status_banner(agi_processes):
    """Print a status banner with information about running AGI processes."""
    now = datetime.datetime.now()
    
    print("\n" + "=" * 80)
    print(f"AGI SYSTEM STATUS - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not agi_processes:
        print("\nNo AGI System processes found running.")
    else:
        # Format process information as a table
        headers = ["PID", "Name", "Uptime", "Memory (MB)", "CPU %", "Children"]
        table_data = [
            [p['pid'], p['name'], p['uptime'], p['memory_mb'], p['cpu_percent'], p['child_count']] 
            for p in agi_processes
        ]
        print("\nRunning AGI Processes:")
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    print("\nSystem Resources:")
    print(format_memory_usage())
    print("\nDisk Usage:")
    print(format_disk_usage())

def check_port_status(port=8000):
    """Check if the specified port is in use and by which process."""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    return f"Port {port} is in use by process {conn.pid} ({process.name()})"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return f"Port {port} is in use by process {conn.pid}"
        return f"Port {port} is not in use"
    except Exception as e:
        return f"Error checking port {port}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Check AGI System Status")
    parser.add_argument("--log-file", default="agi_system.log", help="Path to the AGI system log file")
    parser.add_argument("--lines", type=int, default=10, help="Number of log lines to show")
    parser.add_argument("--watch", action="store_true", help="Watch mode - continuously update status")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds for watch mode")
    args = parser.parse_args()
    
    try:
        if args.watch:
            print("Press Ctrl+C to exit watch mode")
            while True:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Find AGI processes
                agi_processes = find_agi_processes()
                
                # Print status banner
                print_status_banner(agi_processes)
                
                # Check memory server port status
                print("\nMemory Server Status:")
                print(check_port_status(8000))
                
                # Show recent logs
                print("\nRecent Log Entries:")
                for line in get_recent_logs(args.log_file, args.lines):
                    print(line.strip())
                
                # Wait for next update
                time.sleep(args.interval)
        else:
            # Find AGI processes
            agi_processes = find_agi_processes()
            
            # Print status banner
            print_status_banner(agi_processes)
            
            # Check memory server port status
            print("\nMemory Server Status:")
            print(check_port_status(8000))
            
            # Show recent logs
            print("\nRecent Log Entries:")
            for line in get_recent_logs(args.log_file, args.lines):
                print(line.strip())
    
    except KeyboardInterrupt:
        print("\nStatus check interrupted.")
    except Exception as e:
        print(f"\nError checking AGI system status: {e}")

if __name__ == "__main__":
    main() 
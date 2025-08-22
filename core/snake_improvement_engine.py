"""
Snake Continuous Improvement Engine

This module implements a continuous improvement engine that applies
validated improvements to the RAVANA codebase safely and automatically.
"""

import asyncio
import threading
import queue
import time
import uuid
import shutil
import os
import git
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum

from core.snake_data_models import (
    ImprovementProposal, TaskPriority, ImprovementRecord, SnakeAgentConfiguration
)
from core.snake_log_manager import SnakeLogManager


class ImprovementStatus(Enum):
    """Status of improvement application"""
    PENDING = "pending"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class ImprovementApplication:
    """Represents an improvement being applied"""
    proposal: ImprovementProposal
    status: ImprovementStatus
    backup_info: Optional[Dict[str, Any]]
    application_log: List[str]
    start_time: Optional[datetime]
    completion_time: Optional[datetime]
    worker_id: str
    rollback_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "proposal_id": self.proposal.proposal_id,
            "status": self.status.value,
            "backup_info": self.backup_info,
            "application_log": self.application_log,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "worker_id": self.worker_id,
            "rollback_available": self.rollback_available
        }


class SafetyValidator:
    """Validates improvements for safety before application"""
    
    def __init__(self, config: SnakeAgentConfiguration):
        self.config = config
        self.critical_files = {
            'main.py', 'core/system.py', 'core/config.py',
            'database/engine.py', 'core/shutdown_coordinator.py'
        }
        self.protected_patterns = [
            r'DATABASE_URL',
            r'SECRET_KEY',
            r'API_KEY',
            r'PASSWORD',
            r'shutdown.*coordinator',
            r'graceful.*shutdown'
        ]
    
    def validate_improvement(self, proposal: ImprovementProposal) -> tuple[bool, List[str]]:
        """Validate improvement proposal for safety"""
        issues = []
        
        # Check if file is critical
        if self._is_critical_file(proposal.file_path):
            issues.append(f"Critical file modification: {proposal.file_path}")
        
        # Check for protected patterns
        protected_violations = self._check_protected_patterns(proposal.proposed_code)
        issues.extend(protected_violations)
        
        # Check improvement type safety
        if proposal.improvement_type == "security" and proposal.safety_score < 0.9:
            issues.append("Security improvement has low safety score")
        
        # Check code quality
        quality_issues = self._validate_code_quality(proposal.proposed_code)
        issues.extend(quality_issues)
        
        # Check file size
        if len(proposal.proposed_code) > self.config.snake_max_file_size:
            issues.append("Proposed code exceeds maximum file size")
        
        return len(issues) == 0, issues
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if file is critical"""
        return any(critical in file_path for critical in self.critical_files)
    
    def _check_protected_patterns(self, code: str) -> List[str]:
        """Check for protected patterns in code"""
        import re
        violations = []
        
        for pattern in self.protected_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Protected pattern detected: {pattern}")
        
        return violations
    
    def _validate_code_quality(self, code: str) -> List[str]:
        """Validate basic code quality"""
        issues = []
        
        # Check for basic syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
        
        # Check for dangerous operations
        dangerous_ops = ['eval(', 'exec(', 'subprocess.call', '__import__']
        for op in dangerous_ops:
            if op in code:
                issues.append(f"Dangerous operation detected: {op}")
        
        return issues


class BackupManager:
    """Manages backups for safe improvement application"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.backup_dir = self.workspace_path / ".snake_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize git repo if not exists
        self.git_repo = None
        try:
            self.git_repo = git.Repo(self.workspace_path)
        except git.InvalidGitRepositoryError:
            # Not a git repo, that's okay
            pass
    
    def create_backup(self, improvement_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """Create backup before applying improvement"""
        backup_info = {
            "improvement_id": improvement_id,
            "timestamp": datetime.now().isoformat(),
            "files": {},
            "git_commit": None
        }
        
        # Create improvement-specific backup directory
        improvement_backup_dir = self.backup_dir / improvement_id
        improvement_backup_dir.mkdir(exist_ok=True)
        
        # Backup files
        for file_path in file_paths:
            full_path = self.workspace_path / file_path
            if full_path.exists():
                # Copy file to backup
                backup_file_path = improvement_backup_dir / file_path.replace('/', '_')
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, backup_file_path)
                
                backup_info["files"][file_path] = {
                    "backup_path": str(backup_file_path),
                    "original_size": full_path.stat().st_size,
                    "backup_time": datetime.now().isoformat()
                }
        
        # Create git commit if repo available
        if self.git_repo:
            try:
                commit = self.git_repo.head.commit
                backup_info["git_commit"] = {
                    "sha": commit.hexsha,
                    "message": commit.message,
                    "timestamp": commit.committed_datetime.isoformat()
                }
            except Exception:
                pass  # Git operations are optional
        
        return backup_info
    
    def restore_backup(self, backup_info: Dict[str, Any]) -> bool:
        """Restore files from backup"""
        try:
            for file_path, file_info in backup_info.get("files", {}).items():
                backup_path = Path(file_info["backup_path"])
                original_path = self.workspace_path / file_path
                
                if backup_path.exists():
                    # Restore file
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original_path)
            
            return True
            
        except Exception as e:
            return False
    
    def cleanup_backup(self, improvement_id: str):
        """Clean up backup files"""
        try:
            improvement_backup_dir = self.backup_dir / improvement_id
            if improvement_backup_dir.exists():
                shutil.rmtree(improvement_backup_dir)
        except Exception:
            pass  # Best effort cleanup


class ImprovementApplicator:
    """Applies improvements to files safely"""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
    
    def apply_improvement(self, proposal: ImprovementProposal) -> tuple[bool, List[str]]:
        """Apply improvement to file"""
        log = []
        
        try:
            log.append(f"Starting application of improvement {proposal.proposal_id}")
            
            # Apply changes based on improvement type
            if proposal.improvement_type in ["bug_fix", "performance", "maintainability"]:
                success = self._apply_code_replacement(proposal, log)
            elif proposal.improvement_type == "security":
                success = self._apply_security_improvement(proposal, log)
            else:
                success = self._apply_generic_improvement(proposal, log)
            
            log.append(f"Improvement application {'succeeded' if success else 'failed'}")
            return success, log
            
        except Exception as e:
            log.append(f"Error applying improvement: {str(e)}")
            return False, log
    
    def _apply_code_replacement(self, proposal: ImprovementProposal, log: List[str]) -> bool:
        """Apply code replacement improvement"""
        try:
            file_path = Path(proposal.file_path)
            
            # Read current file
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            log.append(f"Read current file: {len(current_content)} characters")
            
            # Replace original code with proposed code
            if proposal.original_code in current_content:
                new_content = current_content.replace(
                    proposal.original_code,
                    proposal.proposed_code,
                    1  # Replace only first occurrence
                )
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                log.append("Code replacement successful")
                return True
            else:
                log.append("Original code not found in file")
                return False
                
        except Exception as e:
            log.append(f"Code replacement error: {str(e)}")
            return False
    
    def _apply_security_improvement(self, proposal: ImprovementProposal, log: List[str]) -> bool:
        """Apply security-focused improvement"""
        log.append("Applying security improvement with extra validation")
        
        # Security improvements get extra validation
        if "password" in proposal.proposed_code.lower() or "secret" in proposal.proposed_code.lower():
            log.append("Security improvement contains sensitive terms - requires manual review")
            return False
        
        # Apply using standard code replacement
        return self._apply_code_replacement(proposal, log)
    
    def _apply_generic_improvement(self, proposal: ImprovementProposal, log: List[str]) -> bool:
        """Apply generic improvement"""
        log.append("Applying generic improvement")
        return self._apply_code_replacement(proposal, log)


class ContinuousImprovementEngine:
    """Continuous improvement engine for RAVANA codebase"""
    
    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager
        
        # Core components
        self.safety_validator = SafetyValidator(config)
        self.backup_manager = BackupManager(os.getcwd())
        self.applicator = ImprovementApplicator(self.backup_manager)
        
        # Task processing
        self.improvement_queue = queue.PriorityQueue(maxsize=config.max_queue_size)
        self.active_applications: Dict[str, ImprovementApplication] = {}
        
        # Worker threads
        self.worker_threads: List[threading.Thread] = []
        self.num_workers = 2  # Conservative number for file modifications
        
        # Control
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Metrics
        self.improvements_processed = 0
        self.improvements_applied = 0
        self.improvements_rejected = 0
        self.improvements_failed = 0
        
        # Approval system
        self.auto_approve_types = {"style", "performance"}
        self.require_approval_types = {"security", "architecture"}
        self.pending_approvals: Dict[str, ImprovementProposal] = {}
        
        # Callbacks
        self.improvement_applied_callback: Optional[callable] = None
        self.approval_required_callback: Optional[callable] = None
    
    async def initialize(self) -> bool:
        """Initialize the improvement engine"""
        try:
            await self.log_manager.log_system_event(
                "improvement_engine_init",
                {"num_workers": self.num_workers},
                worker_id="improvement_engine"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "improvement_engine_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="improvement_engine"
            )
            return False
    
    async def start_workers(self) -> bool:
        """Start improvement worker threads"""
        try:
            if self.running:
                return True
            
            self.running = True
            self.shutdown_event.clear()
            
            # Start worker threads
            for i in range(self.num_workers):
                worker_id = f"improvement_worker_{i}"
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=f"Snake-ImprovementWorker-{i}",
                    daemon=True
                )
                thread.start()
                self.worker_threads.append(thread)
            
            await self.log_manager.log_system_event(
                "improvement_engine_started",
                {"workers_started": len(self.worker_threads)},
                worker_id="improvement_engine"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "improvement_engine_start_failed",
                {"error": str(e)},
                level="error",
                worker_id="improvement_engine"
            )
            return False
    
    def _worker_loop(self, worker_id: str):
        """Main loop for improvement worker"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get improvement from queue
                try:
                    priority, proposal = self.improvement_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process improvement
                self._process_improvement(proposal, worker_id)
                self.improvement_queue.task_done()
                
            except Exception as e:
                # Log worker error
                asyncio.create_task(self.log_manager.log_system_event(
                    "improvement_worker_error",
                    {"worker_id": worker_id, "error": str(e)},
                    level="error",
                    worker_id="improvement_engine"
                ))
                time.sleep(1.0)
    
    def _process_improvement(self, proposal: ImprovementProposal, worker_id: str):
        """Process a single improvement proposal"""
        try:
            application = ImprovementApplication(
                proposal=proposal,
                status=ImprovementStatus.PENDING,
                backup_info=None,
                application_log=[],
                start_time=datetime.now(),
                completion_time=None,
                worker_id=worker_id
            )
            
            self.active_applications[proposal.proposal_id] = application
            application.application_log.append(f"Started processing by {worker_id}")
            
            # Validate safety
            is_safe, safety_issues = self.safety_validator.validate_improvement(proposal)
            if not is_safe:
                application.status = ImprovementStatus.REJECTED
                application.application_log.extend([f"Safety issue: {issue}" for issue in safety_issues])
                self._complete_application(application)
                self.improvements_rejected += 1
                return
            
            application.application_log.append("Safety validation passed")
            
            # Check approval requirements
            if proposal.improvement_type in self.require_approval_types:
                if not self._is_approved(proposal):
                    application.status = ImprovementStatus.PENDING
                    application.application_log.append("Waiting for approval")
                    self.pending_approvals[proposal.proposal_id] = proposal
                    
                    # Call approval callback
                    if self.approval_required_callback:
                        try:
                            self.approval_required_callback(proposal)
                        except Exception:
                            pass
                    
                    return
            
            # Auto-approve certain types
            if proposal.improvement_type in self.auto_approve_types:
                application.status = ImprovementStatus.APPROVED
                application.application_log.append("Auto-approved")
            
            # Apply improvement
            self._apply_improvement(application)
            
        except Exception as e:
            if proposal.proposal_id in self.active_applications:
                app = self.active_applications[proposal.proposal_id]
                app.application_log.append(f"Processing error: {str(e)}")
                app.status = ImprovementStatus.FAILED
                self._complete_application(app)
            
            self.improvements_failed += 1
    
    def _apply_improvement(self, application: ImprovementApplication):
        """Apply the improvement"""
        try:
            proposal = application.proposal
            application.status = ImprovementStatus.APPLYING
            application.application_log.append("Starting improvement application")
            
            # Create backup
            backup_info = self.backup_manager.create_backup(
                proposal.proposal_id,
                [proposal.file_path]
            )
            application.backup_info = backup_info
            application.rollback_available = True
            application.application_log.append("Backup created")
            
            # Apply improvement
            success, apply_log = self.applicator.apply_improvement(proposal)
            application.application_log.extend(apply_log)
            
            if success:
                application.status = ImprovementStatus.APPLIED
                application.application_log.append("Improvement applied successfully")
                self.improvements_applied += 1
                
                # Call success callback
                if self.improvement_applied_callback:
                    try:
                        asyncio.create_task(self.improvement_applied_callback(application))
                    except Exception:
                        pass
            else:
                application.status = ImprovementStatus.FAILED
                application.application_log.append("Improvement application failed")
                
                # Restore backup on failure
                if self.backup_manager.restore_backup(backup_info):
                    application.application_log.append("Backup restored")
                else:
                    application.application_log.append("Backup restoration failed")
                
                self.improvements_failed += 1
            
            self._complete_application(application)
            
        except Exception as e:
            application.status = ImprovementStatus.FAILED
            application.application_log.append(f"Application error: {str(e)}")
            self._complete_application(application)
            self.improvements_failed += 1
    
    def _is_approved(self, proposal: ImprovementProposal) -> bool:
        """Check if proposal is approved (placeholder for approval system)"""
        # For now, auto-approve low-risk improvements
        return proposal.safety_score > 0.8 and proposal.confidence_score > 0.7
    
    def _complete_application(self, application: ImprovementApplication):
        """Complete improvement application"""
        application.completion_time = datetime.now()
        self.improvements_processed += 1
        
        # Log improvement completion
        asyncio.create_task(self.log_manager.log_improvement(
            ImprovementRecord(
                id=application.proposal.proposal_id,
                type=application.proposal.improvement_type,
                description=application.proposal.description,
                file_path=application.proposal.file_path,
                before_state={"original_code": application.proposal.original_code},
                after_state={"proposed_code": application.proposal.proposed_code},
                impact_score=application.proposal.impact_assessment.get("impact_score", 0.5),
                safety_score=application.proposal.safety_score,
                timestamp=application.completion_time,
                worker_id=application.worker_id,
                status=application.status.value
            )
        ))
        
        # Cleanup backup after successful application
        if application.status == ImprovementStatus.APPLIED:
            # Keep backup for a while, then clean up
            # For now, we'll keep it
            pass
    
    def queue_improvement(self, proposal: ImprovementProposal) -> bool:
        """Queue an improvement proposal"""
        try:
            # Priority based on improvement type and safety score
            priority = self._calculate_priority(proposal)
            
            self.improvement_queue.put_nowait((priority, proposal))
            
            asyncio.create_task(self.log_manager.log_system_event(
                "improvement_queued",
                {
                    "proposal_id": proposal.proposal_id,
                    "improvement_type": proposal.improvement_type,
                    "priority": priority
                },
                worker_id="improvement_engine"
            ))
            
            return True
            
        except queue.Full:
            asyncio.create_task(self.log_manager.log_system_event(
                "improvement_queue_full",
                {"proposal_id": proposal.proposal_id},
                level="warning",
                worker_id="improvement_engine"
            ))
            return False
    
    def _calculate_priority(self, proposal: ImprovementProposal) -> int:
        """Calculate priority for improvement (lower number = higher priority)"""
        base_priority = {
            "security": 1,
            "bug_fix": 2,
            "performance": 3,
            "maintainability": 4,
            "style": 5
        }.get(proposal.improvement_type, 6)
        
        # Adjust based on safety and confidence scores
        if proposal.safety_score > 0.9 and proposal.confidence_score > 0.8:
            base_priority -= 1
        
        return max(1, base_priority)
    
    def approve_improvement(self, proposal_id: str) -> bool:
        """Manually approve a pending improvement"""
        if proposal_id in self.pending_approvals:
            proposal = self.pending_approvals.pop(proposal_id)
            
            # Re-queue with approval
            priority = self._calculate_priority(proposal) - 1  # Higher priority for approved
            self.improvement_queue.put_nowait((priority, proposal))
            
            return True
        
        return False
    
    def reject_improvement(self, proposal_id: str, reason: str = "") -> bool:
        """Manually reject a pending improvement"""
        if proposal_id in self.pending_approvals:
            proposal = self.pending_approvals.pop(proposal_id)
            
            # Log rejection
            asyncio.create_task(self.log_manager.log_system_event(
                "improvement_rejected",
                {
                    "proposal_id": proposal_id,
                    "reason": reason
                },
                worker_id="improvement_engine"
            ))
            
            self.improvements_rejected += 1
            return True
        
        return False
    
    def set_callbacks(self, 
                     improvement_applied_callback: Optional[callable] = None,
                     approval_required_callback: Optional[callable] = None):
        """Set callbacks for improvement events"""
        self.improvement_applied_callback = improvement_applied_callback
        self.approval_required_callback = approval_required_callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get improvement engine status"""
        return {
            "running": self.running,
            "queue_size": self.improvement_queue.qsize(),
            "active_applications": len(self.active_applications),
            "pending_approvals": len(self.pending_approvals),
            "improvements_processed": self.improvements_processed,
            "improvements_applied": self.improvements_applied,
            "improvements_rejected": self.improvements_rejected,
            "improvements_failed": self.improvements_failed,
            "success_rate": (
                self.improvements_applied / max(1, self.improvements_processed)
            )
        }
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown the improvement engine"""
        try:
            await self.log_manager.log_system_event(
                "improvement_engine_shutdown",
                {"improvements_processed": self.improvements_processed},
                worker_id="improvement_engine"
            )
            
            self.running = False
            self.shutdown_event.set()
            
            # Wait for workers to finish
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=timeout/len(self.worker_threads))
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "improvement_engine_shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="improvement_engine"
            )
            return False
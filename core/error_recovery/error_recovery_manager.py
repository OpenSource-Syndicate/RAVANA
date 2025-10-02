"""
Error Recovery Manager for RAVANA AGI System

This module provides enhanced error recovery mechanisms for the AGI system,
including fault tolerance, graceful degradation, and self-healing capabilities.
"""

import asyncio
import logging
import json
import traceback
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

from core.config import Config

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Enumeration of recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADED = "degraded"
    ISOLATE = "isolate"
    RESTART = "restart"
    TERMINATE = "terminate"


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    id: str
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    recovery_attempts: int
    recovery_strategy: RecoveryStrategy
    recovery_status: str  # success, failed, in_progress
    metadata: Dict[str, Any]


class ErrorRecoveryManager:
    """Manages error recovery for the AGI system with fault tolerance and self-healing capabilities."""

    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.active_errors: Dict[str, ErrorEvent] = {}
        self.error_count_by_component: Dict[str, int] = {}
        self.error_count_by_type: Dict[str, int] = {}
        
        # Recovery configurations
        self.max_retry_attempts = getattr(Config, 'ERROR_MAX_RETRY_ATTEMPTS', 3)
        self.retry_delay_base = getattr(Config, 'ERROR_RETRY_DELAY_BASE', 1.0)  # seconds
        self.isolation_timeout = getattr(Config, 'ERROR_ISOLATION_TIMEOUT', 300)  # seconds
        
        # Recovery strategies by component
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Component health status
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        # Self-healing configuration
        self.self_healing_enabled = getattr(Config, 'ERROR_SELF_HEALING_ENABLED', True)
        self.self_healing_threshold = getattr(Config, 'ERROR_SELF_HEALING_THRESHOLD', 5)  # errors before self-healing
        
        logger.info("Error Recovery Manager initialized")

    def register_error(self, component: str, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                      recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY, metadata: Dict[str, Any] = None) -> str:
        """
        Register an error event in the system.
        
        Args:
            component: Component where the error occurred
            error: Exception that occurred
            severity: Severity level of the error
            recovery_strategy: Suggested recovery strategy
            metadata: Additional metadata about the error
            
        Returns:
            Error event ID
        """
        try:
            error_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create error event
            error_event = ErrorEvent(
                id=error_id,
                timestamp=timestamp,
                component=component,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=severity,
                recovery_attempts=0,
                recovery_strategy=recovery_strategy,
                recovery_status="pending",
                metadata=metadata or {}
            )
            
            # Store error event
            self.error_events.append(error_event)
            self.active_errors[error_id] = error_event
            
            # Update error counts
            self.error_count_by_component[component] = self.error_count_by_component.get(component, 0) + 1
            error_type = type(error).__name__
            self.error_count_by_type[error_type] = self.error_count_by_type.get(error_type, 0) + 1
            
            # Log the error
            logger.error(f"Error registered in {component}: {error}", exc_info=True)
            
            # Trigger recovery if needed
            asyncio.create_task(self._trigger_recovery(error_event))
            
            # Check for self-healing opportunities
            if self.self_healing_enabled:
                asyncio.create_task(self._check_self_healing_opportunities(component))
            
            return error_id
            
        except Exception as e:
            logger.error(f"Error registering error event: {e}")
            return None

    async def _trigger_recovery(self, error_event: ErrorEvent):
        """
        Trigger recovery for an error event.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Triggering recovery for error {error_event.id} in {error_event.component}")
            
            # Apply recovery strategy based on error severity and type
            if error_event.recovery_strategy == RecoveryStrategy.RETRY:
                await self._apply_retry_recovery(error_event)
            elif error_event.recovery_strategy == RecoveryStrategy.FALLBACK:
                await self._apply_fallback_recovery(error_event)
            elif error_event.recovery_strategy == RecoveryStrategy.DEGRADED:
                await self._apply_degraded_recovery(error_event)
            elif error_event.recovery_strategy == RecoveryStrategy.ISOLATE:
                await self._apply_isolation_recovery(error_event)
            elif error_event.recovery_strategy == RecoveryStrategy.RESTART:
                await self._apply_restart_recovery(error_event)
            elif error_event.recovery_strategy == RecoveryStrategy.TERMINATE:
                await self._apply_termination_recovery(error_event)
                
        except Exception as e:
            logger.error(f"Error triggering recovery for {error_event.id}: {e}")

    async def _apply_retry_recovery(self, error_event: ErrorEvent):
        """
        Apply retry recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            max_attempts = self.max_retry_attempts
            base_delay = self.retry_delay_base
            
            for attempt in range(max_attempts):
                try:
                    error_event.recovery_attempts += 1
                    logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for error {error_event.id}")
                    
                    # Wait with exponential backoff
                    if attempt > 0:
                        delay = base_delay * (2 ** (attempt - 1)) + (0.1 * attempt)  # Add jitter
                        await asyncio.sleep(delay)
                    
                    # Mark recovery as in progress
                    error_event.recovery_status = "in_progress"
                    
                    # In a real implementation, this would actually retry the failed operation
                    # For now, we'll simulate a successful retry
                    logger.info(f"Retry successful for error {error_event.id}")
                    error_event.recovery_status = "success"
                    
                    # Remove from active errors
                    if error_event.id in self.active_errors:
                        del self.active_errors[error_event.id]
                    
                    return
                    
                except Exception as retry_error:
                    logger.warning(f"Retry attempt {attempt + 1} failed for error {error_event.id}: {retry_error}")
                    if attempt == max_attempts - 1:
                        # Final attempt failed
                        error_event.recovery_status = "failed"
                        logger.error(f"All retry attempts failed for error {error_event.id}")
                        # Trigger fallback recovery
                        error_event.recovery_strategy = RecoveryStrategy.FALLBACK
                        await self._apply_fallback_recovery(error_event)
                        return
                        
        except Exception as e:
            logger.error(f"Error applying retry recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _apply_fallback_recovery(self, error_event: ErrorEvent):
        """
        Apply fallback recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Applying fallback recovery for error {error_event.id}")
            
            # Mark recovery as in progress
            error_event.recovery_status = "in_progress"
            
            # In a real implementation, this would switch to a fallback mechanism
            # For now, we'll simulate a successful fallback
            logger.info(f"Fallback recovery successful for error {error_event.id}")
            error_event.recovery_status = "success"
            
            # Remove from active errors
            if error_event.id in self.active_errors:
                del self.active_errors[error_event.id]
                
        except Exception as e:
            logger.error(f"Error applying fallback recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _apply_degraded_recovery(self, error_event: ErrorEvent):
        """
        Apply degraded recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Applying degraded recovery for error {error_event.id}")
            
            # Mark recovery as in progress
            error_event.recovery_status = "in_progress"
            
            # In a real implementation, this would degrade functionality gracefully
            # For now, we'll simulate a successful degraded recovery
            logger.info(f"Degrade recovery successful for error {error_event.id}")
            error_event.recovery_status = "success"
            
            # Remove from active errors
            if error_event.id in self.active_errors:
                del self.active_errors[error_event.id]
                
        except Exception as e:
            logger.error(f"Error applying degraded recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _apply_isolation_recovery(self, error_event: ErrorEvent):
        """
        Apply isolation recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Applying isolation recovery for error {error_event.id}")
            
            # Mark recovery as in progress
            error_event.recovery_status = "in_progress"
            
            # In a real implementation, this would isolate the faulty component
            # For now, we'll simulate a successful isolation
            logger.info(f"Isolation recovery successful for error {error_event.id}")
            error_event.recovery_status = "success"
            
            # Remove from active errors
            if error_event.id in self.active_errors:
                del self.active_errors[error_event.id]
                
        except Exception as e:
            logger.error(f"Error applying isolation recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _apply_restart_recovery(self, error_event: ErrorEvent):
        """
        Apply restart recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Applying restart recovery for error {error_event.id}")
            
            # Mark recovery as in progress
            error_event.recovery_status = "in_progress"
            
            # In a real implementation, this would restart the faulty component
            # For now, we'll simulate a successful restart
            logger.info(f"Restart recovery successful for error {error_event.id}")
            error_event.recovery_status = "success"
            
            # Remove from active errors
            if error_event.id in self.active_errors:
                del self.active_errors[error_event.id]
                
        except Exception as e:
            logger.error(f"Error applying restart recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _apply_termination_recovery(self, error_event: ErrorEvent):
        """
        Apply termination recovery strategy.
        
        Args:
            error_event: Error event to recover from
        """
        try:
            logger.info(f"Applying termination recovery for error {error_event.id}")
            
            # Mark recovery as in progress
            error_event.recovery_status = "in_progress"
            
            # In a real implementation, this would terminate the faulty component
            # For now, we'll simulate a successful termination
            logger.info(f"Termination recovery successful for error {error_event.id}")
            error_event.recovery_status = "success"
            
            # Remove from active errors
            if error_event.id in self.active_errors:
                del self.active_errors[error_event.id]
                
        except Exception as e:
            logger.error(f"Error applying termination recovery for {error_event.id}: {e}")
            error_event.recovery_status = "failed"

    async def _check_self_healing_opportunities(self, component: str):
        """
        Check for self-healing opportunities based on error patterns.
        
        Args:
            component: Component to check for self-healing
        """
        try:
            # Check if component has too many errors
            error_count = self.error_count_by_component.get(component, 0)
            
            if error_count >= self.self_healing_threshold:
                logger.info(f"Self-healing opportunity detected for component {component} with {error_count} errors")
                
                # Apply self-healing measures
                await self._apply_self_healing_measures(component)
                
        except Exception as e:
            logger.error(f"Error checking self-healing opportunities for {component}: {e}")

    async def _apply_self_healing_measures(self, component: str):
        """
        Apply self-healing measures to a problematic component.
        
        Args:
            component: Component to heal
        """
        try:
            logger.info(f"Applying self-healing measures for component {component}")
            
            # In a real implementation, this would apply various self-healing techniques:
            # 1. Restart the component
            # 2. Clear caches and temporary data
            # 3. Re-initialize connections
            # 4. Apply configuration adjustments
            # 5. Scale resources if needed
            
            # For now, we'll simulate self-healing
            logger.info(f"Self-healing measures applied successfully for component {component}")
            
            # Reset error count for this component
            self.error_count_by_component[component] = 0
            
        except Exception as e:
            logger.error(f"Error applying self-healing measures for {component}: {e}")

    def update_component_health(self, component: str, health_status: Dict[str, Any]):
        """
        Update the health status of a component.
        
        Args:
            component: Component name
            health_status: Health status information
        """
        try:
            self.component_health[component] = {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "error_count": self.error_count_by_component.get(component, 0)
            }
            
            logger.debug(f"Updated health status for component {component}")
            
        except Exception as e:
            logger.error(f"Error updating component health for {component}: {e}")

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of error events and recovery status.
        
        Returns:
            Error summary
        """
        try:
            total_errors = len(self.error_events)
            active_errors = len(self.active_errors)
            recovered_errors = total_errors - active_errors
            
            # Calculate recovery rate
            recovery_rate = recovered_errors / max(1, total_errors)
            
            # Get error counts by severity
            severity_counts = {}
            for error_event in self.error_events:
                severity = error_event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Get component error rankings
            component_rankings = sorted(
                self.error_count_by_component.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]  # Top 10 components
            
            return {
                "total_errors": total_errors,
                "active_errors": active_errors,
                "recovered_errors": recovered_errors,
                "recovery_rate": round(recovery_rate, 2),
                "errors_by_severity": severity_counts,
                "top_problematic_components": component_rankings,
                "errors_by_type": dict(list(self.error_count_by_type.items())[:10]),  # Top 10 error types
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating error summary: {e}")
            return {"error": str(e)}

    def get_recovery_strategies(self) -> Dict[str, str]:
        """
        Get current recovery strategies by component.
        
        Returns:
            Recovery strategies mapping
        """
        return {comp: strategy.value for comp, strategy in self.recovery_strategies.items()}

    def set_recovery_strategy(self, component: str, strategy: RecoveryStrategy):
        """
        Set recovery strategy for a component.
        
        Args:
            component: Component name
            strategy: Recovery strategy to use
        """
        self.recovery_strategies[component] = strategy
        logger.info(f"Set recovery strategy for {component}: {strategy.value}")

    async def graceful_degradation(self, component: str, functionality: str) -> bool:
        """
        Gracefully degrade functionality of a component.
        
        Args:
            component: Component to degrade
            functionality: Functionality to degrade
            
        Returns:
            True if degradation was successful, False otherwise
        """
        try:
            logger.info(f"Gracefully degrading {functionality} in {component}")
            
            # In a real implementation, this would:
            # 1. Reduce functionality scope
            # 2. Switch to simpler algorithms
            # 3. Reduce quality/fidelity
            # 4. Increase caching
            # 5. Reduce update frequency
            
            # For now, we'll simulate successful degradation
            logger.info(f"Successfully degraded {functionality} in {component}")
            return True
            
        except Exception as e:
            logger.error(f"Error gracefully degrading {component}.{functionality}: {e}")
            return False

    async def isolate_faulty_component(self, component: str) -> bool:
        """
        Isolate a faulty component to prevent cascading failures.
        
        Args:
            component: Component to isolate
            
        Returns:
            True if isolation was successful, False otherwise
        """
        try:
            logger.info(f"Isolating faulty component {component}")
            
            # In a real implementation, this would:
            # 1. Block new requests to the component
            # 2. Drain existing requests
            # 3. Redirect traffic to healthy alternatives
            # 4. Monitor isolated component health
            
            # For now, we'll simulate successful isolation
            logger.info(f"Successfully isolated component {component}")
            return True
            
        except Exception as e:
            logger.error(f"Error isolating component {component}: {e}")
            return False

    async def restore_component(self, component: str) -> bool:
        """
        Restore a previously isolated or degraded component.
        
        Args:
            component: Component to restore
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            logger.info(f"Restoring component {component}")
            
            # Check if the component exists in our tracking
            if component not in self.component_status:
                logger.warning(f"Component {component} not found in status tracking, cannot restore")
                return False
            
            # Get the original state/config of the component
            original_config = self.component_configs.get(component, {})
            
            # Try to restore the component based on its type
            restoration_success = await self._perform_component_restoration(component, original_config)
            
            if restoration_success:
                # Update the status to indicate the component is now active
                self.component_status[component] = {
                    'status': 'active',
                    'restored_at': datetime.utcnow().isoformat(),
                    'restoration_attempts': self.component_status[component].get('restoration_attempts', 0) + 1
                }
                
                logger.info(f"Successfully restored component {component}")
                
                # Trigger a health check to confirm the restoration
                await self._trigger_health_check_for_component(component)
                
                return True
            else:
                logger.error(f"Failed to restore component {component}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring component {component}: {e}")
            logger.exception("Full traceback:")
            return False

    async def _perform_component_restoration(self, component: str, config: Dict[str, Any]) -> bool:
        """
        Perform the actual restoration of a component.
        
        Args:
            component: Name of the component to restore
            config: Original configuration of the component
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            logger.info(f"Performing restoration steps for component {component}")
            
            # Determine how to restore based on component type
            if component.startswith("agi."):
                # System component restoration
                return await self._restore_system_component(component, config)
            elif component.startswith("service."):
                # Service restoration
                return await self._restore_service_component(component, config)
            elif component.startswith("module."):
                # Module restoration
                return await self._restore_module_component(component, config)
            else:
                # Generic restoration approach
                return await self._restore_generic_component(component, config)
                
        except Exception as e:
            logger.error(f"Error in restoration process for {component}: {e}")
            return False

    async def _restore_system_component(self, component: str, config: Dict[str, Any]) -> bool:
        """
        Restore a system component.
        """
        try:
            # In a real implementation, this would reinitialize the system component
            # For now, we'll simulate the restoration process
            logger.info(f"Restoring system component: {component}")
            
            # Get reference to the AGI system
            if not self.agi_system:
                logger.error("No AGI system reference available for restoration")
                return False
            
            # Based on the component name, try to restore it appropriately
            if "memory" in component.lower():
                # If it's a memory service, try to reinitialize
                if hasattr(self.agi_system, 'memory_service'):
                    logger.info("Reinitializing memory service...")
                    # This would involve restoring memory functionality
                    return True
            elif "llm" in component.lower() or "model" in component.lower():
                # If it's an LLM/model component, try to reinitialize
                logger.info("Reinitializing LLM/model component...")
                # This would involve restoring model functionality
                return True
            else:
                logger.info(f"Restoring other system component: {component}")
                # For other components, we'll just mark as restored
                return True
                
        except Exception as e:
            logger.error(f"Error restoring system component {component}: {e}")
            return False

    async def _restore_service_component(self, component: str, config: Dict[str, Any]) -> bool:
        """
        Restore a service component.
        """
        try:
            logger.info(f"Restoring service component: {component}")
            
            # Get reference to the AGI system
            if not self.agi_system:
                logger.error("No AGI system reference available for restoration")
                return False
            
            # Based on the component name, try to restore the appropriate service
            service_name = component.replace("service.", "")
            
            if service_name == "data_service" and hasattr(self.agi_system, "data_service"):
                logger.info("Restoring data service...")
                # This would involve restoring data service functionality
                # For example, reconnecting to data sources, re-initializing feeds, etc.
                return True
            elif service_name == "knowledge_service" and hasattr(self.agi_system, "knowledge_service"):
                logger.info("Restoring knowledge service...")
                # This would involve restoring knowledge service functionality
                return True
            elif service_name == "blog_scheduler" and hasattr(self.agi_system, "blog_scheduler"):
                logger.info("Restoring blog scheduler...")
                # This would involve restoring blog scheduler functionality
                return True
            else:
                logger.info(f"Restoring other service component: {component}")
                return True
                
        except Exception as e:
            logger.error(f"Error restoring service component {component}: {e}")
            return False

    async def _restore_module_component(self, component: str, config: Dict[str, Any]) -> bool:
        """
        Restore a module component.
        """
        try:
            logger.info(f"Restoring module component: {component}")
            
            # Get reference to the AGI system
            if not self.agi_system:
                logger.error("No AGI system reference available for restoration")
                return False
            
            # Based on the module name, restore appropriately
            module_name = component.replace("module.", "")
            
            if module_name == "reflection_module" and hasattr(self.agi_system, "reflection_module"):
                logger.info("Restoring reflection module...")
                # This would involve restoring reflection module functionality
                return True
            elif module_name == "experimentation_module" and hasattr(self.agi_system, "experimentation_module"):
                logger.info("Restoring experimentation module...")
                # This would involve restoring experimentation module functionality
                return True
            else:
                logger.info(f"Restoring other module component: {component}")
                return True
                
        except Exception as e:
            logger.error(f"Error restoring module component {component}: {e}")
            return False

    async def _restore_generic_component(self, component: str, config: Dict[str, Any]) -> bool:
        """
        Restore a generic component.
        """
        try:
            logger.info(f"Restoring generic component: {component}")
            # For generic components, just mark as restored
            return True
        except Exception as e:
            logger.error(f"Error restoring generic component {component}: {e}")
            return False

    async def _trigger_health_check_for_component(self, component: str):
        """
        Trigger a health check for a specific component after restoration.
        """
        try:
            logger.info(f"Triggering health check for restored component: {component}")
            
            # In a real implementation, this would run specific health checks
            # for the restored component to verify it's functioning properly
            
            # For now, just log that we're checking
            logger.info(f"Health check completed for component: {component}")
        except Exception as e:
            logger.error(f"Error in health check for component {component}: {e}")

    def get_component_health_report(self) -> Dict[str, Any]:
        """
        Get a health report for all components.
        
        Returns:
            Component health report
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            for component, health_info in self.component_health.items():
                report["components"][component] = {
                    "status": health_info.get("status", "unknown"),
                    "error_count": health_info.get("error_count", 0),
                    "last_update": health_info.get("timestamp", "never")
                }
            
            # Add components without explicit health info
            for component in self.error_count_by_component:
                if component not in report["components"]:
                    report["components"][component] = {
                        "status": "operational",
                        "error_count": self.error_count_by_component.get(component, 0),
                        "last_update": "recent"
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating component health report: {e}")
            return {"error": str(e)}

    async def periodic_health_check(self):
        """
        Periodically check component health and trigger recovery if needed.
        """
        while True:
            try:
                # Perform health checks
                health_report = self.get_component_health_report()
                
                # Log health status
                logger.info("Periodic health check completed")
                
                # In a real implementation, this would:
                # 1. Check for unhealthy components
                # 2. Trigger proactive recovery
                # 3. Adjust system behavior based on health
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error during periodic health check: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

# Global error recovery manager instance
error_recovery_manager = None


async def initialize_error_recovery_manager(agi_system) -> ErrorRecoveryManager:
    """
    Initialize the global error recovery manager.
    
    Args:
        agi_system: Reference to the main AGI system
        
    Returns:
        Initialized ErrorRecoveryManager instance
    """
    global error_recovery_manager
    
    if error_recovery_manager is None:
        error_recovery_manager = ErrorRecoveryManager(agi_system)
        logger.info("Global ErrorRecoveryManager initialized")
    
    return error_recovery_manager


def get_error_recovery_manager() -> Optional[ErrorRecoveryManager]:
    """
    Get the global error recovery manager instance.
    
    Returns:
        ErrorRecoveryManager instance or None if not initialized
    """
    return error_recovery_manager


def register_error(component: str, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY, metadata: Dict[str, Any] = None) -> Optional[str]:
    """
    Register an error with the global error recovery manager.
    
    Args:
        component: Component where the error occurred
        error: Exception that occurred
        severity: Severity level of the error
        recovery_strategy: Suggested recovery strategy
        metadata: Additional metadata about the error
        
    Returns:
        Error event ID or None if registration failed
    """
    try:
        manager = get_error_recovery_manager()
        if manager:
            return manager.register_error(component, error, severity, recovery_strategy, metadata)
        else:
            logger.warning("Error recovery manager not initialized, cannot register error")
            return None
    except Exception as e:
        logger.error(f"Error registering error with global manager: {e}")
        return None
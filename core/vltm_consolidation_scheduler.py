"""
Very Long-Term Memory Consolidation Scheduler

This module implements the consolidation scheduler for the Snake Agent's
Very Long-Term Memory System, managing automated consolidation triggers
based on time intervals and system events.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from core.vltm_data_models import (
    ConsolidationType, ConsolidationRequest, VLTMConfiguration
)

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of consolidation triggers"""
    TEMPORAL = "temporal"
    EVENT_BASED = "event_based"
    CAPACITY = "capacity"
    PERFORMANCE = "performance"


@dataclass
class ConsolidationTrigger:
    """Represents a consolidation trigger"""
    trigger_id: str
    trigger_type: TriggerType
    consolidation_type: ConsolidationType
    condition: str
    last_triggered: Optional[datetime]
    next_scheduled: Optional[datetime]
    enabled: bool = True
    priority: int = 1  # 1=high, 5=low


@dataclass
class ConsolidationJob:
    """Represents a scheduled consolidation job"""
    job_id: str
    consolidation_type: ConsolidationType
    scheduled_time: datetime
    priority: int
    trigger_source: str
    parameters: Dict[str, Any]


class ConsolidationScheduler:
    """
    Manages consolidation scheduling for the very long-term memory system.

    Handles temporal triggers (daily, weekly, monthly, quarterly) and
    event-based triggers for memory consolidation operations.
    """

    def __init__(self, config: VLTMConfiguration):
        """
        Initialize the consolidation scheduler.

        Args:
            config: VLTM configuration containing schedules
        """
        self.config = config

        # Consolidation callback
        self.consolidation_callback: Optional[Callable] = None

        # Trigger management
        self.triggers: Dict[str, ConsolidationTrigger] = {}
        self.scheduled_jobs: List[ConsolidationJob] = []
        self.consolidation_intervals: Dict[ConsolidationType, timedelta] = {
            ConsolidationType.DAILY: timedelta(days=1),
            ConsolidationType.WEEKLY: timedelta(weeks=1),
            ConsolidationType.MONTHLY: timedelta(days=30),
            ConsolidationType.QUARTERLY: timedelta(days=90)
        }

        # State tracking
        self.last_consolidation: Dict[ConsolidationType, datetime] = {}
        self.consolidation_counts: Dict[ConsolidationType, int] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

        # Performance metrics
        self.scheduler_metrics = {
            "triggers_activated": 0,
            "jobs_scheduled": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "uptime_start": None
        }

        # Initialize triggers
        self._initialize_triggers()

        logger.info("Consolidation scheduler initialized")

    def _initialize_triggers(self):
        """Initialize default consolidation triggers"""

        # Daily consolidation trigger
        daily_trigger = ConsolidationTrigger(
            trigger_id="daily_consolidation",
            trigger_type=TriggerType.TEMPORAL,
            consolidation_type=ConsolidationType.DAILY,
            condition="daily_at_02:00",
            last_triggered=None,
            next_scheduled=self._calculate_next_daily_trigger(),
            priority=2
        )
        self.triggers["daily_consolidation"] = daily_trigger

        # Weekly consolidation trigger
        weekly_trigger = ConsolidationTrigger(
            trigger_id="weekly_consolidation",
            trigger_type=TriggerType.TEMPORAL,
            consolidation_type=ConsolidationType.WEEKLY,
            condition="sunday_at_03:00",
            last_triggered=None,
            next_scheduled=self._calculate_next_weekly_trigger(),
            priority=1
        )
        self.triggers["weekly_consolidation"] = weekly_trigger

        # Monthly consolidation trigger
        monthly_trigger = ConsolidationTrigger(
            trigger_id="monthly_consolidation",
            trigger_type=TriggerType.TEMPORAL,
            consolidation_type=ConsolidationType.MONTHLY,
            condition="first_sunday_at_04:00",
            last_triggered=None,
            next_scheduled=self._calculate_next_monthly_trigger(),
            priority=1
        )
        self.triggers["monthly_consolidation"] = monthly_trigger

        # Quarterly consolidation trigger
        quarterly_trigger = ConsolidationTrigger(
            trigger_id="quarterly_consolidation",
            trigger_type=TriggerType.TEMPORAL,
            consolidation_type=ConsolidationType.QUARTERLY,
            condition="first_day_at_05:00",
            last_triggered=None,
            next_scheduled=self._calculate_next_quarterly_trigger(),
            priority=1
        )
        self.triggers["quarterly_consolidation"] = quarterly_trigger

        logger.info(f"Initialized {len(self.triggers)} consolidation triggers")

    def set_consolidation_callback(self, callback: Callable):
        """Set the callback function for consolidation operations"""
        self.consolidation_callback = callback
        logger.info("Consolidation callback set")

    async def start_scheduler(self) -> bool:
        """Start the consolidation scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return False

        try:
            self.running = True
            self.scheduler_metrics["uptime_start"] = datetime.utcnow()

            # Start the scheduler task
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())

            logger.info("Consolidation scheduler started")
            return True

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.running = False
            return False

    async def stop_scheduler(self):
        """Stop the consolidation scheduler"""
        if not self.running:
            return

        try:
            self.running = False

            if self.scheduler_task and not self.scheduler_task.done():
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass

            logger.info("Consolidation scheduler stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Starting scheduler loop")

        try:
            while self.running:
                # Check triggers and schedule jobs
                await self._check_triggers()

                # Execute scheduled jobs
                await self._execute_scheduled_jobs()

                # Update trigger schedules
                await self._update_trigger_schedules()

                # Wait before next cycle (check every 5 minutes)
                await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}", exc_info=True)
        finally:
            logger.info("Scheduler loop ended")

    async def _check_triggers(self):
        """Check all triggers and schedule jobs if conditions are met"""
        now = datetime.utcnow()

        for trigger_id, trigger in self.triggers.items():
            if not trigger.enabled:
                continue

            try:
                should_trigger = False

                if trigger.trigger_type == TriggerType.TEMPORAL:
                    should_trigger = await self._check_temporal_trigger(trigger, now)
                elif trigger.trigger_type == TriggerType.EVENT_BASED:
                    should_trigger = await self._check_event_trigger(trigger, now)
                elif trigger.trigger_type == TriggerType.CAPACITY:
                    should_trigger = await self._check_capacity_trigger(trigger, now)
                elif trigger.trigger_type == TriggerType.PERFORMANCE:
                    should_trigger = await self._check_performance_trigger(trigger, now)

                if should_trigger:
                    await self._schedule_consolidation_job(trigger, now)

            except Exception as e:
                logger.error(f"Error checking trigger {trigger_id}: {e}")

    async def _check_temporal_trigger(self, trigger: ConsolidationTrigger, now: datetime) -> bool:
        """Check if a temporal trigger should fire"""

        if trigger.next_scheduled is None:
            return False

        # Check if it's time for the trigger
        if now >= trigger.next_scheduled:
            logger.info(f"Temporal trigger {trigger.trigger_id} activated")
            return True

        return False

    async def _check_event_trigger(self, trigger: ConsolidationTrigger, now: datetime) -> bool:
        """Check if an event-based trigger should fire"""
        # Event triggers would be activated by external events
        # This is a placeholder for future implementation
        return False

    async def _check_capacity_trigger(self, trigger: ConsolidationTrigger, now: datetime) -> bool:
        """Check if a capacity-based trigger should fire"""
        # Capacity triggers would check memory usage thresholds
        # This is a placeholder for future implementation
        return False

    async def _check_performance_trigger(self, trigger: ConsolidationTrigger, now: datetime) -> bool:
        """Check if a performance-based trigger should fire"""
        # Performance triggers would check query times, etc.
        # This is a placeholder for future implementation
        return False

    async def _schedule_consolidation_job(self, trigger: ConsolidationTrigger, trigger_time: datetime):
        """Schedule a consolidation job from a trigger"""
        try:
            job_id = f"{trigger.consolidation_type.value}_{trigger_time.strftime('%Y%m%d_%H%M%S')}"

            # Create consolidation job
            job = ConsolidationJob(
                job_id=job_id,
                consolidation_type=trigger.consolidation_type,
                scheduled_time=trigger_time,
                priority=trigger.priority,
                trigger_source=trigger.trigger_id,
                parameters={
                    "trigger_type": trigger.trigger_type.value,
                    "trigger_condition": trigger.condition
                }
            )

            # Add to scheduled jobs
            self.scheduled_jobs.append(job)
            self.scheduled_jobs.sort(
                key=lambda j: (j.priority, j.scheduled_time))

            # Update trigger state
            trigger.last_triggered = trigger_time

            # Calculate next trigger time
            trigger.next_scheduled = self._calculate_next_trigger_time(trigger)

            # Update metrics
            self.scheduler_metrics["triggers_activated"] += 1
            self.scheduler_metrics["jobs_scheduled"] += 1

            logger.info(f"Scheduled consolidation job: {job_id} "
                        f"(type: {trigger.consolidation_type}, priority: {job.priority})")

        except Exception as e:
            logger.error(
                f"Error scheduling job from trigger {trigger.trigger_id}: {e}")

    async def _execute_scheduled_jobs(self):
        """Execute scheduled consolidation jobs"""
        now = datetime.utcnow()

        # Find jobs ready to execute
        ready_jobs = [
            job for job in self.scheduled_jobs if job.scheduled_time <= now]

        for job in ready_jobs:
            try:
                await self._execute_consolidation_job(job)
                self.scheduled_jobs.remove(job)

            except Exception as e:
                logger.error(f"Error executing job {job.job_id}: {e}")
                self.scheduled_jobs.remove(job)  # Remove failed job
                self.scheduler_metrics["jobs_failed"] += 1

    async def _execute_consolidation_job(self, job: ConsolidationJob):
        """Execute a specific consolidation job"""
        if not self.consolidation_callback:
            logger.warning(
                f"No consolidation callback set for job {job.job_id}")
            return

        try:
            logger.info(f"Executing consolidation job: {job.job_id}")

            # Create consolidation request
            request = ConsolidationRequest(
                consolidation_type=job.consolidation_type,
                force_consolidation=True
            )

            # Execute consolidation
            result = await self.consolidation_callback(request)

            # Update state
            self.last_consolidation[job.consolidation_type] = datetime.utcnow()
            self.consolidation_counts[job.consolidation_type] = \
                self.consolidation_counts.get(job.consolidation_type, 0) + 1

            # Update metrics
            if result.success:
                self.scheduler_metrics["jobs_completed"] += 1
                logger.info(
                    f"Consolidation job {job.job_id} completed successfully")
            else:
                self.scheduler_metrics["jobs_failed"] += 1
                logger.error(
                    f"Consolidation job {job.job_id} failed: {result.error_message}")

        except Exception as e:
            logger.error(
                f"Error executing consolidation job {job.job_id}: {e}")
            self.scheduler_metrics["jobs_failed"] += 1

    async def _update_trigger_schedules(self):
        """Update trigger schedules based on current time"""
        now = datetime.utcnow()

        for trigger in self.triggers.values():
            if trigger.trigger_type == TriggerType.TEMPORAL and trigger.next_scheduled:
                # Ensure next scheduled time is in the future
                if trigger.next_scheduled <= now:
                    trigger.next_scheduled = self._calculate_next_trigger_time(
                        trigger)

    def _calculate_next_trigger_time(self, trigger: ConsolidationTrigger) -> datetime:
        """Calculate the next trigger time for a trigger"""

        if trigger.consolidation_type == ConsolidationType.DAILY:
            return self._calculate_next_daily_trigger()
        elif trigger.consolidation_type == ConsolidationType.WEEKLY:
            return self._calculate_next_weekly_trigger()
        elif trigger.consolidation_type == ConsolidationType.MONTHLY:
            return self._calculate_next_monthly_trigger()
        elif trigger.consolidation_type == ConsolidationType.QUARTERLY:
            return self._calculate_next_quarterly_trigger()
        else:
            # Default to next day
            return datetime.utcnow() + timedelta(days=1)

    def _calculate_next_daily_trigger(self) -> datetime:
        """Calculate next daily trigger time (02:00 daily)"""
        now = datetime.utcnow()
        next_trigger = now.replace(hour=2, minute=0, second=0, microsecond=0)

        # If it's already past 02:00 today, schedule for tomorrow
        if now.hour >= 2:
            next_trigger += timedelta(days=1)

        return next_trigger

    def _calculate_next_weekly_trigger(self) -> datetime:
        """Calculate next weekly trigger time (Sunday 03:00)"""
        now = datetime.utcnow()

        # Find next Sunday
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 3:
            days_until_sunday = 7  # Next Sunday if it's already past 03:00 this Sunday

        next_sunday = now + timedelta(days=days_until_sunday)
        next_trigger = next_sunday.replace(
            hour=3, minute=0, second=0, microsecond=0)

        return next_trigger

    def _calculate_next_monthly_trigger(self) -> datetime:
        """Calculate next monthly trigger time (first Sunday 04:00)"""
        now = datetime.utcnow()

        # Find first Sunday of next month
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)

        # Find first Sunday
        first_sunday = next_month
        while first_sunday.weekday() != 6:  # Sunday is 6
            first_sunday += timedelta(days=1)

        next_trigger = first_sunday.replace(
            hour=4, minute=0, second=0, microsecond=0)

        return next_trigger

    def _calculate_next_quarterly_trigger(self) -> datetime:
        """Calculate next quarterly trigger time (first day 05:00)"""
        now = datetime.utcnow()

        # Find next quarter
        current_quarter = (now.month - 1) // 3 + 1
        next_quarter = current_quarter + 1

        if next_quarter > 4:
            next_quarter = 1
            next_year = now.year + 1
        else:
            next_year = now.year

        next_quarter_month = (next_quarter - 1) * 3 + 1
        next_trigger = datetime(next_year, next_quarter_month, 1, 5, 0, 0)

        return next_trigger

    # Public interface methods

    def is_consolidation_due(self, consolidation_type: ConsolidationType) -> bool:
        """Check if consolidation is due for the given type"""
        last_time = self.last_consolidation.get(consolidation_type)
        if not last_time:
            return True

        interval = self.consolidation_intervals.get(
            consolidation_type, timedelta(days=1))
        return datetime.utcnow() - last_time >= interval

    def schedule_immediate_consolidation(self, consolidation_type: ConsolidationType, priority: int = 1) -> str:
        """Schedule an immediate consolidation job"""
        try:
            job_id = f"immediate_{consolidation_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            job = ConsolidationJob(
                job_id=job_id,
                consolidation_type=consolidation_type,
                scheduled_time=datetime.utcnow(),
                priority=priority,
                trigger_source="manual",
                parameters={"immediate": True}
            )

            self.scheduled_jobs.append(job)
            self.scheduled_jobs.sort(
                key=lambda j: (j.priority, j.scheduled_time))

            self.scheduler_metrics["jobs_scheduled"] += 1

            logger.info(f"Scheduled immediate consolidation: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Error scheduling immediate consolidation: {e}")
            return ""

    def add_custom_trigger(self, trigger: ConsolidationTrigger):
        """Add a custom consolidation trigger"""
        self.triggers[trigger.trigger_id] = trigger
        logger.info(f"Added custom trigger: {trigger.trigger_id}")

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a consolidation trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            logger.info(f"Removed trigger: {trigger_id}")
            return True
        return False

    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a consolidation trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True
            logger.info(f"Enabled trigger: {trigger_id}")
            return True
        return False

    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a consolidation trigger"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False
            logger.info(f"Disabled trigger: {trigger_id}")
            return True
        return False

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        now = datetime.utcnow()
        uptime = None

        if self.scheduler_metrics["uptime_start"]:
            uptime = (
                now - self.scheduler_metrics["uptime_start"]).total_seconds()

        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "triggers_count": len(self.triggers),
            "enabled_triggers": len([t for t in self.triggers.values() if t.enabled]),
            "scheduled_jobs": len(self.scheduled_jobs),
            "last_consolidation_times": {
                k.value: v.isoformat() for k, v in self.last_consolidation.items()
            },
            "consolidation_counts": {
                k.value: v for k, v in self.consolidation_counts.items()
            },
            "metrics": self.scheduler_metrics.copy(),
            "next_trigger_times": {
                trigger_id: trigger.next_scheduled.isoformat() if trigger.next_scheduled else None
                for trigger_id, trigger in self.triggers.items()
            }
        }

    def get_trigger_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all triggers"""
        return [
            {
                "trigger_id": trigger.trigger_id,
                "trigger_type": trigger.trigger_type.value,
                "consolidation_type": trigger.consolidation_type.value,
                "condition": trigger.condition,
                "enabled": trigger.enabled,
                "priority": trigger.priority,
                "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                "next_scheduled": trigger.next_scheduled.isoformat() if trigger.next_scheduled else None
            }
            for trigger in self.triggers.values()
        ]

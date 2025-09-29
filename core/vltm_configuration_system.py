"""
Very Long-Term Memory Configuration System

This module implements a comprehensive configuration system for VLTM operations,
including retention policies, consolidation schedules, and performance parameters.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigScope(str, Enum):
    """Configuration scope levels"""
    GLOBAL = "global"
    SYSTEM = "system"
    COMPONENT = "component"
    USER = "user"


class ConfigType(str, Enum):
    """Types of configuration settings"""
    RETENTION_POLICY = "retention_policy"
    CONSOLIDATION_SCHEDULE = "consolidation_schedule"
    PERFORMANCE_PARAMETERS = "performance_parameters"
    STORAGE_SETTINGS = "storage_settings"
    COMPRESSION_SETTINGS = "compression_settings"
    INDEXING_SETTINGS = "indexing_settings"
    ALERT_SETTINGS = "alert_settings"


@dataclass
class RetentionPolicy:
    """Memory retention policy configuration"""
    policy_name: str
    memory_type: str
    short_term_days: int = 7
    medium_term_days: int = 30
    long_term_days: int = 365
    very_long_term_days: int = 3650  # 10 years
    archive_after_days: Optional[int] = None
    delete_after_days: Optional[int] = None
    importance_threshold_archive: float = 0.3
    importance_threshold_delete: float = 0.1
    preserve_strategic_always: bool = True
    compress_after_days: int = 90


@dataclass
class ConsolidationSchedule:
    """Memory consolidation schedule configuration"""
    schedule_name: str
    daily_consolidation_hour: int = 2  # 2 AM
    weekly_consolidation_day: int = 0  # Sunday
    monthly_consolidation_day: int = 1  # 1st of month
    quarterly_consolidation_month: int = 1  # January, April, July, October
    pattern_extraction_interval_hours: int = 6
    strategic_synthesis_interval_hours: int = 24
    cleanup_interval_hours: int = 168  # Weekly
    max_batch_size: int = 1000
    consolidation_timeout_minutes: int = 120


@dataclass
class PerformanceParameters:
    """Performance tuning parameters"""
    parameter_set_name: str
    max_concurrent_operations: int = 10
    cache_size_mb: int = 256
    cache_ttl_seconds: int = 3600
    query_timeout_seconds: int = 30
    index_rebuild_threshold: float = 0.7  # Rebuild when 70% fragmented
    compression_batch_size: int = 100
    retrieval_page_size: int = 50
    connection_pool_size: int = 20
    memory_warning_threshold_mb: int = 512
    memory_critical_threshold_mb: int = 1024


@dataclass
class StorageSettings:
    """Storage configuration settings"""
    storage_name: str
    postgresql_connection_string: str = ""
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    file_storage_path: str = "./vltm_storage"
    backup_path: str = "./vltm_backup"
    max_file_size_mb: int = 100
    compression_enabled: bool = True
    encryption_enabled: bool = False
    backup_interval_hours: int = 24
    cleanup_old_backups_days: int = 30


@dataclass
class CompressionSettings:
    """Compression configuration settings"""
    compression_name: str
    enable_compression: bool = True
    compression_algorithm: str = "zlib"
    compression_level: int = 6
    age_threshold_days: int = 30
    importance_threshold: float = 0.5
    pattern_abstraction_enabled: bool = True
    semantic_compression_enabled: bool = True
    temporal_compression_enabled: bool = True
    max_compression_ratio: float = 0.8


@dataclass
class IndexingSettings:
    """Indexing configuration settings"""
    indexing_name: str
    enable_semantic_index: bool = True
    enable_temporal_index: bool = True
    enable_causal_index: bool = True
    enable_strategic_index: bool = True
    enable_pattern_index: bool = True
    index_update_interval_minutes: int = 60
    semantic_similarity_threshold: float = 0.7
    index_cache_size_mb: int = 128
    rebuild_threshold_entries: int = 10000
    optimize_interval_hours: int = 24


@dataclass
class AlertSettings:
    """Alert configuration settings"""
    alert_name: str
    enable_alerts: bool = True
    email_notifications: bool = False
    webhook_notifications: bool = False
    max_operation_time_ms: float = 5000
    min_success_rate: float = 95.0
    max_error_rate: float = 5.0
    max_memory_usage_mb: float = 1000
    min_cache_hit_rate: float = 80.0
    alert_cooldown_minutes: int = 30
    critical_alert_immediate: bool = True


class VLTMConfigurationManager:
    """
    Comprehensive configuration management system for VLTM.

    Manages configuration for:
    - Memory retention policies
    - Consolidation schedules
    - Performance parameters
    - Storage settings
    - Compression settings
    - Indexing settings
    - Alert settings
    """

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Configuration storage
        self.configurations: Dict[str, Dict[str, Any]] = {
            ConfigType.RETENTION_POLICY.value: {},
            ConfigType.CONSOLIDATION_SCHEDULE.value: {},
            ConfigType.PERFORMANCE_PARAMETERS.value: {},
            ConfigType.STORAGE_SETTINGS.value: {},
            ConfigType.COMPRESSION_SETTINGS.value: {},
            ConfigType.INDEXING_SETTINGS.value: {},
            ConfigType.ALERT_SETTINGS.value: {}
        }

        # Active configurations
        self.active_configs: Dict[ConfigType, str] = {}

        # Configuration history
        self.config_history: List[Dict[str, Any]] = []

        logger.info(
            f"VLTM Configuration Manager initialized with config dir: {config_dir}")

    async def initialize(self) -> bool:
        """Initialize configuration manager with default configurations"""
        try:
            # Create default configurations
            await self._create_default_configurations()

            # Load existing configurations
            await self._load_configurations()

            # Set default active configurations
            await self._set_default_active_configurations()

            logger.info("Configuration Manager initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Configuration Manager: {e}")
            return False

    async def _create_default_configurations(self):
        """Create default configuration sets"""

        # Default retention policy
        default_retention = RetentionPolicy(
            policy_name="default",
            memory_type="all",
            short_term_days=7,
            medium_term_days=30,
            long_term_days=365,
            very_long_term_days=3650,
            archive_after_days=1095,  # 3 years
            importance_threshold_archive=0.3,
            preserve_strategic_always=True,
            compress_after_days=90
        )

        # Strategic memory retention policy
        strategic_retention = RetentionPolicy(
            policy_name="strategic",
            memory_type="strategic_knowledge",
            short_term_days=30,
            medium_term_days=90,
            long_term_days=1825,  # 5 years
            very_long_term_days=7300,  # 20 years
            importance_threshold_archive=0.1,
            preserve_strategic_always=True,
            compress_after_days=180
        )

        # Default consolidation schedule
        default_consolidation = ConsolidationSchedule(
            schedule_name="default",
            daily_consolidation_hour=2,
            weekly_consolidation_day=0,
            monthly_consolidation_day=1,
            pattern_extraction_interval_hours=6,
            strategic_synthesis_interval_hours=24,
            cleanup_interval_hours=168,
            max_batch_size=1000
        )

        # High-frequency consolidation schedule
        high_freq_consolidation = ConsolidationSchedule(
            schedule_name="high_frequency",
            daily_consolidation_hour=2,
            pattern_extraction_interval_hours=2,
            strategic_synthesis_interval_hours=8,
            cleanup_interval_hours=24,
            max_batch_size=500
        )

        # Default performance parameters
        default_performance = PerformanceParameters(
            parameter_set_name="default",
            max_concurrent_operations=10,
            cache_size_mb=256,
            cache_ttl_seconds=3600,
            query_timeout_seconds=30,
            compression_batch_size=100,
            retrieval_page_size=50
        )

        # High-performance parameters
        high_performance = PerformanceParameters(
            parameter_set_name="high_performance",
            max_concurrent_operations=20,
            cache_size_mb=512,
            cache_ttl_seconds=7200,
            query_timeout_seconds=60,
            compression_batch_size=200,
            retrieval_page_size=100
        )

        # Default storage settings
        default_storage = StorageSettings(
            storage_name="default",
            postgresql_connection_string="postgresql://localhost:5432/vltm",
            chromadb_host="localhost",
            chromadb_port=8000,
            file_storage_path="./vltm_storage",
            backup_path="./vltm_backup",
            compression_enabled=True,
            backup_interval_hours=24
        )

        # Default compression settings
        default_compression = CompressionSettings(
            compression_name="default",
            enable_compression=True,
            compression_algorithm="zlib",
            compression_level=6,
            age_threshold_days=30,
            importance_threshold=0.5,
            pattern_abstraction_enabled=True
        )

        # Default indexing settings
        default_indexing = IndexingSettings(
            indexing_name="default",
            enable_semantic_index=True,
            enable_temporal_index=True,
            enable_causal_index=True,
            enable_strategic_index=True,
            index_update_interval_minutes=60,
            semantic_similarity_threshold=0.7
        )

        # Default alert settings
        default_alerts = AlertSettings(
            alert_name="default",
            enable_alerts=True,
            max_operation_time_ms=5000,
            min_success_rate=95.0,
            max_error_rate=5.0,
            max_memory_usage_mb=1000,
            alert_cooldown_minutes=30
        )

        # Store default configurations
        await self.add_configuration(ConfigType.RETENTION_POLICY, "default", asdict(default_retention))
        await self.add_configuration(ConfigType.RETENTION_POLICY, "strategic", asdict(strategic_retention))
        await self.add_configuration(ConfigType.CONSOLIDATION_SCHEDULE, "default", asdict(default_consolidation))
        await self.add_configuration(ConfigType.CONSOLIDATION_SCHEDULE, "high_frequency", asdict(high_freq_consolidation))
        await self.add_configuration(ConfigType.PERFORMANCE_PARAMETERS, "default", asdict(default_performance))
        await self.add_configuration(ConfigType.PERFORMANCE_PARAMETERS, "high_performance", asdict(high_performance))
        await self.add_configuration(ConfigType.STORAGE_SETTINGS, "default", asdict(default_storage))
        await self.add_configuration(ConfigType.COMPRESSION_SETTINGS, "default", asdict(default_compression))
        await self.add_configuration(ConfigType.INDEXING_SETTINGS, "default", asdict(default_indexing))
        await self.add_configuration(ConfigType.ALERT_SETTINGS, "default", asdict(default_alerts))

    async def _set_default_active_configurations(self):
        """Set default active configurations"""
        for config_type in ConfigType:
            if config_type.value in self.configurations and self.configurations[config_type.value]:
                # Set first available configuration as active
                first_config = list(
                    self.configurations[config_type.value].keys())[0]
                self.active_configs[config_type] = first_config

    async def add_configuration(self, config_type: ConfigType, name: str,
                                config_data: Dict[str, Any]) -> bool:
        """Add a new configuration"""
        try:
            if config_type.value not in self.configurations:
                self.configurations[config_type.value] = {}

            self.configurations[config_type.value][name] = {
                "config_data": config_data,
                "created_at": datetime.utcnow().isoformat(),
                "modified_at": datetime.utcnow().isoformat(),
                "version": 1
            }

            # Save to file
            await self._save_configuration_to_file(config_type, name)

            # Add to history
            self._add_to_history("add", config_type, name, config_data)

            logger.info(f"Added configuration: {config_type.value}/{name}")
            return True

        except Exception as e:
            logger.error(
                f"Error adding configuration {config_type.value}/{name}: {e}")
            return False

    async def update_configuration(self, config_type: ConfigType, name: str,
                                   config_data: Dict[str, Any]) -> bool:
        """Update an existing configuration"""
        try:
            if config_type.value not in self.configurations or name not in self.configurations[config_type.value]:
                logger.error(
                    f"Configuration not found: {config_type.value}/{name}")
                return False

            # Increment version
            current_config = self.configurations[config_type.value][name]
            current_config["version"] += 1
            current_config["config_data"] = config_data
            current_config["modified_at"] = datetime.utcnow().isoformat()

            # Save to file
            await self._save_configuration_to_file(config_type, name)

            # Add to history
            self._add_to_history("update", config_type, name, config_data)

            logger.info(f"Updated configuration: {config_type.value}/{name}")
            return True

        except Exception as e:
            logger.error(
                f"Error updating configuration {config_type.value}/{name}: {e}")
            return False

    async def get_configuration(self, config_type: ConfigType, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific configuration"""
        try:
            if config_type.value in self.configurations and name in self.configurations[config_type.value]:
                return self.configurations[config_type.value][name]["config_data"]
            return None
        except Exception as e:
            logger.error(
                f"Error getting configuration {config_type.value}/{name}: {e}")
            return None

    async def get_active_configuration(self, config_type: ConfigType) -> Optional[Dict[str, Any]]:
        """Get the currently active configuration for a type"""
        try:
            if config_type in self.active_configs:
                active_name = self.active_configs[config_type]
                return await self.get_configuration(config_type, active_name)
            return None
        except Exception as e:
            logger.error(
                f"Error getting active configuration {config_type.value}: {e}")
            return None

    async def set_active_configuration(self, config_type: ConfigType, name: str) -> bool:
        """Set the active configuration for a type"""
        try:
            # Verify configuration exists
            if config_type.value not in self.configurations or name not in self.configurations[config_type.value]:
                logger.error(
                    f"Configuration not found: {config_type.value}/{name}")
                return False

            self.active_configs[config_type] = name

            # Add to history
            self._add_to_history("activate", config_type, name, {})

            logger.info(
                f"Set active configuration: {config_type.value}/{name}")
            return True

        except Exception as e:
            logger.error(
                f"Error setting active configuration {config_type.value}/{name}: {e}")
            return False

    async def list_configurations(self, config_type: ConfigType) -> List[str]:
        """List all configurations of a specific type"""
        try:
            if config_type.value in self.configurations:
                return list(self.configurations[config_type.value].keys())
            return []
        except Exception as e:
            logger.error(
                f"Error listing configurations {config_type.value}: {e}")
            return []

    async def delete_configuration(self, config_type: ConfigType, name: str) -> bool:
        """Delete a configuration"""
        try:
            if config_type.value not in self.configurations or name not in self.configurations[config_type.value]:
                logger.error(
                    f"Configuration not found: {config_type.value}/{name}")
                return False

            # Cannot delete active configuration
            if config_type in self.active_configs and self.active_configs[config_type] == name:
                logger.error(
                    f"Cannot delete active configuration: {config_type.value}/{name}")
                return False

            # Remove from memory
            del self.configurations[config_type.value][name]

            # Remove file
            config_file = self.config_dir / f"{config_type.value}_{name}.json"
            if config_file.exists():
                config_file.unlink()

            # Add to history
            self._add_to_history("delete", config_type, name, {})

            logger.info(f"Deleted configuration: {config_type.value}/{name}")
            return True

        except Exception as e:
            logger.error(
                f"Error deleting configuration {config_type.value}/{name}: {e}")
            return False

    async def export_configuration(self, config_type: ConfigType, name: str,
                                   export_path: str) -> bool:
        """Export a configuration to a file"""
        try:
            config_data = await self.get_configuration(config_type, name)
            if not config_data:
                return False

            export_data = {
                "config_type": config_type.value,
                "config_name": name,
                "config_data": config_data,
                "exported_at": datetime.utcnow().isoformat()
            }

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(
                f"Exported configuration {config_type.value}/{name} to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

    async def import_configuration(self, import_path: str) -> bool:
        """Import a configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            config_type = ConfigType(import_data["config_type"])
            config_name = import_data["config_name"]
            config_data = import_data["config_data"]

            return await self.add_configuration(config_type, config_name, config_data)

        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

    async def validate_configuration(self, config_type: ConfigType,
                                     config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration against its schema"""
        errors = []

        try:
            if config_type == ConfigType.RETENTION_POLICY:
                errors.extend(self._validate_retention_policy(config_data))
            elif config_type == ConfigType.CONSOLIDATION_SCHEDULE:
                errors.extend(
                    self._validate_consolidation_schedule(config_data))
            elif config_type == ConfigType.PERFORMANCE_PARAMETERS:
                errors.extend(
                    self._validate_performance_parameters(config_data))
            elif config_type == ConfigType.STORAGE_SETTINGS:
                errors.extend(self._validate_storage_settings(config_data))
            elif config_type == ConfigType.COMPRESSION_SETTINGS:
                errors.extend(self._validate_compression_settings(config_data))
            elif config_type == ConfigType.INDEXING_SETTINGS:
                errors.extend(self._validate_indexing_settings(config_data))
            elif config_type == ConfigType.ALERT_SETTINGS:
                errors.extend(self._validate_alert_settings(config_data))

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False, [str(e)]

    def _validate_retention_policy(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate retention policy configuration"""
        errors = []

        required_fields = ["policy_name", "memory_type",
                           "short_term_days", "medium_term_days", "long_term_days"]
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")

        # Validate day values are positive and in order
        if all(field in config_data for field in ["short_term_days", "medium_term_days", "long_term_days"]):
            if not (config_data["short_term_days"] < config_data["medium_term_days"] < config_data["long_term_days"]):
                errors.append("Retention periods must be in ascending order")

        return errors

    def _validate_consolidation_schedule(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate consolidation schedule configuration"""
        errors = []

        if "daily_consolidation_hour" in config_data:
            hour = config_data["daily_consolidation_hour"]
            if not 0 <= hour <= 23:
                errors.append(
                    "daily_consolidation_hour must be between 0 and 23")

        if "weekly_consolidation_day" in config_data:
            day = config_data["weekly_consolidation_day"]
            if not 0 <= day <= 6:
                errors.append(
                    "weekly_consolidation_day must be between 0 and 6")

        return errors

    def _validate_performance_parameters(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate performance parameters configuration"""
        errors = []

        # Validate positive values
        positive_fields = ["max_concurrent_operations",
                           "cache_size_mb", "cache_ttl_seconds"]
        for field in positive_fields:
            if field in config_data and config_data[field] <= 0:
                errors.append(f"{field} must be positive")

        return errors

    def _validate_storage_settings(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate storage settings configuration"""
        errors = []

        if "chromadb_port" in config_data:
            port = config_data["chromadb_port"]
            if not 1 <= port <= 65535:
                errors.append("chromadb_port must be between 1 and 65535")

        return errors

    def _validate_compression_settings(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate compression settings configuration"""
        errors = []

        if "compression_level" in config_data:
            level = config_data["compression_level"]
            if not 1 <= level <= 9:
                errors.append("compression_level must be between 1 and 9")

        return errors

    def _validate_indexing_settings(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate indexing settings configuration"""
        errors = []

        if "semantic_similarity_threshold" in config_data:
            threshold = config_data["semantic_similarity_threshold"]
            if not 0.0 <= threshold <= 1.0:
                errors.append(
                    "semantic_similarity_threshold must be between 0.0 and 1.0")

        return errors

    def _validate_alert_settings(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate alert settings configuration"""
        errors = []

        if "min_success_rate" in config_data:
            rate = config_data["min_success_rate"]
            if not 0.0 <= rate <= 100.0:
                errors.append("min_success_rate must be between 0.0 and 100.0")

        return errors

    async def _save_configuration_to_file(self, config_type: ConfigType, name: str):
        """Save configuration to file"""
        try:
            config_file = self.config_dir / f"{config_type.value}_{name}.json"
            config_data = self.configurations[config_type.value][name]

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")

    async def _load_configurations(self):
        """Load configurations from files"""
        try:
            for config_file in self.config_dir.glob("*.json"):
                if "_" in config_file.stem:
                    config_type_str, name = config_file.stem.split("_", 1)

                    try:
                        config_type = ConfigType(config_type_str)

                        with open(config_file, 'r') as f:
                            config_data = json.load(f)

                        if config_type.value not in self.configurations:
                            self.configurations[config_type.value] = {}

                        self.configurations[config_type.value][name] = config_data

                    except ValueError:
                        logger.warning(
                            f"Unknown config type in file: {config_file}")

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")

    def _add_to_history(self, action: str, config_type: ConfigType, name: str, config_data: Dict[str, Any]):
        """Add action to configuration history"""
        history_entry = {
            "action": action,
            "config_type": config_type.value,
            "config_name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "config_data_preview": str(config_data)[:200] + "..." if len(str(config_data)) > 200 else str(config_data)
        }

        self.config_history.append(history_entry)

        # Keep only recent history (last 1000 entries)
        if len(self.config_history) > 1000:
            self.config_history = self.config_history[-500:]

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        summary = {}

        for config_type in ConfigType:
            type_configs = self.configurations.get(config_type.value, {})
            active_config = self.active_configs.get(config_type, "none")

            summary[config_type.value] = {
                "total_configurations": len(type_configs),
                "active_configuration": active_config,
                "available_configurations": list(type_configs.keys())
            }

        return {
            "configuration_summary": summary,
            "total_history_entries": len(self.config_history),
            "config_directory": str(self.config_dir),
            "last_activity": self.config_history[-1]["timestamp"] if self.config_history else None
        }

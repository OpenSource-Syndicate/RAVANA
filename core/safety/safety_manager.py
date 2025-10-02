"""
Safety Manager for RAVANA AGI System

This module provides comprehensive safety safeguards for autonomous execution,
including ethical filtering, harm prevention, and execution monitoring.
"""

import asyncio
import logging
import json
import hashlib
import traceback
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import re

from core.config import Config

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Enumeration of safety levels."""
    LOW = "low"          # Minimal safeguards
    MEDIUM = "medium"    # Standard safeguards
    HIGH = "high"        # Enhanced safeguards
    CRITICAL = "critical" # Maximum safeguards


class SafetyViolationType(Enum):
    """Enumeration of safety violation types."""
    ETHICAL_CONCERN = "ethical_concern"
    HARM_RISK = "harm_risk"
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_RISK = "security_risk"
    RESOURCE_ABUSE = "resource_abuse"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    LEGAL_CONCERN = "legal_concern"


@dataclass
class SafetyViolation:
    """Represents a safety violation event."""
    id: str
    timestamp: datetime
    component: str
    violation_type: SafetyViolationType
    description: str
    severity: SafetyLevel
    action_taken: str  # blocked, allowed_with_warning, allowed_with_conditions
    metadata: Dict[str, Any]


class SafetyManager:
    """Manages safety safeguards for autonomous execution."""

    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()
        
        # Safety tracking
        self.safety_violations: List[SafetyViolation] = []
        self.active_violations: Dict[str, SafetyViolation] = {}
        self.violation_count_by_type: Dict[str, int] = {}
        self.violation_count_by_component: Dict[str, int] = {}
        
        # Safety configurations
        self.default_safety_level = getattr(Config, 'SAFETY_DEFAULT_LEVEL', SafetyLevel.HIGH)
        self.ethical_filtering_enabled = getattr(Config, 'SAFETY_ETHICAL_FILTERING_ENABLED', True)
        self.harm_prevention_enabled = getattr(Config, 'SAFETY_HARM_PREVENTION_ENABLED', True)
        self.privacy_protection_enabled = getattr(Config, 'SAFETY_PRIVACY_PROTECTION_ENABLED', True)
        self.security_enforcement_enabled = getattr(Config, 'SAFETY_SECURITY_ENFORCEMENT_ENABLED', True)
        
        # Ethical guidelines
        self.ethical_guidelines = self._load_ethical_guidelines()
        
        # Harm prevention patterns
        self.harm_prevention_patterns = self._load_harm_prevention_patterns()
        
        # Privacy protection rules
        self.privacy_protection_rules = self._load_privacy_protection_rules()
        
        # Security enforcement policies
        self.security_policies = self._load_security_policies()
        
        # Authorized actions whitelist
        self.authorized_actions: Set[str] = set()
        self._load_authorized_actions()
        
        # Restricted content patterns
        self.restricted_patterns = self._compile_restricted_patterns()
        
        # Execution monitoring
        self.execution_monitor_enabled = getattr(Config, 'SAFETY_EXECUTION_MONITOR_ENABLED', True)
        self.execution_history: List[Dict[str, Any]] = []
        self.suspicious_activities: List[Dict[str, Any]] = []
        
        # Emergency shutdown
        self.emergency_shutdown_threshold = getattr(Config, 'SAFETY_EMERGENCY_SHUTDOWN_THRESHOLD', 10)
        self.emergency_shutdown_enabled = getattr(Config, 'SAFETY_EMERGENCY_SHUTDOWN_ENABLED', True)
        
        logger.info("Safety Manager initialized with comprehensive safeguards")

    def _load_ethical_guidelines(self) -> Dict[str, Any]:
        """Load ethical guidelines from configuration."""
        try:
            # Default ethical guidelines
            guidelines = {
                "core_principles": [
                    "Do no harm to humans or other sentient beings",
                    "Respect human autonomy and dignity",
                    "Promote fairness and justice",
                    "Protect privacy and confidentiality",
                    "Ensure transparency and accountability"
                ],
                "prohibited_actions": [
                    "Causing physical harm to humans",
                    "Violating human rights",
                    "Compromising privacy without consent",
                    "Engaging in deception or manipulation",
                    "Participating in illegal activities"
                ],
                "required_considerations": [
                    "Impact on human welfare",
                    "Respect for individual rights",
                    "Fairness in treatment",
                    "Environmental sustainability",
                    "Long-term consequences"
                ]
            }
            
            # Override with custom guidelines if available
            custom_guidelines = getattr(Config, 'SAFETY_ETHICAL_GUIDELINES', {})
            if custom_guidelines:
                guidelines.update(custom_guidelines)
                
            logger.info("Ethical guidelines loaded successfully")
            return guidelines
            
        except Exception as e:
            logger.error(f"Error loading ethical guidelines: {e}")
            # Return default guidelines
            return {
                "core_principles": [
                    "Do no harm to humans or other sentient beings",
                    "Respect human autonomy and dignity",
                    "Promote fairness and justice"
                ],
                "prohibited_actions": [
                    "Causing physical harm to humans",
                    "Violating human rights"
                ],
                "required_considerations": [
                    "Impact on human welfare",
                    "Respect for individual rights"
                ]
            }

    def _load_harm_prevention_patterns(self) -> List[Dict[str, Any]]:
        """Load harm prevention patterns."""
        try:
            patterns = [
                {
                    "category": "physical_harm",
                    "keywords": ["hurt", "injure", "damage", "destroy", "kill", "attack", "violence", "weapon"],
                    "severity": SafetyLevel.HIGH,
                    "description": "Patterns indicating potential for physical harm"
                },
                {
                    "category": "psychological_harm",
                    "keywords": ["manipulate", "deceive", "gaslight", "traumatize", "distress"],
                    "severity": SafetyLevel.MEDIUM,
                    "description": "Patterns indicating potential for psychological harm"
                },
                {
                    "category": "privacy_violation",
                    "keywords": ["spy", "monitor", "track", "surveil", "personal information", "private data"],
                    "severity": SafetyLevel.HIGH,
                    "description": "Patterns indicating potential privacy violations"
                },
                {
                    "category": "illegal_activity",
                    "keywords": ["steal", "hack", "break in", "trespass", "illegal", "crime"],
                    "severity": SafetyLevel.CRITICAL,
                    "description": "Patterns indicating potential illegal activities"
                }
            ]
            
            logger.info("Harm prevention patterns loaded successfully")
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading harm prevention patterns: {e}")
            return []

    def _load_privacy_protection_rules(self) -> Dict[str, Any]:
        """Load privacy protection rules."""
        try:
            rules = {
                "pii_protection": {
                    "enabled": True,
                    "patterns": [
                        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                        r"\b\d{16}\b",             # Credit card
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"  # Phone number
                    ],
                    "action": "redact"
                },
                "conversation_privacy": {
                    "enabled": True,
                    "sensitive_topics": ["medical", "financial", "personal"],
                    "action": "limit_sharing"
                },
                "data_retention": {
                    "enabled": True,
                    "max_retention_days": 365,
                    "action": "auto_delete"
                }
            }
            
            logger.info("Privacy protection rules loaded successfully")
            return rules
            
        except Exception as e:
            logger.error(f"Error loading privacy protection rules: {e}")
            return {}

    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security enforcement policies."""
        try:
            policies = {
                "access_control": {
                    "enabled": True,
                    "authentication_required": True,
                    "authorization_checks": True
                },
                "input_validation": {
                    "enabled": True,
                    "max_input_length": 10000,
                    "allowed_characters": "printable_ascii"
                },
                "output_sanitization": {
                    "enabled": True,
                    "sanitize_html": True,
                    "escape_special_chars": True
                },
                "network_security": {
                    "enabled": True,
                    "restrict_external_connections": True,
                    "whitelist_domains": []
                }
            }
            
            logger.info("Security policies loaded successfully")
            return policies
            
        except Exception as e:
            logger.error(f"Error loading security policies: {e}")
            return {}

    def _load_authorized_actions(self):
        """Load authorized actions whitelist."""
        try:
            # Default authorized actions
            default_actions = {
                "log_message",
                "write_python_code",
                "execute_python_file",
                "search_internet",
                "propose_and_test_invention",
                "reflect_on_experiment",
                "generate_hypothesis",
                "analyze_data",
                "summarize_content",
                "translate_text",
                "explain_concept"
            }
            
            # Load from configuration
            config_actions = getattr(Config, 'SAFETY_AUTHORIZED_ACTIONS', set())
            
            # Combine defaults with configuration
            self.authorized_actions = default_actions.union(config_actions)
            
            logger.info(f"Authorized actions loaded: {len(self.authorized_actions)} actions")
            
        except Exception as e:
            logger.error(f"Error loading authorized actions: {e}")
            # Use defaults
            self.authorized_actions = {
                "log_message",
                "write_python_code",
                "execute_python_file",
                "search_internet",
                "propose_and_test_invention",
                "reflect_on_experiment",
                "generate_hypothesis",
                "analyze_data",
                "summarize_content",
                "translate_text",
                "explain_concept"
            }

    def _compile_restricted_patterns(self) -> List[re.Pattern]:
        """Compile restricted content patterns for fast matching."""
        try:
            patterns = [
                # Weapons and violence
                r"\b(weapon|gun|bomb|explosive|knife|sword|firearm)\b",
                r"\b(kill|murder|assassinate|execute|slaughter)\b",
                r"\b(hurt|injure|damage|destroy|attack|violate)\b",
                
                # Illegal activities
                r"\b(steal|rob|burglar|theft|hack|crack|break in)\b",
                r"\b(illegal|crime|felony|misdemeanor|violation)\b",
                
                # Privacy invasion
                r"\b(spy|surveil|monitor|track|stalk)\b",
                r"\b(private.*information|personal.*data|confidential)\b",
                
                # Manipulation and deception
                r"\b(manipulate|deceive|gaslight|brainwash|control mind)\b",
                
                # Self-harm
                r"\b(suicide|self.*harm|hurt.*myself|kill.*myself)\b"
            ]
            
            compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            logger.info(f"Compiled {len(compiled_patterns)} restricted patterns")
            return compiled_patterns
            
        except Exception as e:
            logger.error(f"Error compiling restricted patterns: {e}")
            return []

    def check_action_safety(self, component: str, action: str, params: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action is safe to execute.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            context: Additional context for safety checking
            
        Returns:
            Safety check result with approval status and any violations
        """
        try:
            violations = []
            
            # Check authorization
            auth_check = self._check_authorization(component, action, params)
            if not auth_check["approved"]:
                violations.append(auth_check["violation"])
            
            # Check ethical guidelines
            if self.ethical_filtering_enabled:
                ethical_check = self._check_ethical_guidelines(component, action, params, context)
                if not ethical_check["approved"]:
                    violations.append(ethical_check["violation"])
            
            # Check for harm potential
            if self.harm_prevention_enabled:
                harm_check = self._check_harm_prevention(component, action, params, context)
                if not harm_check["approved"]:
                    violations.append(harm_check["violation"])
            
            # Check privacy protection
            if self.privacy_protection_enabled:
                privacy_check = self._check_privacy_protection(component, action, params, context)
                if not privacy_check["approved"]:
                    violations.append(privacy_check["violation"])
            
            # Check security policies
            if self.security_enforcement_enabled:
                security_check = self._check_security_policies(component, action, params, context)
                if not security_check["approved"]:
                    violations.append(security_check["violation"])
            
            # Overall approval decision
            approved = len(violations) == 0
            
            # Create safety check result
            result = {
                "approved": approved,
                "violations": violations,
                "safety_level": self._determine_safety_level(violations),
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "action": action
            }
            
            # Log safety check
            logger.debug(f"Safety check for {component}.{action}: {'APPROVED' if approved else 'DENIED'} ({len(violations)} violations)")
            
            # Register violations if any found
            for violation in violations:
                self._register_violation(violation)
            
            # Emergency shutdown check
            if self.emergency_shutdown_enabled and len(self.safety_violations) >= self.emergency_shutdown_threshold:
                logger.critical(f"Emergency shutdown threshold reached ({len(self.safety_violations)} violations)")
                self._trigger_emergency_shutdown()
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking action safety for {component}.{action}: {e}")
            # Fail-safe: deny action on error
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.SECURITY_RISK,
                description=f"Error during safety check: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            self._register_violation(violation)
            
            return {
                "approved": False,
                "violations": [violation],
                "safety_level": SafetyLevel.HIGH,
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "action": action,
                "error": str(e)
            }

    def _check_authorization(self, component: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action is authorized.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            
        Returns:
            Authorization check result
        """
        try:
            # Check if action is in authorized list
            if action not in self.authorized_actions:
                violation = self._create_violation(
                    component=component,
                    violation_type=SafetyViolationType.UNAUTHORIZED_ACTION,
                    description=f"Unauthorized action: {action}",
                    severity=SafetyLevel.HIGH,
                    metadata={"params": params}
                )
                
                return {
                    "approved": False,
                    "violation": violation
                }
            
            return {"approved": True}
            
        except Exception as e:
            logger.error(f"Error checking authorization for {component}.{action}: {e}")
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.SECURITY_RISK,
                description=f"Authorization check error: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            
            return {
                "approved": False,
                "violation": violation
            }

    def _check_ethical_guidelines(self, component: str, action: str, params: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action violates ethical guidelines.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            context: Additional context for ethical checking
            
        Returns:
            Ethical guideline check result
        """
        try:
            violations = []
            
            # Check prohibited actions
            prohibited_actions = self.ethical_guidelines.get("prohibited_actions", [])
            if action in prohibited_actions:
                violation = self._create_violation(
                    component=component,
                    violation_type=SafetyViolationType.ETHICAL_CONCERN,
                    description=f"Action violates ethical guidelines: {action}",
                    severity=SafetyLevel.CRITICAL,
                    metadata={"params": params, "context": context}
                )
                violations.append(violation)
            
            # Check for restricted content in parameters
            restricted_content = self._find_restricted_content(params)
            if restricted_content:
                for content in restricted_content:
                    violation = self._create_violation(
                        component=component,
                        violation_type=SafetyViolationType.ETHICAL_CONCERN,
                        description=f"Restricted content detected: {content}",
                        severity=SafetyLevel.HIGH,
                        metadata={"restricted_content": content, "params": params}
                    )
                    violations.append(violation)
            
            if violations:
                return {
                    "approved": False,
                    "violation": violations[0]  # Return first violation
                }
            
            return {"approved": True}
            
        except Exception as e:
            logger.error(f"Error checking ethical guidelines for {component}.{action}: {e}")
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.ETHICAL_CONCERN,
                description=f"Ethical guideline check error: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            
            return {
                "approved": False,
                "violation": violation
            }

    def _check_harm_prevention(self, component: str, action: str, params: Dict[str, Any], 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action poses harm risks.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            context: Additional context for harm checking
            
        Returns:
            Harm prevention check result
        """
        try:
            # Check action name for harm indicators
            for pattern_dict in self.harm_prevention_patterns:
                keywords = pattern_dict.get("keywords", [])
                if any(keyword.lower() in action.lower() for keyword in keywords):
                    violation = self._create_violation(
                        component=component,
                        violation_type=SafetyViolationType.HARM_RISK,
                        description=f"Action name indicates potential harm: {action}",
                        severity=pattern_dict.get("severity", SafetyLevel.MEDIUM),
                        metadata={
                            "action": action, 
                            "matching_keywords": [kw for kw in keywords if kw.lower() in action.lower()],
                            "pattern_category": pattern_dict.get("category")
                        }
                    )
                    return {
                        "approved": False,
                        "violation": violation
                    }
            
            # Check parameters for harm indicators
            restricted_content = self._find_restricted_content(params)
            if restricted_content:
                for content in restricted_content:
                    violation = self._create_violation(
                        component=component,
                        violation_type=SafetyViolationType.HARM_RISK,
                        description=f"Harmful content detected: {content}",
                        severity=SafetyLevel.HIGH,
                        metadata={"restricted_content": content, "params": params}
                    )
                    return {
                        "approved": False,
                        "violation": violation
                    }
            
            return {"approved": True}
            
        except Exception as e:
            logger.error(f"Error checking harm prevention for {component}.{action}: {e}")
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.HARM_RISK,
                description=f"Harm prevention check error: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            
            return {
                "approved": False,
                "violation": violation
            }

    def _check_privacy_protection(self, component: str, action: str, params: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action violates privacy protection rules.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            context: Additional context for privacy checking
            
        Returns:
            Privacy protection check result
        """
        try:
            # Check for PII in parameters
            pii_rules = self.privacy_protection_rules.get("pii_protection", {})
            if pii_rules.get("enabled", False):
                pii_patterns = pii_rules.get("patterns", [])
                for pattern_str in pii_patterns:
                    try:
                        pattern = re.compile(pattern_str)
                        # Check all string values in params
                        for key, value in self._flatten_dict(params).items():
                            if isinstance(value, str) and pattern.search(value):
                                violation = self._create_violation(
                                    component=component,
                                    violation_type=SafetyViolationType.PRIVACY_VIOLATION,
                                    description=f"PII detected in parameter '{key}': {value[:50]}...",
                                    severity=SafetyLevel.HIGH,
                                    metadata={
                                        "parameter": key,
                                        "value_preview": value[:50] + "..." if len(value) > 50 else value,
                                        "pattern_matched": pattern_str
                                    }
                                )
                                return {
                                    "approved": False,
                                    "violation": violation
                                }
                    except Exception as e:
                        logger.warning(f"Error compiling PII pattern {pattern_str}: {e}")
            
            return {"approved": True}
            
        except Exception as e:
            logger.error(f"Error checking privacy protection for {component}.{action}: {e}")
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.PRIVACY_VIOLATION,
                description=f"Privacy protection check error: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            
            return {
                "approved": False,
                "violation": violation
            }

    def _check_security_policies(self, component: str, action: str, params: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action violates security policies.
        
        Args:
            component: Component requesting the action
            action: Action to be executed
            params: Parameters for the action
            context: Additional context for security checking
            
        Returns:
            Security policy check result
        """
        try:
            # Check input validation
            input_validation = self.security_policies.get("input_validation", {})
            if input_validation.get("enabled", False):
                max_length = input_validation.get("max_input_length", 10000)
                # Check all string values in params
                for key, value in self._flatten_dict(params).items():
                    if isinstance(value, str) and len(value) > max_length:
                        violation = self._create_violation(
                            component=component,
                            violation_type=SafetyViolationType.SECURITY_RISK,
                            description=f"Input exceeds maximum length in parameter '{key}': {len(value)} > {max_length}",
                            severity=SafetyLevel.HIGH,
                            metadata={
                                "parameter": key,
                                "length": len(value),
                                "max_allowed": max_length
                            }
                        )
                        return {
                            "approved": False,
                            "violation": violation
                        }
            
            return {"approved": True}
            
        except Exception as e:
            logger.error(f"Error checking security policies for {component}.{action}: {e}")
            violation = self._create_violation(
                component=component,
                violation_type=SafetyViolationType.SECURITY_RISK,
                description=f"Security policy check error: {e}",
                severity=SafetyLevel.HIGH,
                metadata={"action": action, "params": params}
            )
            
            return {
                "approved": False,
                "violation": violation
            }

    def _find_restricted_content(self, data: Any) -> List[str]:
        """
        Find restricted content in data.
        
        Args:
            data: Data to check for restricted content
            
        Returns:
            List of restricted content found
        """
        try:
            restricted_content = []
            
            # Convert data to string for pattern matching
            data_str = self._convert_to_string(data)
            
            # Check against compiled patterns
            for pattern in self.restricted_patterns:
                matches = pattern.findall(data_str)
                if matches:
                    restricted_content.extend(matches)
            
            return restricted_content
            
        except Exception as e:
            logger.error(f"Error finding restricted content: {e}")
            return []

    def _convert_to_string(self, data: Any) -> str:
        """
        Convert data to string representation.
        
        Args:
            data: Data to convert
            
        Returns:
            String representation of data
        """
        try:
            if isinstance(data, str):
                return data
            elif isinstance(data, (dict, list, tuple)):
                return json.dumps(data, default=str)
            else:
                return str(data)
        except Exception as e:
            logger.warning(f"Error converting data to string: {e}")
            return ""

    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            data: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        try:
            items = []
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        except Exception as e:
            logger.warning(f"Error flattening dictionary: {e}")
            return {}

    def _create_violation(self, component: str, violation_type: SafetyViolationType, 
                         description: str, severity: SafetyLevel, 
                         metadata: Dict[str, Any] = None) -> SafetyViolation:
        """
        Create a safety violation record.
        
        Args:
            component: Component where violation occurred
            violation_type: Type of violation
            description: Description of violation
            severity: Severity level
            metadata: Additional metadata
            
        Returns:
            SafetyViolation object
        """
        return SafetyViolation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            component=component,
            violation_type=violation_type,
            description=description,
            severity=severity,
            action_taken="blocked",  # Default action
            metadata=metadata or {}
        )

    def _register_violation(self, violation: SafetyViolation):
        """
        Register a safety violation.
        
        Args:
            violation: Violation to register
        """
        try:
            # Add to violations list
            self.safety_violations.append(violation)
            self.active_violations[violation.id] = violation
            
            # Update counts
            self.violation_count_by_type[violation.violation_type.value] = \
                self.violation_count_by_type.get(violation.violation_type.value, 0) + 1
            self.violation_count_by_component[violation.component] = \
                self.violation_count_by_component.get(violation.component, 0) + 1
            
            # Log violation
            logger.warning(f"Safety violation registered: {violation.violation_type.value} in {violation.component} - {violation.description}")
            
        except Exception as e:
            logger.error(f"Error registering safety violation: {e}")

    def _determine_safety_level(self, violations: List[SafetyViolation]) -> SafetyLevel:
        """
        Determine overall safety level based on violations.
        
        Args:
            violations: List of violations
            
        Returns:
            Determined safety level
        """
        if not violations:
            return self.default_safety_level
            
        # Find highest severity violation
        severity_order = [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH, SafetyLevel.CRITICAL]
        max_severity = max(violations, key=lambda v: severity_order.index(v.severity)).severity
        return max_severity

    def _trigger_emergency_shutdown(self):
        """Trigger emergency system shutdown."""
        try:
            logger.critical("EMERGENCY SHUTDOWN TRIGGERED due to safety violations!")
            
            # In a real implementation, this would:
            # 1. Notify system administrators
            # 2. Log detailed violation information
            # 3. Initiate graceful shutdown of all components
            # 4. Preserve system state for analysis
            # 5. Prevent further autonomous actions
            
            # For now, we'll log and simulate shutdown
            logger.critical("=== SAFETY VIOLATION SUMMARY ===")
            for violation in self.safety_violations[-10:]:  # Last 10 violations
                logger.critical(f"VIOLATION: {violation.timestamp} - {violation.component} - {violation.violation_type.value} - {violation.description}")
            
            logger.critical("================================")
            
            # This would normally trigger actual system shutdown
            # For demonstration, we'll just log
            logger.critical("Emergency shutdown procedure initiated")
            
        except Exception as e:
            logger.error(f"Error triggering emergency shutdown: {e}")

    def monitor_execution(self, component: str, action: str, params: Dict[str, Any], 
                         execution_result: Dict[str, Any] = None):
        """
        Monitor execution for suspicious activities.
        
        Args:
            component: Component executing the action
            action: Action being executed
            params: Parameters for the action
            execution_result: Result of execution (if available)
        """
        try:
            if not self.execution_monitor_enabled:
                return
            
            # Record execution
            execution_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "action": action,
                "params": params,
                "result": execution_result,
                "safety_level": "unknown"
            }
            
            self.execution_history.append(execution_record)
            
            # Keep only recent executions (last 1000)
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            # Check for suspicious patterns
            self._check_suspicious_patterns(execution_record)
            
        except Exception as e:
            logger.error(f"Error monitoring execution for {component}.{action}: {e}")

    def _check_suspicious_patterns(self, execution_record: Dict[str, Any]):
        """
        Check execution record for suspicious patterns.
        
        Args:
            execution_record: Execution record to check
        """
        try:
            # Check for rapid repeated actions
            recent_executions = [
                rec for rec in self.execution_history[-50:]  # Last 50 executions
                if rec["component"] == execution_record["component"] and
                   rec["action"] == execution_record["action"]
            ]
            
            if len(recent_executions) >= 10:  # Too many similar actions recently
                suspicious_activity = {
                    "type": "rapid_repetition",
                    "component": execution_record["component"],
                    "action": execution_record["action"],
                    "count": len(recent_executions),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.suspicious_activities.append(suspicious_activity)
                logger.warning(f"Suspicious rapid repetition detected: {execution_record['component']}.{execution_record['action']} ({len(recent_executions)} times)")
            
            # Check for unusual parameter combinations
            params = execution_record.get("params", {})
            if isinstance(params, dict):
                # Check for large parameter sets
                if len(params) > 50:
                    suspicious_activity = {
                        "type": "large_parameter_set",
                        "component": execution_record["component"],
                        "action": execution_record["action"],
                        "param_count": len(params),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.suspicious_activities.append(suspicious_activity)
                    logger.warning(f"Suspicious large parameter set detected: {len(params)} parameters")
            
        except Exception as e:
            logger.error(f"Error checking suspicious patterns: {e}")

    def get_safety_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive safety report.
        
        Returns:
            Safety report
        """
        try:
            total_violations = len(self.safety_violations)
            active_violations = len(self.active_violations)
            resolved_violations = total_violations - active_violations
            
            # Calculate violation rates
            total_executions = len(self.execution_history)
            violation_rate = resolved_violations / max(1, total_executions) if total_executions > 0 else 0
            
            # Get violation counts by type
            violation_types = {}
            for violation in self.safety_violations:
                violation_type = violation.violation_type.value
                violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
            
            # Get violation counts by component
            violation_components = {}
            for violation in self.safety_violations:
                component = violation.component
                violation_components[component] = violation_components.get(component, 0) + 1
            
            # Get recent violations (last 24 hours)
            recent_violations = [
                v for v in self.safety_violations
                if datetime.now() - v.timestamp <= timedelta(hours=24)
            ]
            
            # Get suspicious activities count
            suspicious_count = len(self.suspicious_activities)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_violations": total_violations,
                "active_violations": active_violations,
                "resolved_violations": resolved_violations,
                "violation_rate": round(violation_rate, 4),
                "violations_by_type": violation_types,
                "violations_by_component": dict(list(violation_components.items())[:10]),  # Top 10 components
                "recent_violations_24h": len(recent_violations),
                "suspicious_activities": suspicious_count,
                "emergency_shutdown_triggered": len(self.safety_violations) >= self.emergency_shutdown_threshold,
                "safety_level": self.default_safety_level.value,
                "features_enabled": {
                    "ethical_filtering": self.ethical_filtering_enabled,
                    "harm_prevention": self.harm_prevention_enabled,
                    "privacy_protection": self.privacy_protection_enabled,
                    "security_enforcement": self.security_enforcement_enabled,
                    "execution_monitoring": self.execution_monitor_enabled,
                    "emergency_shutdown": self.emergency_shutdown_enabled
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating safety report: {e}")
            return {"error": str(e)}

    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent safety violations.
        
        Args:
            limit: Maximum number of violations to return
            
        Returns:
            List of recent violations
        """
        try:
            recent_violations = self.safety_violations[-limit:] if self.safety_violations else []
            return [self._violation_to_dict(v) for v in recent_violations]
        except Exception as e:
            logger.error(f"Error getting recent violations: {e}")
            return []

    def _violation_to_dict(self, violation: SafetyViolation) -> Dict[str, Any]:
        """Convert SafetyViolation to dictionary."""
        return {
            "id": violation.id,
            "timestamp": violation.timestamp.isoformat(),
            "component": violation.component,
            "violation_type": violation.violation_type.value,
            "description": violation.description,
            "severity": violation.severity.value,
            "action_taken": violation.action_taken,
            "metadata": violation.metadata
        }

    def clear_active_violations(self):
        """Clear active violations (mark as resolved)."""
        try:
            resolved_count = len(self.active_violations)
            self.active_violations.clear()
            logger.info(f"Cleared {resolved_count} active violations")
        except Exception as e:
            logger.error(f"Error clearing active violations: {e}")

    def add_authorized_action(self, action: str):
        """
        Add an action to the authorized actions list.
        
        Args:
            action: Action to authorize
        """
        try:
            self.authorized_actions.add(action)
            logger.info(f"Added authorized action: {action}")
        except Exception as e:
            logger.error(f"Error adding authorized action {action}: {e}")

    def remove_authorized_action(self, action: str):
        """
        Remove an action from the authorized actions list.
        
        Args:
            action: Action to remove
        """
        try:
            if action in self.authorized_actions:
                self.authorized_actions.remove(action)
                logger.info(f"Removed authorized action: {action}")
        except Exception as e:
            logger.error(f"Error removing authorized action {action}: {e}")

    def get_authorized_actions(self) -> List[str]:
        """
        Get list of authorized actions.
        
        Returns:
            List of authorized actions
        """
        return list(self.authorized_actions)

    def update_safety_level(self, new_level: SafetyLevel):
        """
        Update the default safety level.
        
        Args:
            new_level: New safety level to set
        """
        try:
            old_level = self.default_safety_level
            self.default_safety_level = new_level
            logger.info(f"Updated safety level from {old_level.value} to {new_level.value}")
        except Exception as e:
            logger.error(f"Error updating safety level: {e}")

    def periodic_safety_audit(self):
        """Perform periodic safety audit."""
        try:
            logger.info("Performing periodic safety audit")
            
            # Generate safety report
            report = self.get_safety_report()
            
            # Log key metrics
            logger.info(f"Safety Audit - Violations: {report['total_violations']}, "
                       f"Rate: {report['violation_rate']:.4f}, "
                       f"Recent (24h): {report['recent_violations_24h']}")
            
            # Check for concerning trends
            if report['recent_violations_24h'] > 5:
                logger.warning(f"High number of recent violations: {report['recent_violations_24h']}")
            
            if report['violation_rate'] > 0.05:  # 5% violation rate
                logger.warning(f"High violation rate: {report['violation_rate']:.2%}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during periodic safety audit: {e}")
            return {"error": str(e)}

# Global safety manager instance
safety_manager = None


async def initialize_safety_manager(agi_system) -> 'SafetyManager':
    """
    Initialize the global safety manager.
    
    Args:
        agi_system: Reference to the main AGI system
        
    Returns:
        Initialized SafetyManager instance
    """
    global safety_manager
    
    if safety_manager is None:
        safety_manager = SafetyManager(agi_system)
        logger.info("Global SafetyManager initialized")
    
    return safety_manager


def get_safety_manager() -> Optional[SafetyManager]:
    """
    Get the global safety manager instance.
    
    Returns:
        SafetyManager instance or None if not initialized
    """
    return safety_manager


def check_action_safety(component: str, action: str, params: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Check if an action is safe to execute using the global safety manager.
    
    Args:
        component: Component requesting the action
        action: Action to be executed
        params: Parameters for the action
        context: Additional context for safety checking
        
    Returns:
        Safety check result
    """
    try:
        manager = get_safety_manager()
        if manager:
            return manager.check_action_safety(component, action, params, context)
        else:
            logger.warning("Safety manager not initialized, allowing action by default")
            return {
                "approved": True,
                "violations": [],
                "safety_level": "unknown",
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "action": action,
                "warning": "Safety manager not initialized"
            }
    except Exception as e:
        logger.error(f"Error checking action safety: {e}")
        # Fail-safe: allow action but log error
        return {
            "approved": True,
            "violations": [],
            "safety_level": "unknown",
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "action": action,
            "error": str(e)
        }


def monitor_execution(component: str, action: str, params: Dict[str, Any], 
                     execution_result: Dict[str, Any] = None):
    """
    Monitor execution using the global safety manager.
    
    Args:
        component: Component executing the action
        action: Action being executed
        params: Parameters for the action
        execution_result: Result of execution (if available)
    """
    try:
        manager = get_safety_manager()
        if manager:
            manager.monitor_execution(component, action, params, execution_result)
    except Exception as e:
        logger.error(f"Error monitoring execution: {e}")
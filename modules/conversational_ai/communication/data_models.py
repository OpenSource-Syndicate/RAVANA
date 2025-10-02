from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


class CommunicationType(Enum):
    """Types of communication messages."""
    PROPOSAL = "proposal"
    STATUS_UPDATE = "status_update"
    EMERGENCY = "emergency"
    THOUGHT_EXCHANGE = "thought_exchange"
    TASK_RESULT = "task_result"
    NOTIFICATION = "notification"
    USER_MESSAGE = "user_message"
    LEARNING_EXPERIENCE = "learning_experience"
    IMPROVEMENT_OPPORTUNITY = "improvement_opportunity"


class Priority(Enum):
    """Message priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommunicationMessage:
    """Represents a communication message between systems."""
    id: str
    type: CommunicationType
    priority: Priority
    timestamp: datetime
    sender: str
    recipient: str
    subject: str
    content: Dict[str, Any]
    requires_response: bool = False
    response_timeout: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = asdict(self)
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationMessage':
        """Create message from dictionary."""
        data['type'] = CommunicationType(data['type'])
        data['priority'] = Priority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class UserPlatformProfile:
    """Tracks user platform preferences."""
    user_id: str
    last_platform: str  # discord, telegram
    platform_user_id: str
    preferences: Dict[str, Any]
    last_interaction: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        result = asdict(self)
        result['last_interaction'] = self.last_interaction.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPlatformProfile':
        """Create profile from dictionary."""
        data['last_interaction'] = datetime.fromisoformat(
            data['last_interaction'])
        return cls(**data)

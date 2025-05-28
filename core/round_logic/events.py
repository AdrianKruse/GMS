"""
Events module defining the possible events in the core round logic.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Any


class Event:
    """Base class for all events in the core logic."""
    pass


@dataclass
class AgentMovedEvent(Event):
    """
    Event indicating the agent has moved to a new position.
    """
    position: Tuple[int, int]


@dataclass
class AgentDamagedEvent(Event):
    """
    Event indicating the agent took damage.
    """
    damage: int
    health_remaining: int


@dataclass
class TowerDamagedEvent(Event):
    """
    Event indicating a tower took damage.
    """
    tower_id: str
    damage: int
    health_remaining: int


@dataclass
class TowerDestroyedEvent(Event):
    """
    Event indicating a tower was destroyed.
    """
    tower_id: str


@dataclass
class ProjectileCreatedEvent(Event):
    """
    Event indicating a new projectile was created.
    """
    position: Tuple[float, float]
    direction: Tuple[float, float]


@dataclass
class ProjectileRemovedEvent(Event):
    """
    Event indicating a projectile was removed.
    """
    position: Tuple[float, float]


@dataclass
class RoundOverEvent(Event):
    """
    Event indicating the round is over.
    """
    agent_survived: bool  # True if agent survived, False if agent died 
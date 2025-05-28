from dataclasses import dataclass
from typing import Optional, Tuple, Any


class AgentAction:
    """Base class for all player actions."""
    pass


@dataclass
class MoveTo(AgentAction):
    """
    Move Agent to x,y - a planning style action.
    The agent will continue to move towards this position over multiple ticks.
    """
    x: int
    y: int


@dataclass
class Attack(AgentAction):
    """
    Agent attacks a target (building/enemy) with ID.
    The agent will continue attacking until the target is destroyed or another action is issued.
    """
    target_id: Any  # ID of the target to attack


@dataclass
class Stand(AgentAction):
    """
    Agent stands still for one tick.
    This is a one-step action.
    """
    pass


@dataclass
class Resume(AgentAction):
    """
    Agent continues with the current planning-style action.
    This is a one-step action that tells the agent to keep following its current plan.
    """
    pass


from dataclasses import dataclass
from typing import Optional, Tuple


class AgentAction:
    """Base class for all player actions."""
    pass

@dataclass
class MoveTo(AgentAction):
    """Move Agent to x,y"""
    x: int
    y: int


@dataclass
class Stand(AgentAction):
    """Agent Stands Still"""
    pass


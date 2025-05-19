from dataclasses import dataclass
from typing import Optional, Tuple


class Action:
    """Base class for all player actions."""
    pass

@dataclass
class StartRound(Action):
    """Start moving the block in the current facing direction."""
    pass


@dataclass
class Quit(Action):
    """Exit the game."""
    pass


@dataclass
class ErrorAction(Action):
    """Represents an invalid command."""
    message: str


@dataclass
class PlaceTower(Action):
    """Place a tower at the specified position."""
    x: int
    y: int
    direction: int


@dataclass
class ShowStats(Action):
    """Show standard game statistics in the info panel."""
    pass


@dataclass
class ShowTowers(Action):
    """Show tower information in the info panel."""
    pass


@dataclass
class ShowPosition(Action):
    """Show information about a specific position."""
    x: int
    y: int 

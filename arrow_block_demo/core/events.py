from dataclasses import dataclass
from typing import Tuple, Optional


class Event:
    """Base class for all system events."""
    pass


@dataclass
class PosChanged(Event):
    """Agent position has changed."""
    x: int
    y: int
    
    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class RoundEnded(Event):
    """The current round has ended."""
    pass


@dataclass
class CurrencyDelta(Event):
    """Currency amount has changed."""
    amount: int 

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Projectile:
    """
    Represents a projectile on the grid
    """
    position: Tuple[float, float]  # (x, y)
    direction: Tuple[float, float]


@dataclass
class RoundState:
    """
    RoundState holds dynamic objects that reset at the end of each round.
    """
    agent_pos: Tuple[int, int]  # (x, y)
    agent_health: int = 100
    is_moving: bool = False
    tick_index: int = 0  # Used for deterministic replay
    projectiles: List[Projectile] = field(default_factory=list)

    @property
    def x(self) -> int:
        return self.agent_pos[0]

    @property
    def y(self) -> int:
        return self.agent_pos[1]

    @property
    def get_projectiles(self) -> List[Projectile]:
        return self.projectiles


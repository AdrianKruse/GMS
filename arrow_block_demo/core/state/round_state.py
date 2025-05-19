from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
from ..agent_actions import AgentAction, MoveTo, Attack


@dataclass
class Projectile:
    """
    Represents a projectile on the grid
    """
    position: Tuple[float, float]  # (x, y)
    direction: Tuple[float, float]


@dataclass
class PlanningAction:
    """
    Represents a long-term planning action the agent is currently executing.
    This could be moving to a position or attacking a target.
    """
    action_type: str  # "move_to" or "attack"
    target_pos: Optional[Tuple[int, int]] = None  # For MoveTo actions
    target_id: Optional[Any] = None  # For Attack actions
    path: List[Tuple[int, int]] = field(default_factory=list)  # Path to follow for MoveTo
    completed: bool = False  # Whether the action has been completed


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
    current_plan: Optional[PlanningAction] = None  # Current planning-style action

    @property
    def x(self) -> int:
        return self.agent_pos[0]

    @property
    def y(self) -> int:
        return self.agent_pos[1]

    @property
    def get_projectiles(self) -> List[Projectile]:
        return self.projectiles
        
    def set_move_plan(self, target_x: int, target_y: int, path: List[Tuple[int, int]] = None) -> None:
        """Set the agent's movement plan to move to a target position."""
        if path is None:
            # Default to direct path if none provided
            path = [self.agent_pos, (target_x, target_y)]
            
        self.current_plan = PlanningAction(
            action_type="move_to",
            target_pos=(target_x, target_y),
            path=path
        )
        
    def set_attack_plan(self, target_id: Any) -> None:
        """Set the agent's plan to attack a specific target."""
        self.current_plan = PlanningAction(
            action_type="attack",
            target_id=target_id
        )
        
    def clear_plan(self) -> None:
        """Clear the agent's current plan."""
        self.current_plan = None


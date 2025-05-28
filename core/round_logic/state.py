"""
State module containing the unified core state representation for round logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from enum import Enum
import uuid
from core.round_logic.actions import MOVE_ACTION
import logging
import copy
import random
import math

# Configure logging
logger = logging.getLogger('core')


class BlockType(Enum):
    """Represents the type of block on the grid."""
    EMPTY = 0
    WALL = 1
    TOWER = 2
    START = 3


@dataclass
class Tower:
    """
    Tower entity in the game.
    Completely self-contained for core round logic.
    """
    position: Tuple[int, int]  # (x, y)
    direction: Tuple[float, float]
    health: int = 100
    rate: int = 8  # Rate of fire - ticks between shots
    tick: int = 0  # Current tick counter
    tower_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    @property
    def is_destroyed(self) -> bool:
        return self.health <= 0


@dataclass
class Projectile:
    """
    Projectile entity in the game.
    """
    position: Tuple[float, float]  # (x, y) as floats for smooth movement
    direction: Tuple[float, float]  # Direction vector


@dataclass
class RoundState:
    """
    Unified state representation for core round logic.
    Contains everything needed to process a round.
    """
    # Grid properties
    grid_width: int
    grid_height: int
    grid_layout: List[List[BlockType]]
    
    # Dynamic elements
    towers: List[Tower] = field(default_factory=list)
    projectiles: List[Projectile] = field(default_factory=list)
    
    current_active_directive: Optional[Dict[str, Any]] = None
    last_interrupted_directive: Optional[Dict[str, Any]] = None
    position: Tuple[int, int] = (0, 0)
    health: int = 100
    
    # Game progress
    tick_index: int = 0

    @property
    def is_moving(self) -> bool:
        """Check if the agent is moving."""
        return self.current_active_directive is not None and self.current_active_directive['action_type'] == MOVE_ACTION
    
    @property
    def is_round_over(self) -> bool:
        """Check if the round is over."""
        # Round ends when agent health reaches zero or all towers are destroyed
        return self.health <= 0 or all(tower.health <= 0 for tower in self.towers)
               
    
    def is_position_valid(self, x: int, y: int) -> bool:
        """Check if a position is valid for movement."""
        # Check if within grid bounds
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            logger.debug(f"Position ({x},{y}) is out of bounds")
            return False
        
        # Check grid layout first (walls take priority)
        try:
            block_type = self.grid_layout[y][x]
            if block_type == BlockType.WALL:
                logger.debug(f"Position ({x},{y}) is a wall")
                return False
        except IndexError:
            logger.error(f"Grid layout index error for position ({x},{y}) - grid size is {self.grid_width}x{self.grid_height}")
            return False
        except Exception as e:
            logger.error(f"Error checking grid layout for position ({x},{y}): {e}")
            return False
        
        # Check if there's a tower at this position
        for tower in self.towers:
            if tower.health > 0 and tower.position == (x, y):
                logger.debug(f"Position ({x},{y}) contains a non-destroyed tower {tower.tower_id}")
                return False  # Can't move through towers
            
        # Position is valid if it's within bounds, not a wall, and contains no non-destroyed towers
        logger.debug(f"Position ({x},{y}) is valid for movement")
        return True
    
    def get_tower_by_id(self, tower_id: str) -> Optional[Tower]:
        """Get a tower by its ID."""
        for tower in self.towers:
            if tower.tower_id == tower_id:
                return tower
        return None

    def transform(
        self,
        rotate_quarters: int = 0,
        flip_h: bool = False,
        flip_v: bool = False
    ) -> "RoundState":
        """
        Return a deep-copied RoundState rotated by `rotate_quarters` × 90° clockwise
        and optionally flipped horizontally / vertically.
        """
        k = rotate_quarters & 3
        src_w, src_h = self.grid_width, self.grid_height
        dst_w, dst_h = (src_w, src_h) if k in (0, 2) else (src_h, src_w)

        new = copy.deepcopy(self)
        new.grid_width, new.grid_height = dst_w, dst_h

        # --- grid layout -------------------------------------------------
        new_grid = [[BlockType.EMPTY for _ in range(dst_w)] for _ in range(dst_h)]
        for y in range(src_h):
            for x in range(src_w):
                xx, yy = _rot_pt(x, y, src_w, src_h, k)
                xx, yy = _flip_pt(xx, yy, dst_w, dst_h, flip_h, flip_v)
                new_grid[yy][xx] = self.grid_layout[y][x]
        new.grid_layout = new_grid

        # --- agent position ---------------------------------------------
        ax, ay = _rot_pt(*self.position, src_w, src_h, k)
        ax, ay = _flip_pt(ax, ay, dst_w, dst_h, flip_h, flip_v)
        new.position = (ax, ay)

        # --- towers ------------------------------------------------------
        for tw in new.towers:
            x, y   = _rot_pt(*tw.position, src_w, src_h, k)
            x, y   = _flip_pt(x, y, dst_w, dst_h, flip_h, flip_v)
            dx, dy = _rot_vec(*tw.direction, k)
            dx, dy = _flip_vec(dx, dy, flip_h, flip_v)
            tw.position  = (x, y)
            tw.direction = (dx, dy)

        # --- projectiles -------------------------------------------------
        for pr in new.projectiles:
            x, y   = _rot_pt(int(round(pr.position[0])), int(round(pr.position[1])),
                            src_w, src_h, k)
            x, y   = _flip_pt(x, y, dst_w, dst_h, flip_h, flip_v)
            dx, dy = _rot_vec(*pr.direction, k)
            dx, dy = _flip_vec(dx, dy, flip_h, flip_v)
            pr.position  = (float(x), float(y))
            pr.direction = (dx, dy)

        return new


    def random_transform(self) -> "RoundState":
        """Random 0/90/180/270 rotation plus random H/V flips."""
        return self.transform(
            rotate_quarters=random.randint(0, 3),
            flip_h=random.choice([False, True]),
            flip_v=random.choice([False, True]),
        )


@staticmethod
def _rot_pt(x: int, y: int, w: int, h: int, k: int) -> Tuple[int, int]:
    """Rotate (x,y) around the grid origin by k quarter-turns clockwise."""
    k %= 4
    if k == 0:   # 0°
        return x, y
    if k == 1:   # 90°
        return h - 1 - y, x
    if k == 2:   # 180°
        return w - 1 - x, h - 1 - y
    if k == 3:   # 270°
        return y, w - 1 - x

@staticmethod
def _rot_vec(dx: float, dy: float, k: int) -> Tuple[float, float]:
    """
    Rotate a direction vector by k quarter-turns clockwise
    in screen coordinates (x right, y down).
    """
    k &= 3
    if k == 0:           # 0°
        return dx, dy
    if k == 1:           # 90° CW  →  (x',y') = (-y,  x)
        return -dy, dx
    if k == 2:           # 180°    →  (-x,-y)
        return -dx, -dy
    # k == 3            # 270° CW  →  (y', -x)
    return  dy, -dx


@staticmethod
def _flip_pt(x: int, y: int, w: int, h: int,
                flip_h: bool, flip_v: bool) -> Tuple[int, int]:
    if flip_h:
        x = w - 1 - x
    if flip_v:
        y = h - 1 - y
    return x, y

@staticmethod
def _flip_vec(dx: float, dy: float, flip_h: bool, flip_v: bool):
    if flip_h:
        dx = -dx
    if flip_v:
        dy = -dy
    return dx, dy

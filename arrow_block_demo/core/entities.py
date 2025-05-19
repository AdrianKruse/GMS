from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
from enum import Enum, auto
from .agent_actions import (
    AgentAction, MoveTo, Stand, Attack, Resume
)
import random
import logging
from core.systems import pathfinding

logger = logging.getLogger('game')


class EntityType(Enum):
    """Types of entities in the game"""
    AGENT = auto()
    FREE = auto()
    BLOCKED = auto()
    TOWER = auto()


@dataclass
class Entity:
    """Base class for all entities in the game"""
    entity_type: EntityType
    sprite_name: str
    passable: bool = True
    
    def get_sprite_name(self) -> str:
        """Get the sprite name for this entity"""
        return self.sprite_name


# Placeholder for AgentState, to be expanded for agent perception and decision-making
class AgentState:
    pass

# Use a regular class instead of dataclass inheritance
class Agent:
    """The player-controlled block"""
    def __init__(self, pos: Tuple[int, int]):
        self.entity_type = EntityType.AGENT
        self.sprite_name = "agent"
        self.passable = False
        self.pos = pos
        self.target_tower = None
        self.is_attacking = False
        self.attack_cooldown = 0
        self.target_adjacent_pos = None
    
    def get_sprite_name(self) -> str:
        """Get the sprite name for this entity"""
        return self.sprite_name

    def action(self, game_state, round_state) -> AgentAction:
        """
        Determine the agent's next action.
        This now implements AI behavior to automatically find towers, navigate to them, and attack.
        """
        # Update position in case it changed
        self.pos = round_state.agent_pos
        
        # If we're already executing a plan, continue with it
        if round_state.current_plan and not round_state.current_plan.completed:
            return Resume()
        
        # If we're attacking a tower, check if it's still valid and continue the attack
        if self.is_attacking and self.target_tower:
            # Check if we're still adjacent to the tower
            tower = None
            for t in game_state.towers:
                if t.tower_id == self.target_tower:
                    tower = t
                    break
                    
            if tower and self._is_adjacent_to(tower.position):
                # Continue attacking if tower not destroyed
                if not tower.is_destroyed:
                    return Attack(target_id=tower.tower_id)
                else:
                    # Tower destroyed, clear state
                    self.is_attacking = False
                    self.target_tower = None
                    self.target_adjacent_pos = None
                    logger.debug(f"Tower destroyed, looking for new target")
            else:
                # No longer adjacent to tower or tower doesn't exist
                self.is_attacking = False
                self.target_tower = None
                self.target_adjacent_pos = None
        
        # Find the nearest non-destroyed tower
        nearest_tower = self._find_nearest_tower(game_state)
        
        if nearest_tower:
            # If adjacent to tower, attack it
            if self._is_adjacent_to(nearest_tower.position):
                logger.debug(f"Agent attacking tower at {nearest_tower.position}")
                self.is_attacking = True
                self.target_tower = nearest_tower.tower_id
                return Attack(target_id=nearest_tower.tower_id)
            
            # Find a valid adjacent position to the tower if we don't have one
            if not self.target_adjacent_pos or not game_state.is_position_valid(*self.target_adjacent_pos):
                self.target_adjacent_pos = self._find_valid_adjacent_position(game_state, nearest_tower.position)
                if not self.target_adjacent_pos:
                    logger.debug(f"No valid adjacent positions to tower at {nearest_tower.position}")
                    return Stand()
            
            # Move toward the adjacent position
            logger.debug(f"Agent moving to position {self.target_adjacent_pos} adjacent to tower at {nearest_tower.position}")
            return MoveTo(x=self.target_adjacent_pos[0], y=self.target_adjacent_pos[1])
        
        # No towers to attack, just stand
        return Stand()
    
    def _find_nearest_tower(self, game_state):
        """Find the nearest non-destroyed tower"""
        nearest_tower = None
        min_distance = float('inf')
        
        for tower in game_state.towers:
            if tower.is_destroyed:
                continue
                
            dx = abs(tower.position[0] - self.pos[0])
            dy = abs(tower.position[1] - self.pos[1])
            distance = dx + dy  # Manhattan distance
            
            if distance < min_distance:
                min_distance = distance
                nearest_tower = tower
                
        return nearest_tower
    
    def _is_adjacent_to(self, pos: Tuple[int, int]) -> bool:
        """Check if the agent is adjacent to the given position"""
        dx = abs(pos[0] - self.pos[0])
        dy = abs(pos[1] - self.pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _find_valid_adjacent_position(self, game_state, tower_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find a valid position adjacent to the tower"""
        x, y = tower_pos
        # Check all adjacent positions in a fixed order for consistency
        paths = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_x, adj_y = x + dx, y + dy
            if game_state.is_position_valid(adj_x, adj_y):
                path = pathfinding.astar(game_state, (self.pos[0], self.pos[1]), (adj_x, adj_y))
                if path:
                    paths.append((len(path), (adj_x, adj_y)))
        if paths:
            return sorted(paths)[0][1]
        return None


@dataclass
class MapBlock(Entity):
    """A static block on the map"""
    def __init__(self, entity_type: EntityType, sprite_name: str, passable: bool = True):
        super().__init__(
            entity_type=entity_type,
            sprite_name=sprite_name,
            passable=passable
        )


# Entity type registry
ENTITY_TYPES = {
    "free": MapBlock(EntityType.FREE, "free", True),
    "blocked": MapBlock(EntityType.BLOCKED, "blocked", False),
    "tower": MapBlock(EntityType.TOWER, "tower", False),
}


def get_entity_by_name(name: str) -> Optional[Entity]:
    """Get an entity from the registry by name"""
    return ENTITY_TYPES.get(name) 

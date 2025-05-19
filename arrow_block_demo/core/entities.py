from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
from enum import Enum, auto
from .agent_actions import (
    AgentAction, MoveTo, Stand
)
import random


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
    
    def get_sprite_name(self) -> str:
        """Get the sprite name for this entity"""
        return self.sprite_name

    def action(self, agentState: AgentState) -> AgentAction:
        x, y = self.pos
        r = random.randint(1,4)
        if r == 1:
            return MoveTo(x=x, y=y-1)
        elif r == 2:
            return MoveTo(x=x+1, y=y)
        elif r == 3:
            return MoveTo(x=x, y=y+1)
        elif r == 4:
            return MoveTo(x=x-1, y=y)
        return Stand()


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

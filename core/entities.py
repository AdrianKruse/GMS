from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
import logging


logger = logging.getLogger('game')


class EntityType(Enum):
    """Types of entities in the game"""
    AGENT = 1
    TOWER = 2
    PROJECTILE = 3
    MAP_ELEMENT = 4
    FREE = 5
    BLOCKED = 6
    START_POSITION = 7


@dataclass
class Entity:
    """Base class for all entities in the game"""
    entity_type: EntityType
    sprite_name: str
    passable: bool = True
    
    def get_sprite_name(self) -> str:
        """Get the sprite name for this entity"""
        return self.sprite_name


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
    "start": MapBlock(EntityType.START_POSITION, "start", True),
}


def get_entity_by_name(name: str) -> Optional[Entity]:
    """Get an entity from the registry by name"""
    return ENTITY_TYPES.get(name) 

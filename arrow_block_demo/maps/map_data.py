from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from core.entities import Entity

# Import get_entity_by_name function directly to avoid circular dependencies
from core.entities import get_entity_by_name


@dataclass
class GameMap:
    """Represents a playable game map with its layout and entity mappings"""
    name: str
    layout: List[str]
    entity_mappings: Dict[str, str]  # Maps characters to entity names
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        """Initialize derived properties after instantiation"""
        if self.layout:
            self.height = len(self.layout)
            self.width = max(len(row) for row in self.layout)
    
    def get_entity_at(self, x: int, y: int) -> Optional['Entity']:
        """Get the entity at the specified grid position"""
        if 0 <= y < self.height and 0 <= x < len(self.layout[y]):
            char = self.layout[y][x]
            entity_name = self.entity_mappings.get(char)
            if entity_name:
                return get_entity_by_name(entity_name)
        return None
    
    def is_position_valid(self, x: int, y: int) -> bool:
        """Check if a position is valid and passable"""
        if 0 <= y < self.height and 0 <= x < len(self.layout[y]):
            char = self.layout[y][x]
            entity_name = self.entity_mappings.get(char)
            if entity_name:
                entity = get_entity_by_name(entity_name)
                return entity and entity.passable
        return False
    
    def get_starting_position(self) -> Tuple[int, int]:
        """Find a valid starting position for the player block"""
        # Simple implementation: find the first valid position
        for y in range(self.height):
            for x in range(len(self.layout[y])):
                if self.is_position_valid(x, y):
                    return (x, y)
        # Fallback to center of map if no valid position found
        return (self.width // 2, self.height // 2) 
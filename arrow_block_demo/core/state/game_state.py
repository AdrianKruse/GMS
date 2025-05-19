from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from maps.map_data import GameMap


@dataclass
class Tower:
    """
    Represents a tower placed on the grid that can shoot projectiles
    """
    position: Tuple[int, int]  # (x, y)
    direction: Tuple[float, float]
    health: int = 100
    rate: int = 8
    tick: int = 0


@dataclass
class GameState:
    """
    GameState holds the persistent state that survives across rounds.
    """
    grid_width: int
    grid_height: int
    difficulty: int = 1
    currency: int = 0
    lives: int = 3
    wave_counter: int = 0
    towers: List[Tower] = field(default_factory=list)
    current_map: Optional['GameMap'] = None
    
    @property
    def grid_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the grid"""
        if self.current_map:
            # Use map dimensions if available
            return (self.current_map.width, self.current_map.height)
        # Fall back to default grid size
        return (self.grid_width, self.grid_height)
    
    def get_tower_at(self, x: int, y: int) -> Optional[Tower]:
        """Get a tower at the specified position if it exists"""
        # Check manually placed towers
        for tower in self.towers:
            if tower.position == (x, y):
                return tower
                
        # Then check if the current map has a tower at this position
        if self.current_map and self.current_map.is_position_tower(x, y):
            # Create a virtual tower for map-defined towers
            return Tower(position=(x, y), direction=(1.0, 0.0))
                
        return None
    
    def is_position_valid(self, x: int, y: int) -> bool:
        """Check if a position is valid for movement"""
        # Check if within grid bounds
        grid_w, grid_h = self.grid_dimensions
        if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
            return False
            
        # If we have a map, check if the position is passable
        if self.current_map:
            return self.current_map.is_position_valid(x, y)
            
        # Without a map, everywhere within bounds is valid
        return True 

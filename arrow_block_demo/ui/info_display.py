from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


class InfoDisplayType(Enum):
    """Enum representing different types of information that can be displayed"""
    STATS = auto()      # Standard game statistics
    TOWERS = auto()     # List of towers and their positions
    POSITION = auto()   # Information about a specific position


@dataclass
class InfoDisplayState:
    """
    Represents the current state of what's being shown in the info display
    """
    display_type: InfoDisplayType = InfoDisplayType.STATS
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_stats_view(self) -> bool:
        return self.display_type == InfoDisplayType.STATS
    
    @property
    def is_towers_view(self) -> bool:
        return self.display_type == InfoDisplayType.TOWERS
    
    @property
    def is_position_view(self) -> bool:
        return self.display_type == InfoDisplayType.POSITION


class InfoDisplayManager:
    """
    Manages what information is currently being displayed in the info panel
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InfoDisplayManager, cls).__new__(cls)
            cls._instance._display_state = InfoDisplayState()
        return cls._instance
    
    @property
    def display_state(self) -> InfoDisplayState:
        return self._display_state
    
    def show_stats(self) -> None:
        """Switch to showing standard game statistics"""
        self._display_state.display_type = InfoDisplayType.STATS
        self._display_state.context = {}
    
    def show_towers(self) -> None:
        """Switch to showing tower information"""
        self._display_state.display_type = InfoDisplayType.TOWERS
        self._display_state.context = {}
    
    def show_position_info(self, x: int, y: int) -> None:
        """Switch to showing information about a specific position"""
        self._display_state.display_type = InfoDisplayType.POSITION
        self._display_state.context = {'x': x, 'y': y} 
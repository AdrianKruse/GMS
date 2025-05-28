"""
Controller adapter module.

This adapter bridges between the controller layer and the core round logic.
It translates controller events and actions to and from the core round logic.
"""
from typing import List, Tuple, Dict, Optional, Any
import logging

from ..round_logic.state import RoundState, Tower, Projectile, BlockType
from ..round_logic.events import (
    Event, AgentMovedEvent, AgentDamagedEvent, TowerDamagedEvent,
    TowerDestroyedEvent, ProjectileCreatedEvent, ProjectileRemovedEvent,
    RoundOverEvent
)
from ..round_logic.step import step
from ..round_logic.state import RoundState
from ..agent_actions import MoveTo, Attack, Stand, Resume
from ..events import Event as ControllerEvent, PosChanged, RoundEnded, TowerDestroyed
from ..state.game_state import GameState

# Set up logger
logger = logging.getLogger("core.adapters.controller_adapter")


class ControllerAdapter:
    """
    Adapter that bridges between the controller and core round logic.
    """
    @staticmethod
    def initialize_round_state(game_state: GameState) -> RoundState:
        # Create grid layout from map if available, otherwise create empty grid
        grid_width = game_state.grid_width
        grid_height = game_state.grid_height
        grid_layout = []
        
        if game_state.current_map:
            # Convert map layout to BlockType grid
            for y in range(grid_height):
                row = []
                for x in range(grid_width):
                    if game_state.current_map.get_entity_at(x, y) == BlockType.START:
                        row.append(BlockType.START)
                    if game_state.current_map.is_position_valid(x, y):
                        row.append(BlockType.EMPTY)
                    else:
                        row.append(BlockType.WALL)
                grid_layout.append(row)
        else:
            # Create empty grid if no map
            grid_layout = [[BlockType.EMPTY for _ in range(grid_width)] for _ in range(grid_height)]
        
        initial_position = game_state.current_map.get_starting_position()

        return RoundState(
            grid_width=grid_width,
            grid_height=grid_height,
            grid_layout=grid_layout,
            towers=game_state.towers,
            projectiles=[],
            position=initial_position
        )
    
    # @staticmethod
    # def translate_events(core_events: List[Event]) -> List[ControllerEvent]:
    #     """
    #     Translate core events to controller events.
    #    
    #     Args:
    #         core_events: List of core events
    #        
    #     Returns:
    #         List of controller events
    #     """
    #     controller_events = []
    #    
    #     for event in core_events:
    #         if isinstance(event, AgentMovedEvent):
    #             controller_events.append(PosChanged(
    #                 x=event.position[0],
    #                 y=event.position[1]
    #             ))
    #         elif isinstance(event, TowerDestroyedEvent):
    #             controller_events.append(TowerDestroyed(
    #                 tower_id=event.tower_id
    #             ))
    #         elif isinstance(event, RoundOverEvent):
    #             controller_events.append(RoundEnded())
    #         # Other event types can be added as needed
    #    
    #     return controller_events

from core.round_logic.state import RoundState
from core.state.game_state import GameState
from maps.loader import get_map_loader
from core.adapters.controller_adapter import ControllerAdapter
from core.state.game_state import Tower
from typing import Tuple
import logging

logger = logging.getLogger('system')

def generate_round_state(map_name: str) -> Tuple[RoundState, GameState]:
    current_map = get_map_loader().get_map(map_name)
    game_state = GameState(grid_width=current_map.width, grid_height=current_map.height)
    game_state.current_map = current_map
    assert game_state.current_map is not None

    towers = []

    if map_name == "cross":
        towers = [
            Tower(position=(14, 1), direction=(-1, 0), health=100, tower_id=0),
            Tower(position=(14, 2), direction=(-1, 0), health=100, tower_id=1),
            Tower(position=(1, 14), direction=(0, -1), health=100, tower_id=2),
            Tower(position=(2, 14), direction=(0, -1), health=100, tower_id=3),
        ]
    elif map_name == "garden":
        towers = [
            Tower(position=(7, 7), direction=(-1, 0), health=100, tower_id=0),
            Tower(position=(7, 8), direction=(0, 1), health=100, tower_id=1),
            Tower(position=(8, 7), direction=(0, -1), health=100, tower_id=2),
            Tower(position=(8, 8), direction=(1, 0), health=100, tower_id=3),
        ]
    elif map_name == "default":
        towers = [
            Tower(position=(3, 3), direction=(1, 0), health=100, tower_id=0),
        ]
    else:
        raise ValueError(f"Map {map_name} not found")

    game_state.towers = towers

    round_state = ControllerAdapter.initialize_round_state(game_state)
    round_state = round_state.random_transform()
    game_state.towers = round_state.towers

    logger.info(f"Agent position in round state: {round_state.position}")

    return round_state, game_state




from core.state.game_state import GameState
from core.state.round_state import RoundState


def init_round(game_state: GameState) -> RoundState:
    """
    Initialize a new round state with default values.
    """
    # Determine starting position
    if game_state.current_map:
        # Use the map's designated starting position
        start_x, start_y = game_state.current_map.get_starting_position()
    else:
        # Default to center of grid
        start_x = game_state.grid_width // 2
        start_y = game_state.grid_height // 2

    return RoundState(
        agent_pos=(start_x, start_y),
        projectiles=[]
    )

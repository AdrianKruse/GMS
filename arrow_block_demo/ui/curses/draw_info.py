import curses
from typing import Optional, List, Tuple
from core.state.game_state import GameState
from core.state.round_state import RoundState
from controller.phases import Phase
from ui.info_display import InfoDisplayManager, InfoDisplayType


def render(win, game_state: GameState, round_state: Optional[RoundState], phase: Phase, dirty_rects: Optional[List[Tuple[int, int]]] = None) -> None:
    """
    Draw the info panel showing game state.
    """
    if win is None or game_state is None:
        return
    
    # Get the current info display state
    info_manager = InfoDisplayManager()
    display_state = info_manager.display_state
    
    # Use erase instead of clear to avoid full window redraw
    win.erase()
    win.box()
    
    # Set the title based on display type
    if display_state.is_stats_view:
        win.addstr(0, 2, "Info - Stats")
    elif display_state.is_towers_view:
        win.addstr(0, 2, "Info - Towers")
    elif display_state.is_position_view:
        win.addstr(0, 2, f"Info - Position ({display_state.context.get('x', '?')},{display_state.context.get('y', '?')})")
    else:
        win.addstr(0, 2, "Info")
    
    # Get window dimensions to ensure text fits
    rows, cols = win.getmaxyx()
    
    # Render different content based on info display type
    if display_state.is_stats_view:
        render_stats_view(win, rows, cols, game_state, round_state, phase)
    elif display_state.is_towers_view:
        render_towers_view(win, rows, cols, game_state)
    elif display_state.is_position_view:
        render_position_view(win, rows, cols, game_state, display_state.context)
    
    # Queue for refresh but don't refresh yet
    win.noutrefresh()


def render_stats_view(win, rows: int, cols: int, game_state: GameState, round_state: Optional[RoundState], phase: Phase) -> None:
    """Render the standard game statistics view"""
    # Make a list of lines to draw to check for bounds
    info_lines = [
        (2, f"Grid size: {game_state.grid_width}x{game_state.grid_height}"),
        (4, f"Wave: {game_state.wave_counter}"),
        (6, f"Currency: {game_state.currency}"),
        (8, f"Lives: {game_state.lives}"),
        (10, f"Phase: {phase.name}"),
    ]
    
    # Add map info if available
    if game_state.current_map and 14 < rows - 1:
        info_lines.append((14, f"Map: {game_state.current_map.name}"))
    
    # Draw game state information
    for y, text in info_lines:
        if y < rows - 1:
            win.addstr(y, 2, text[:cols-4])  # Truncate text if needed
    
    # Draw agent information if round_state exists
    agent_start_y = 16 if game_state.current_map else 14
    if round_state and agent_start_y < rows - 1:
        win.addstr(agent_start_y, 2, "Agent:")
        
        agent_lines = [
            (agent_start_y + 2, f"  Position: ({round_state.x}, {round_state.y})"),
            (agent_start_y + 4, f"  Health: {round_state.agent_health}"),
            (agent_start_y + 6, f"  Moving: {'Yes' if round_state.is_moving else 'No'}")
        ]
        
        for y, text in agent_lines:
            if y < rows - 1:
                win.addstr(y, 2, text[:cols-4])  # Truncate text if needed


def render_towers_view(win, rows: int, cols: int, game_state: GameState) -> None:
    """Render the towers information view"""
    if len(game_state.towers) == 0:
        win.addstr(2, 2, "No towers placed yet")
        win.addstr(4, 2, "Use 'tower x y dir' to place a tower")
        return

    # List each tower with empty lines between them
    for i, tower in enumerate(game_state.towers):
        y_pos = 4 + i * 2  # Double spacing
        if y_pos < rows - 1:
            win.addstr(y_pos, 2, f"{i+1}. ({tower.position[0]}, {tower.position[1]}) - Direction: {tower.direction}")
    
    # Add usage instructions with spacing
    instructions_y = 4 + len(game_state.towers) * 2 + 2
    if instructions_y < rows - 1:
        win.addstr(instructions_y, 2, "Use 'tower x y dir' to place a tower")


def render_position_view(win, rows: int, cols: int, game_state: GameState, context: dict) -> None:
    """Render information about a specific position"""
    x = context.get('x', 0)
    y = context.get('y', 0)
    
    win.addstr(2, 2, f"Position: ({x}, {y})")
    
    # Check if position is within grid bounds
    grid_width, grid_height = game_state.grid_dimensions
    if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
        win.addstr(4, 2, "Out of bounds!")
        return
    
    # Check for map entity
    if game_state.current_map:
        entity = game_state.current_map.get_entity_at(x, y)
        if entity:
            win.addstr(4, 2, f"Map Entity: {entity.entity_type.name}")
            win.addstr(6, 2, f"  Sprite: {entity.sprite_name}")
            win.addstr(8, 2, f"  Passable: {'Yes' if entity.passable else 'No'}")
        else:
            win.addstr(4, 2, "No objects at this position")
    else:
        win.addstr(4, 2, "No objects at this position") 

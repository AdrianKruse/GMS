import curses
from typing import Optional, List, Tuple, Dict
from core.state.game_state import GameState
from core.round_logic.state import RoundState, Projectile
from core.entities import EntityType
from sprites.loader import get_loader
import logging
import time


logger = logging.getLogger('graphics')
# Sprite loader
sprite_loader = get_loader()

# Cell size constants
CELL_WIDTH = 5
CELL_HEIGHT = 5

# Cache for static map elements to avoid redrawing them every frame
map_cache = {}

def init_map_cache(win, game_state: GameState) -> None:
    """Cache the static map elements to avoid redrawing them every frame"""
    global map_cache
    if not game_state.current_map:
        return
    
    grid_width, grid_height = game_state.grid_dimensions
    
    # Create a new cache
    map_cache = {}
    
    # Cache all static map elements
    for y in range(grid_height):
        for x in range(grid_width):
            entity = game_state.current_map.get_entity_at(x, y)
            if entity:
                map_cache[(x, y)] = entity.get_sprite_name()
            else:
                map_cache[(x, y)] = None

def render(win, game_state: GameState, round_state: Optional[RoundState], dirty_rects: Optional[List[Tuple[int, int]]] = None) -> None:
    """
    Draw the game field with entities from the map, towers, projectiles, and the agent using sprite-based rendering.
    """
    start_time = time.time()
    
    if win is None or game_state is None:
        return
    
    # Initialize map cache if it's empty and we have a map
    if not map_cache and game_state.current_map:
        init_map_cache(win, game_state)
    
    # Use partial redraw instead of erasing the entire window
    win.erase()  # Re-enable erase to fix rendering issues
    win.box()
    win.addstr(0, 2, "Game Field")
    
    rows, cols = win.getmaxyx()
    field_width = cols - 2
    field_height = rows - 2
    
    # Calculate how many cells we can fit in the view
    grid_width, grid_height = game_state.grid_dimensions
    visible_cols = min(grid_width, field_width // CELL_WIDTH)
    visible_rows = min(grid_height, field_height // CELL_HEIGHT)
    
    # Keep track of which cells we've drawn this frame
    drawn_cells = set()
    
    # Draw map with entities - always render the map from either direct map or cache
    if game_state.current_map:
        for y in range(visible_rows):
            for x in range(visible_cols):
                # Get entity from cache or directly from map
                if (x, y) in map_cache:
                    sprite_name = map_cache[(x, y)]
                    if sprite_name:
                        draw_sprite(win, x, y, sprite_name)
                    else:
                        draw_empty_cell(win, x, y)
                else:
                    # Fallback to direct map access
                    entity = game_state.current_map.get_entity_at(x, y)
                    if entity:
                        draw_sprite(win, x, y, entity.get_sprite_name())
                    else:
                        draw_empty_cell(win, x, y)
                drawn_cells.add((x, y))
    else:
        # Draw empty grid if no map, using grid_layout from round_state if available
        for y in range(visible_rows):
            for x in range(visible_cols):
                if round_state and round_state.grid_layout:
                    # Convert BlockType to appropriate sprite
                    block_type = round_state.grid_layout[y][x]
                    if block_type.value == 1:  # WALL
                        draw_sprite(win, x, y, "blocked")
                    else:  # EMPTY
                        draw_empty_cell(win, x, y)
                else:
                    draw_empty_cell(win, x, y)
                drawn_cells.add((x, y))

    # Draw towers (only if they've changed or we're doing a full redraw)
    tower_positions = set()
    for tower in game_state.towers:
        tower_x, tower_y = tower.position
        tower_positions.add((tower_x, tower_y))
        if 0 <= tower_x < visible_cols and 0 <= tower_y < visible_rows:
            sprite_name_to_draw = ""
            if tower.health <= 0 and tower.sprite_name_override:
                sprite_name_to_draw = tower.sprite_name_override
            elif tower.health <= 0:
                # Skip drawing if tower is destroyed and no override sprite
                continue
            else: # Tower is alive, determine sprite by direction
                direction = tower.direction
                if direction[0] > 0: sprite_name_to_draw = "tower_right"
                elif direction[0] < 0: sprite_name_to_draw = "tower_left"
                elif direction[1] > 0: sprite_name_to_draw = "tower_down"
                elif direction[1] < 0: sprite_name_to_draw = "tower_up"
                else: sprite_name_to_draw = "tower" # Default for active tower if no direction
            
            if sprite_name_to_draw:
                 draw_sprite(win, tower_x, tower_y, sprite_name_to_draw)
                 drawn_cells.add((tower_x, tower_y))

    # Remember old projectile positions to clear them
    old_projectile_positions = set()
    if round_state and hasattr(round_state, 'old_projectiles'):
        for projectile in round_state.old_projectiles:
            proj_x, proj_y = round(projectile.position[0]), round(projectile.position[1])
            old_projectile_positions.add((proj_x, proj_y))

    # Draw projectiles
    current_projectile_positions = set()
    if round_state:
        for projectile in round_state.projectiles:
            # Interpolate projectile position for smoother rendering
            # We calculate a position between the previous and next positions
            proj_x, proj_y = projectile.position
            
            # Get interpolated position
            rounded_x, rounded_y = round(proj_x), round(proj_y)
            current_projectile_positions.add((rounded_x, rounded_y))
            
            if 0 <= rounded_x < visible_cols and 0 <= rounded_y < visible_rows:
                # Draw the projectile
                draw_sprite(win, rounded_x, rounded_y, "projectile")
                drawn_cells.add((rounded_x, rounded_y))
        
        # Save current projectiles for next frame
        round_state.old_projectiles = list(round_state.projectiles)

    # Clear old projectile positions that don't have projectiles anymore
    for pos_x, pos_y in old_projectile_positions - current_projectile_positions:
        if (pos_x, pos_y) not in drawn_cells and 0 <= pos_x < visible_cols and 0 <= pos_y < visible_rows:
            # Only redraw if there's no tower or agent here
            if ((pos_x, pos_y) not in tower_positions and 
                (round_state is None or (pos_x, pos_y) != round_state.position)):
                # Redraw the original map tile
                if (pos_x, pos_y) in map_cache:
                    sprite_name = map_cache[(pos_x, pos_y)]
                    if sprite_name:
                        draw_sprite(win, pos_x, pos_y, sprite_name)
                    else:
                        draw_empty_cell(win, pos_x, pos_y)
                else:
                    draw_empty_cell(win, pos_x, pos_y)
                drawn_cells.add((pos_x, pos_y))

    # Remember old agent position to clear it
    old_position = (-1, -1)
    if round_state and hasattr(round_state, 'old_position'):
        old_position = round_state.old_position

    # Draw agent if round_state exists (always on top)
    if round_state:
        agent_x, agent_y = round_state.position
        if 0 <= agent_x < visible_cols and 0 <= agent_y < visible_rows:
            # Draw the agent with health indicator
            health_percentage = round_state.health / 100.0  # Assuming max health is 100
            if health_percentage > 0.7:
                sprite_name = "agent_healthy"
            elif health_percentage > 0.3:
                sprite_name = "agent_damaged"
            else:
                sprite_name = "agent_critical"
            
            draw_sprite(win, agent_x, agent_y, sprite_name)
            drawn_cells.add((agent_x, agent_y))
        
        # Clear old agent position if different from current
        if old_position != round_state.position and old_position != (-1, -1):
            old_x, old_y = old_position
            if (old_x, old_y) not in drawn_cells and 0 <= old_x < visible_cols and 0 <= old_y < visible_rows:
                # Only redraw if there's no tower or projectile here
                if (old_x, old_y) not in tower_positions and (old_x, old_y) not in current_projectile_positions:
                    # Redraw the original map tile
                    if (old_x, old_y) in map_cache:
                        sprite_name = map_cache[(old_x, old_y)]
                        if sprite_name:
                            draw_sprite(win, old_x, old_y, sprite_name)
                        else:
                            draw_empty_cell(win, old_x, old_y)
                    else:
                        draw_empty_cell(win, old_x, old_y)
        
        # Save current agent position for next frame
        round_state.old_position = round_state.position
    
    # Queue the window for refresh but don't refresh yet
    win.noutrefresh()
    
    # Log rendering time
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    if execution_time_ms > 10:  # Only log if rendering took more than 10ms
        logger.debug(f"Field render executed in {execution_time_ms:.2f}ms")


def draw_empty_cell(win, grid_x: int, grid_y: int) -> None:
    """Draw an empty cell at the specified grid position"""
    screen_x = 1 + grid_x * CELL_WIDTH
    screen_y = 1 + grid_y * CELL_HEIGHT
    
    # Draw empty cell (just blank space)
    for y_offset in range(CELL_HEIGHT):
        if screen_y + y_offset < win.getmaxyx()[0] - 1:  # Check y bounds
            for x_offset in range(CELL_WIDTH):
                if screen_x + x_offset < win.getmaxyx()[1] - 1:  # Check x bounds
                    win.addch(screen_y + y_offset, screen_x + x_offset, ' ')


def draw_sprite(win, grid_x: int, grid_y: int, sprite_name: str) -> None:
    """Draw a sprite at the specified grid position"""
    # Only log detailed info for non-terrain sprites to reduce log spam
    if sprite_name not in ["free", "blocked"]:
        logger.debug(f"Drawing sprite '{sprite_name}' at grid_pos=({grid_x},{grid_y})")
    
    sprite = sprite_loader.get_sprite(sprite_name)
    if not sprite:
        # Always log a warning if any sprite is not found
        logger.warning(f"Sprite '{sprite_name}' not found. Drawing empty cell at grid_pos=({grid_x},{grid_y}).")
        # Fallback if sprite not found
        draw_empty_cell(win, grid_x, grid_y)
        return
    
    screen_x = 1 + grid_x * CELL_WIDTH
    screen_y = 1 + grid_y * CELL_HEIGHT
    
    # Draw sprite line by line
    for y_offset, line in enumerate(sprite):
        if y_offset >= CELL_HEIGHT or screen_y + y_offset >= win.getmaxyx()[0] - 1:
            break
        
        for x_offset, char in enumerate(line):
            if x_offset >= CELL_WIDTH or screen_x + x_offset >= win.getmaxyx()[1] - 1:
                break
            
            win.addch(screen_y + y_offset, screen_x + x_offset, char) 

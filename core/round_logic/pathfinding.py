"""
Pathfinding module implementing A* algorithm for the core round logic.
"""
import heapq
import logging
from typing import List, Tuple, Dict, Set, Optional

from .state import RoundState

# Configure logging
logger = logging.getLogger('core')

def find_path(state: RoundState, start: Tuple[int, int], 
              end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find a path from start to end using A* algorithm.
    
    Args:
        state: The round state containing grid information
        start: Starting position (x, y)
        end: Target position (x, y)
        
    Returns:
        A list of positions [(x1, y1), (x2, y2), ...] from start to end,
    """
    logger.debug(f"Finding path from {start} to {end}")
    
    if start == end:
        return []
    
    # Validate grid bounds first
    if (not 0 <= start[0] < state.grid_width or not 0 <= start[1] < state.grid_height or
        not 0 <= end[0] < state.grid_width or not 0 <= end[1] < state.grid_height):
        logger.error(f"Start {start} or end {end} out of bounds ({state.grid_width}x{state.grid_height})")
        return []
    
    # Print information about grid state
    logger.debug(f"Grid dimensions: {state.grid_width}x{state.grid_height}")
    logger.debug(f"Agent position: {state.position}")
    logger.debug(f"Number of towers: {len(state.towers)}")
    
    # Check if start or end is invalid
    start_valid = state.is_position_valid(*start)
    end_valid = state.is_position_valid(*end)
    
    if not start_valid:
        logger.error(f"Start position {start} is invalid")
        
        # Print more details about why start is invalid
        if state.grid_layout[start[1]][start[0]] == 1:  # Assuming 1 is WALL
            logger.error(f"Start position {start} is a wall")
        
        for tower in state.towers:
            if not tower.health <= 0 and tower.position == start:
                logger.error(f"Start position {start} contains a non-destroyed tower {tower.tower_id}")
        
        return []
    
    if not end_valid:
        logger.error(f"End position {end} is invalid")
        
        # Print more details about why end is invalid
        try:
            if state.grid_layout[end[1]][end[0]] == 1:  # Assuming 1 is WALL
                logger.error(f"End position {end} is a wall")
        except IndexError:
            logger.error(f"End position {end} is out of bounds of the grid layout")
            
        for tower in state.towers:
            if not tower.health <= 0 and tower.position == end:
                logger.error(f"End position {end} contains a non-destroyed tower {tower.tower_id}")
        
        # Try to find a valid position adjacent to the target if possible
        alt_ends = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            alt_end = (end[0] + dx, end[1] + dy)
            if (0 <= alt_end[0] < state.grid_width and 
                0 <= alt_end[1] < state.grid_height and
                state.is_position_valid(*alt_end)):
                alt_ends.append(alt_end)
                logger.debug(f"Found alternative valid end position: {alt_end}")
        
        if alt_ends:
            # Find the closest valid end position
            end = min(alt_ends, key=lambda pos: abs(pos[0] - start[0]) + abs(pos[1] - start[1]))
            logger.debug(f"Selected closest alternative end position: {end}")
        else:
            logger.error("No valid position found near target")
            return []
    
    # If start and end are the same, return a single-position path
    if start == end:
        logger.debug("Start and end positions are the same")
        return [start]
    
    # Print the grid layout for debugging
    logger.debug("Grid layout:")
    for y in range(state.grid_height):
        row = []
        for x in range(state.grid_width):
            if (x, y) == start:
                row.append('S')
            elif (x, y) == end:
                row.append('E')
            elif not state.is_position_valid(x, y):
                row.append('#')
            else:
                row.append('.')
        logger.debug(''.join(row))
    
    # Initialize open and closed sets
    open_set: List[Tuple[float, int, Tuple[int, int]]] = []  # (f_score, counter, position)
    counter = 0  # Used for tie-breaking in heapq
    heapq.heappush(open_set, (heuristic(start, end), counter, start))
    
    # For tracking path
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # Cost from start to each node
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    
    # Estimated total cost from start to goal through each node
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, end)}
    
    # Positions seen so far
    open_set_hash: Set[Tuple[int, int]] = {start}
    
    # Limit the number of iterations for safety
    max_iterations = state.grid_width * state.grid_height * 4
    iterations = 0
    
    # Main A* loop
    while open_set and iterations < max_iterations:
        iterations += 1
        
        # Get position with lowest f_score
        _, _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)
        
        # Check if we've reached the target
        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            logger.debug(f"Path found with {len(path)} steps: {path}")
            return path
        
        # Explore neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if neighbor is within bounds
            if (not 0 <= neighbor[0] < state.grid_width or 
                not 0 <= neighbor[1] < state.grid_height):
                continue
            
            # Check if neighbor is valid
            if not state.is_position_valid(*neighbor):
                logger.debug(f"Position {neighbor} is not valid for pathfinding")
                continue
            
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + 1
            
            # Check if this path is better than previous ones
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Update path information
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                
                # Add to open set if not already there
                if neighbor not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    open_set_hash.add(neighbor)
    
    # No path found
    if iterations >= max_iterations:
        logger.error(f"Pathfinding failed: reached maximum iterations ({max_iterations})")
    else:
        logger.error("Pathfinding failed: no path exists")
    
    # Log the current state of the search
    logger.error(f"Open set size at termination: {len(open_set)}")
    logger.error(f"Positions explored: {iterations}")
    logger.error(f"No valid path exists from {start} to {end}")
    
    return []


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculate Manhattan distance heuristic.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) 

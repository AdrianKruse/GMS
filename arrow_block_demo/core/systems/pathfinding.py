from typing import List, Tuple, Dict, Set, Optional
import heapq
import logging

logger = logging.getLogger('game')

def astar(game_state, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm to find the shortest path from start to goal.
    Returns a list of positions from start to goal, or an empty list if no path exists.
    """
    if start == goal:
        return [start]
    
    # Initialize the open and closed sets
    open_set = []
    closed_set = set()
    
    # Dictionary to keep track of the most efficient path so far
    came_from = {}
    
    # g_score: cost from start to current node
    g_score = {start: 0}
    
    # f_score: estimated total cost from start to goal through the current node
    f_score = {start: manhattan_distance(start, goal)}
    
    # Add the start node to the open set
    heapq.heappush(open_set, (f_score[start], start))
    
    while open_set:
        # Get the node with the lowest f_score
        _, current = heapq.heappop(open_set)
        
        # If we reached the goal, reconstruct and return the path
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Add current to closed set
        closed_set.add(current)
        
        # Check all adjacent nodes
        for neighbor in get_neighbors(game_state, current):
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + 1  # Cost is always 1 for adjacent cells
            
            # If this neighbor is not in the open set, add it
            if neighbor not in [item[1] for item in open_set]:
                heapq.heappush(open_set, (f_score.get(neighbor, float('inf')), neighbor))
            # If this path to neighbor is worse than a previous one, skip
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
                
            # This path is the best until now, record it
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)
    
    # No path found
    logger.debug(f"No path found from {start} to {goal}")
    return []

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(game_state, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get all valid neighbors (adjacent cells) for a position"""
    x, y = position
    neighbors = []
    
    # Check all 4 directions
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        
        # Check if position is valid
        if game_state.is_position_valid(new_x, new_y):
            neighbors.append((new_x, new_y))
    
    return neighbors

def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                     current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct the path from the came_from dictionary"""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    
    # Reverse to get path from start to goal
    total_path.reverse()
    return total_path 
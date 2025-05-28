"""
Step module containing the pure step function for round logic.
"""
import math
import copy
import logging
import time
from typing import List, Tuple, Optional, Dict

from .state import RoundState, Tower, Projectile, BlockType
from .actions import MOVE_ACTION, ATTACK_ACTION, STAND_ACTION, RESUME_ACTION, N_ACTIONS
from .events import (
    Event, AgentMovedEvent, AgentDamagedEvent, TowerDamagedEvent,
    TowerDestroyedEvent, ProjectileCreatedEvent, ProjectileRemovedEvent,
    RoundOverEvent
)
from .pathfinding import find_path

# Configure logging
logger = logging.getLogger('core')

# Constants
AGENT_DAMAGE = 20  # Damage dealt by agent per attack
PROJECTILE_DAMAGE = 10  # Damage dealt by projectile to agent
# AGENT_ACTION_RATE removed - process actions every tick


def step(state: RoundState, action: Dict[str, any]) -> Tuple[RoundState, List[Event], float]:
    """
    Pure function that advances the game state by one tick.
    
    Args:
        state: Current game state
        action: Action to perform
        
    Returns:
        new_state: New game state
        events: List of events that occurred
        reward: Reward signal (for RL)
    """
    # Create a new state object to avoid modifying the original
    new_state = copy.deepcopy(state)
    events = []
    reward = -0.2
    
    # Process the tick counter
    new_state.tick_index += 1
    
    old_distance_to_nearest_tower = 0
    for tower in new_state.towers:
        if tower.health > 0:
            old_distance_to_nearest_tower = manhattan_distance(new_state.position, tower.position)
            break
    
    # Track the old position for projectile collision detection
    old_position = new_state.position
    
    # ALWAYS process agent action (removed tick rate check)
    logger.debug(f"Processing agent action: {action}")

    if action['action_type'] != RESUME_ACTION:
        new_state.last_interrupted_directive = state.current_active_directive
        new_state.current_active_directive = action
    else:
        new_state.current_active_directive = state.current_active_directive
        new_state.last_interrupted_directive = state.last_interrupted_directive
    
    if new_state.current_active_directive is None:
        logger.debug(f"No current active directive, this should not happen")
        return new_state, events, reward

    match new_state.current_active_directive['action_type']:
        case x if x == MOVE_ACTION:
            target_position = new_state.current_active_directive['target_position']
            if not new_state.is_position_valid(*target_position):
                logger.error(f"Invalid target position: {target_position}")
                new_state.current_active_directive = None
                new_state.last_interrupted_directive = None
                return new_state, events, reward
            if new_state.position == target_position:
                new_state.current_active_directive = None
                new_state.last_interrupted_directive = None
                return new_state, events, reward
            path = find_path(new_state, new_state.position, target_position)
            if len(path) > 1:
                new_state.position = path[1]
                events.append(AgentMovedEvent(position=new_state.position))
            else:
                new_state.current_active_directive = None
                new_state.last_interrupted_directive = None
                logger.error(f"Failed to find path from {new_state.position} to {target_position}")
        case x if x == ATTACK_ACTION:
            #TODO: remove this dumbass logic
            agent_target_tower_id = new_state.current_active_directive['target_id']
            tower = find_nearby_tower(new_state, new_state.position)
            #new_state.get_tower_by_id(agent_target_tower_id)
            if tower:
                if is_adjacent_to(new_state.position, tower.position):
                    tower.health -= AGENT_DAMAGE
                    events.append(TowerDamagedEvent(tower_id=tower.tower_id,
                                                    damage=AGENT_DAMAGE,
                                                    health_remaining=tower.health))
                if tower.health <= 0:
                    tower.health = 0
                    events.append(TowerDestroyedEvent(tower_id=tower.tower_id))
                    new_state.current_active_directive = None
                    new_state.last_interrupted_directive = None
            else:
                new_state.current_active_directive = None
                new_state.last_interrupted_directive = None
                logger.error(f"Failed to find tower with ID {agent_target_tower_id}")
        case x if x == STAND_ACTION:
            logger.debug(f"Agent is standing still")
            new_state.current_active_directive = new_state.last_interrupted_directive
            new_state.last_interrupted_directive = None
        case _:
            logger.error(f"Invalid action type: {new_state.current_active_directive['action_type']}")

    # Check if agent reached target position (if applicable)
    if new_state.current_active_directive and new_state.current_active_directive['action_type'] == MOVE_ACTION:
        target_position = new_state.current_active_directive['target_position']
        if new_state.position == target_position:
            events.append(AgentMovedEvent(position=new_state.position))
            new_state.current_active_directive = None
            new_state.last_interrupted_directive = None

    process_towers(new_state, events)

    # Process projectiles (movement and collisions)
    # If the agent moved, pass the previous position for collision detection
    position_changed = old_position != new_state.position
    process_projectiles(new_state, events, old_position if position_changed else None)
    
    # Calculate reward
    for event in events:
        if isinstance(event, TowerDamagedEvent):
            reward += 5
        elif isinstance(event, TowerDestroyedEvent):
            reward += 30
        elif isinstance(event, AgentDamagedEvent):
            reward -= 5
    
    for tower in new_state.towers:
        if tower.health > 0:
            distance = manhattan_distance(new_state.position, tower.position)
            if distance < old_distance_to_nearest_tower:
                reward += 1
            old_distance_to_nearest_tower = distance

    # Check if round is over
    if new_state.is_round_over:
        agent_survived = new_state.health > 0
        events.append(RoundOverEvent(agent_survived=agent_survived))
        
        # Give final reward based on outcome
        if agent_survived:
            reward += 200.0  # Big reward for surviving
        else:
            reward -= 100.0  # Penalty for dying
    
    # Log the state after processing
    logger.debug(f"After step processing - Agent position: {new_state.position}, Current active directive: {new_state.current_active_directive}, Last interrupted directive: {new_state.last_interrupted_directive}")
    
    return new_state, events, reward


def find_nearby_tower(state: RoundState, position: Tuple[int, int]) -> Optional[Tower]:
    for tower in state.towers:
        if tower.health > 0:
            if is_adjacent_to(position, tower.position):
                return tower

    return None

def process_towers(state: RoundState, events: List[Event]) -> None:
    """
    Process towers (projectile generation).
    """
    for tower in state.towers:
        if tower.health <= 0:
            continue
        
        tower.tick += 1
        if tower.tick >= tower.rate:
            # Create new projectile
            new_projectile = Projectile(
                position=tower.position,
                direction=tower.direction
            )
            state.projectiles.append(new_projectile)
            events.append(ProjectileCreatedEvent(
                position=tower.position,
                direction=tower.direction
            ))
            tower.tick = 0


def process_projectiles(state: RoundState, events: List[Event], old_position: Optional[Tuple[int, int]] = None) -> None:
    """
    Process projectiles (movement and collisions).
    
    Args:
        state: Round state
        events: List of events to append to
        old_position: Previous agent position (to check for collisions when agent moves)
    """
    surviving_projectiles = []
    
    for proj in state.projectiles:
        # Calculate new position
        new_pos = (
            proj.position[0] + proj.direction[0],
            proj.position[1] + proj.direction[1]
        )
        
        # Round position for collision detection
        rounded_pos = (round(new_pos[0]), round(new_pos[1]))
        new_projectile = Projectile(
            position=new_pos,
            direction=proj.direction
        )
        
        # Flag to track if this projectile survives this turn
        projectile_destroyed = False
        
        # Check for collisions with agent's current position
        if rounded_pos == state.position:
            # Agent hit by projectile
            logger.debug(f"Agent hit by projectile at {rounded_pos}")
            state.health -= PROJECTILE_DAMAGE
            if state.health < 0:
                state.health = 0
            
            events.append(AgentDamagedEvent(
                damage=PROJECTILE_DAMAGE,
                health_remaining=state.health
            ))
            events.append(ProjectileRemovedEvent(position=new_pos))
            projectile_destroyed = True
        
        # Also check for collisions with agent's previous position (if agent moved)
        # This catches the case where agent and projectile "pass through" each other
        elif old_position and rounded_pos == old_position:
            # Agent crossed paths with projectile
            logger.debug(f"Agent crossed paths with projectile at {rounded_pos}")
            state.health -= PROJECTILE_DAMAGE
            if state.health < 0:
                state.health = 0
                
            events.append(AgentDamagedEvent(
                damage=PROJECTILE_DAMAGE,
                health_remaining=state.health
            ))
            events.append(ProjectileRemovedEvent(position=new_pos))
            projectile_destroyed = True
        
        # Check if projectile is still in bounds
        elif not (0 <= rounded_pos[0] < state.grid_width and 
            0 <= rounded_pos[1] < state.grid_height and
            state.grid_layout[rounded_pos[1]][rounded_pos[0]] != BlockType.WALL):  # Not a wall
            logger.debug(f"Projectile out of bounds at {new_pos}")
            events.append(ProjectileRemovedEvent(position=new_pos))
            projectile_destroyed = True
 
        # Check if projectile hits a tower
        else:
            for tower in state.towers:
                if tower.health > 0 and tower.position == rounded_pos:
                    events.append(ProjectileRemovedEvent(position=new_pos))
                    logger.debug(f"Projectile hit tower at {rounded_pos}")
                    projectile_destroyed = True
                    break
        
        # Only add projectile to survivors if it wasn't destroyed
        if not projectile_destroyed:
            surviving_projectiles.append(new_projectile)
    
    state.projectiles = surviving_projectiles

def is_adjacent_to(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
    """Check if two positions are adjacent."""
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2) == 1


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_adjacent_position(state: RoundState, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Find a valid position adjacent to the given position."""
    x, y = pos
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        adj_x, adj_y = x + dx, y + dy
        if state.is_position_valid(adj_x, adj_y):
            logger.debug(f"Found valid adjacent position ({adj_x}, {adj_y}) to ({x}, {y})")
            return (adj_x, adj_y)
    logger.error(f"Could not find valid adjacent position to ({x}, {y})")
    return None 
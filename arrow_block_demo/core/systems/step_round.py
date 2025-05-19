from typing import List, Tuple
from core.state.game_state import GameState
from core.state.round_state import RoundState, Projectile
from core.events import Event, PosChanged, RoundEnded
from core.agent_actions import AgentAction, MoveTo, Stand
import math
import time
import logging

logger = logging.getLogger('game')

# Define the agent action rate
AGENT_ACTION_RATE = 2  # Agent moves every 2 ticks

def handle_agent_movement(game_state: GameState, round_state: RoundState, 
                          agent_action: AgentAction) -> List[Event]:
    """Process agent movement and return any events."""
    events = []
    old_x, old_y = round_state.agent_pos
    new_x, new_y = old_x, old_y
    
    match agent_action:
        case MoveTo(x=to_x, y=to_y):
            new_x, new_y = to_x, to_y
            logger.debug(f"Agent attempting move from ({old_x},{old_y}) to ({new_x},{new_y})")
        case Stand():
            logger.debug(f"Agent standing still at ({old_x},{old_y})")
            
    # Check if the new agent position is valid
    if game_state.is_position_valid(new_x, new_y):
        round_state.agent_pos = (new_x, new_y)
        logger.debug(f"Agent moved successfully to ({new_x},{new_y})")
        events.append(PosChanged(x=new_x, y=new_y))
    else:
        logger.debug(f"Invalid move attempted to ({new_x},{new_y})")
        
    return events

def process_projectiles(game_state: GameState, round_state: RoundState) -> List[Projectile]:
    """Process projectile creation and movement."""
    # Create a new list to avoid modifying while iterating
    projectiles = list(round_state.projectiles)
    
    # Spawn new projectiles from towers
    for tower in game_state.towers:
        tower.tick += 1
        if tower.tick >= tower.rate:
            spawn_position = tower.position
            direction = tower.direction
            # Adjust direction for consistent speed regardless of agent action rate
            adjusted_direction = (direction[0] / AGENT_ACTION_RATE, direction[1] / AGENT_ACTION_RATE)
            projectiles.append(Projectile(
                position=spawn_position,
                direction=adjusted_direction
            ))
            tower.tick = 0  # Reset tower tick
            
    # Move projectiles
    moved_projectiles = []
    for proj in projectiles:
        # Calculate new position
        new_pos = (
            proj.position[0] + proj.direction[0],
            proj.position[1] + proj.direction[1]
        )
        
        # Check if the projectile is still valid
        if game_state.is_position_valid(round(new_pos[0]), round(new_pos[1])):
            # Create new projectile with updated position
            moved_projectiles.append(Projectile(
                position=new_pos,
                direction=proj.direction
            ))
            
    return moved_projectiles

def handle_collisions(round_state: RoundState, projectiles: List[Projectile], 
                      agent_old_pos: Tuple[int, int]) -> Tuple[List[Projectile], List[Event]]:
    """Check for collisions between agent and projectiles."""
    events = []
    surviving_projectiles = []
    agent_current_pos = round_state.agent_pos
    
    for proj in projectiles:
        proj_new_pos = (round(proj.position[0]), round(proj.position[1]))
        
        # Calculate projectile's previous position
        proj_old_pos = (
            round(proj.position[0] - proj.direction[0]),
            round(proj.position[1] - proj.direction[1])
        )
        
        collided = False
        # Direct hit: Agent's current position matches projectile's current position
        if agent_current_pos == proj_new_pos:
            collided = True
            logger.debug(f"Collision: Agent at {agent_current_pos} hit by projectile at {proj_new_pos}")
        # Crossing paths: Agent and projectile swapped positions
        elif agent_current_pos == proj_old_pos and agent_old_pos == proj_new_pos:
            collided = True
            logger.debug(f"Collision: Agent and projectile crossed paths. Agent: {agent_old_pos}->{agent_current_pos}, Proj: {proj_old_pos}->{proj_new_pos}")
            
        if collided:
            round_state.agent_health -= 10
            if round_state.agent_health <= 0:
                round_state.agent_health = 0  # Cap health at 0
                # Only add RoundEnded event once
                if not any(isinstance(e, RoundEnded) for e in events):
                    events.append(RoundEnded())
                    logger.debug("Round ended due to agent health reaching zero.")
        else:
            surviving_projectiles.append(proj)
            
    return surviving_projectiles, events

def step_round(game_state: GameState, round_state: RoundState,
               agent_action: AgentAction) -> Tuple[RoundState, List[Event]]:
    """
    Advance the round state by one tick and return the (modified) state and any events that occurred.
    """
    start_time = time.time()  # Start timing the function execution
    
    events = []
    
    # Only process if the agent is moving
    if not round_state.is_moving:
        return round_state, events
        
    # Get agent's current position for collision detection later
    agent_old_pos = round_state.agent_pos
    
    # Process agent movement if it's time (every AGENT_ACTION_RATE ticks)
    agent_move_events = []
    should_move_agent = round_state.tick_index % AGENT_ACTION_RATE == 0
    
    if should_move_agent:
        agent_move_events = handle_agent_movement(game_state, round_state, agent_action)
        events.extend(agent_move_events)
    
    # Process projectiles (creation and movement) - always process every tick
    moved_projectiles = process_projectiles(game_state, round_state)
    
    # Handle collisions between agent and projectiles
    surviving_projectiles, collision_events = handle_collisions(
        round_state, moved_projectiles, agent_old_pos
    )
    events.extend(collision_events)
    
    # Update round state with surviving projectiles
    round_state.projectiles = surviving_projectiles
    
    # Increment tick counter
    round_state.tick_index += 1
    
    # Log execution time
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    logger.debug(f"step_round executed in {execution_time_ms:.2f}ms")
    
    return round_state, events


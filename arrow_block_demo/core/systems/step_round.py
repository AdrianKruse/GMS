from typing import List, Tuple
from core.state.game_state import GameState
from core.state.round_state import RoundState, Projectile, PlanningAction
from core.events import Event, PosChanged, RoundEnded
from core.agent_actions import AgentAction, MoveTo, Stand, Attack, Resume
from core.systems.pathfinding import astar
import math
import time
import logging

logger = logging.getLogger('game')

# Define the agent action rate
AGENT_ACTION_RATE = 2  # Agent moves every 2 ticks

def get_next_step_in_path(current_pos: Tuple[int, int], 
                         target_pos: Tuple[int, int], 
                         path: List[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Get the next step from current position towards the target position.
    If a path is provided, uses that; otherwise uses simple direct path logic.
    """
    if path and len(path) > 1:
        # Use the provided path - find current position in path and return next step
        try:
            current_index = path.index(current_pos)
            if current_index < len(path) - 1:
                return path[current_index + 1]
        except ValueError:
            # Current position not in path, regenerate path
            logger.debug(f"Current position {current_pos} not in path. Will need to regenerate.")
            pass
    
    # Fallback to direct path logic
    curr_x, curr_y = current_pos
    target_x, target_y = target_pos
    
    # Simple direct path for now, using Manhattan distance
    if curr_x < target_x:
        return (curr_x + 1, curr_y)
    elif curr_x > target_x:
        return (curr_x - 1, curr_y)
    elif curr_y < target_y:
        return (curr_x, curr_y + 1)
    elif curr_y > target_y:
        return (curr_x, curr_y - 1)
    else:
        # Already at target
        return current_pos

def handle_agent_attack(game_state: GameState, round_state: RoundState, 
                       target_id: str) -> List[Event]:
    """Handle agent attacking a tower."""
    events = []
    
    # Find tower with matching ID
    target_tower = None
    for tower in game_state.towers:
        if tower.tower_id == target_id:
            target_tower = tower
            break
            
    if not target_tower:
        logger.debug(f"Tower with ID {target_id} not found for attack")
        return events
        
    # Check if agent is adjacent to tower
    agent_x, agent_y = round_state.agent_pos
    tower_x, tower_y = target_tower.position
    
    if abs(agent_x - tower_x) + abs(agent_y - tower_y) <= 1:  # Manhattan distance <= 1
        # Apply damage to tower
        damage = 20  # Damage per attack
        was_destroyed = target_tower.take_damage(damage)
        logger.debug(f"Agent attacking tower at {tower_x}, {tower_y}. Damage: {damage}. Remaining health: {target_tower.health}")
        
        if was_destroyed:
            logger.debug(f"Tower at {tower_x}, {tower_y} was destroyed!")
            # TODO: Handle tower destruction event if needed
    else:
        logger.debug(f"Agent not adjacent to tower for attack. Agent at {round_state.agent_pos}, tower at {target_tower.position}")
        
    return events

def handle_agent_movement(game_state: GameState, round_state: RoundState, 
                          agent_action: AgentAction) -> List[Event]:
    """Process agent movement and return any events."""
    events = []
    old_x, old_y = round_state.agent_pos
    new_x, new_y = old_x, old_y
    
    # Handle different action types
    if isinstance(agent_action, MoveTo):
        # A new MoveTo action - set up a plan
        to_x, to_y = agent_action.x, agent_action.y
        logger.debug(f"New movement plan: Agent to move from ({old_x},{old_y}) to ({to_x},{to_y})")
        
        # Generate path using A* pathfinding
        path = astar(game_state, round_state.agent_pos, (to_x, to_y))
        if path:
            round_state.set_move_plan(to_x, to_y, path)
            logger.debug(f"Path found: {path}")
        else:
            logger.debug(f"No path found to ({to_x},{to_y})")
        # Don't move immediately, we'll follow the plan on subsequent ticks
            
    elif isinstance(agent_action, Attack):
        # A new Attack action - set up a plan
        logger.debug(f"New attack plan: Agent to attack target {agent_action.target_id}")
        round_state.set_attack_plan(agent_action.target_id)
        # Attack handling
        events.extend(handle_agent_attack(game_state, round_state, agent_action.target_id))
            
    elif isinstance(agent_action, Stand):
        # Agent stands still for this action
        logger.debug(f"Agent standing still at ({old_x},{old_y})")
        round_state.clear_plan()  # Clear any existing plan
            
    elif isinstance(agent_action, Resume):
        # Continue with current plan if there is one
        logger.debug(f"Agent resuming current plan")
        # No action needed here, we'll process the existing plan below
    
    # Process the current plan if one exists
    if round_state.current_plan and not round_state.current_plan.completed:
        plan = round_state.current_plan
        
        if plan.action_type == "move_to":
            # Get the next step towards the target using the plan's path
            target_x, target_y = plan.target_pos
            
            # Use the stored path if available, otherwise use simple pathfinding
            next_pos = get_next_step_in_path(round_state.agent_pos, plan.target_pos, plan.path)
            new_x, new_y = next_pos
            
            # Check if we've reached the target
            if (new_x, new_y) == plan.target_pos:
                logger.debug(f"Agent reached target position ({new_x},{new_y})")
                plan.completed = True
                
            logger.debug(f"Agent following movement plan: step from ({old_x},{old_y}) to ({new_x},{new_y})")
            
        elif plan.action_type == "attack":
            # Attack logic
            events.extend(handle_agent_attack(game_state, round_state, plan.target_id))
    
    # Check if the new agent position is valid and update position if it is
    if (new_x, new_y) != round_state.agent_pos and game_state.is_position_valid(new_x, new_y):
        round_state.agent_pos = (new_x, new_y)
        logger.debug(f"Agent moved successfully to ({new_x},{new_y})")
        events.append(PosChanged(x=new_x, y=new_y))
    elif (new_x, new_y) != round_state.agent_pos:
        logger.debug(f"Invalid move attempted to ({new_x},{new_y})")
        # If the move is invalid, the plan might be blocked, so we should reconsider it
        if round_state.current_plan and round_state.current_plan.action_type == "move_to":
            # Try to get a new path
            target = round_state.current_plan.target_pos
            new_path = astar(game_state, round_state.agent_pos, target)
            if new_path:
                logger.debug(f"Regenerating path to target: {new_path}")
                round_state.current_plan.path = new_path
            else:
                # If no path is possible, clear the plan
                logger.debug(f"Movement plan blocked and no alternative path, clearing plan")
                round_state.clear_plan()
        
    return events

def process_projectiles(game_state: GameState, round_state: RoundState) -> List[Projectile]:
    """Process projectile creation and movement."""
    # Create a new list to avoid modifying while iterating
    projectiles = list(round_state.projectiles)
    
    # Spawn new projectiles from towers
    for tower in game_state.towers:
        # Skip destroyed towers
        if tower.is_destroyed:
            continue
            
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
               agent_action: AgentAction, agent) -> Tuple[RoundState, List[Event]]:
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
    
    # On the first tick that a new action is received, we always process it
    is_new_action = (not isinstance(agent_action, Resume) and 
                    (not round_state.current_plan or 
                    (isinstance(agent_action, MoveTo) and 
                     (round_state.current_plan.action_type != "move_to" or 
                      agent_action.x != round_state.current_plan.target_pos[0] or 
                      agent_action.y != round_state.current_plan.target_pos[1])) or
                    (isinstance(agent_action, Attack) and 
                     (round_state.current_plan.action_type != "attack" or 
                      agent_action.target_id != round_state.current_plan.target_id))))
    
    should_move_agent = round_state.tick_index % AGENT_ACTION_RATE == 0 or is_new_action
    
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


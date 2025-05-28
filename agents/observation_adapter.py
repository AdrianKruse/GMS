from typing import Dict, Any
from dataclasses import asdict
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.round_logic.state import RoundState
from core.round_logic.actions import MOVE_ACTION, ATTACK_ACTION, STAND_ACTION, RESUME_ACTION, N_ACTIONS


def get_observation_from_round_state2(state: RoundState) -> Dict[str, Any]:
    """
    Convert a core state to an agent observation.
    
        Args:
            state: Core round state
            
        Returns:
            Observation dictionary
        """
    # Create a simplified observation of the game state for the agent
    observation = {
        'position': state.position,
        'agent_health': state.health,
        'towers': [
            {
                'position': tower.position,
                'health': tower.health,
                'is_destroyed': tower.health <= 0,
                'id': tower.tower_id
            }
            for tower in state.towers
        ],
        'grid_state': [
            [
                0 if state.is_position_valid(x, y) else 1
                for x in range(state.grid_width)
            ]
            for y in range(state.grid_height)
        ],
        'current_active_directive': None,
        'last_interrupted_directive': None,
    }
    
    return observation


def get_observation_from_round_state(state: RoundState) -> dict:
    # 1. grid_state: np.zeros() defaults to float64
    grid_state = np.zeros((state.grid_height, state.grid_width, 6), dtype=np.float32)
    grid_state[state.position[1], state.position[0], 0] = 1
    
    # Update grid_state[:, :, 1] assignment
    for y in range(state.grid_height):
        for x in range(state.grid_width):
            grid_state[y, x, 1] = 0 if state.is_position_valid(x, y) else 1

    # Update tower and projectile positions
    for tower in state.towers:
        grid_state[tower.position[1], tower.position[0], 2] = 1
    for projectile in state.projectiles:
        grid_state[round(projectile.position[1]), round(projectile.position[0]), 3] += 1
        grid_state[round(projectile.position[1]), round(projectile.position[0]), 4] = projectile.direction[0]
        grid_state[round(projectile.position[1]), round(projectile.position[0]), 5] = projectile.direction[1]

    # 2. shortest_distance_to_tower: np.inf defaults to float64
    shortest_distance_to_tower = np.float32(np.inf) # <-- Explicitly cast np.inf to float32

    # Ensure positions are converted to float32 NumPy arrays before subtraction
    # (You likely fixed this already, but double-check if it's consistent)
    for tower in state.towers:
        shortest_distance_to_tower = np.float32(min(shortest_distance_to_tower,
                                                 np.linalg.norm(np.array(state.position, dtype=np.float32) - 
                                                                np.array(tower.position, dtype=np.float32))))


    # 3. vector_state: np.array() can default to float64 if elements are Python floats
    directive = state.current_active_directive or {}
    last_dir = state.last_interrupted_directive or {}

    vector_state = np.array([
        np.float32(state.health / 100),
        np.float32(1 if directive.get('action_type') == MOVE_ACTION else 0),
        np.float32(1 if directive.get('action_type') == ATTACK_ACTION else 0),
        np.float32(1 if directive.get('action_type') == STAND_ACTION else 0),
        np.float32(1 if directive.get('action_type') == RESUME_ACTION else 0),
        np.float32(directive.get('target_position', [0, 0])[0] / state.grid_width if directive.get('action_type') == ATTACK_ACTION else 0),
        np.float32(directive.get('target_position', [0, 0])[1] / state.grid_height if directive.get('action_type') == ATTACK_ACTION else 0),
        np.float32(directive.get('target_id', 0) if directive.get('action_type') == ATTACK_ACTION else 0),
        np.float32(last_dir.get('action_type', 0)),
        np.float32(last_dir.get('target_position', [0, 0])[0] / state.grid_width if last_dir.get('action_type') == ATTACK_ACTION else 0),
        np.float32(last_dir.get('target_position', [0, 0])[1] / state.grid_height if last_dir.get('action_type') == ATTACK_ACTION else 0),
        np.float32(last_dir.get('target_id', 0) if last_dir.get('action_type') == ATTACK_ACTION else 0),
        shortest_distance_to_tower,
    ], dtype=np.float32)

    # 4. nearest_projectiles: np.zeros() defaults to float64
    nearest_projectiles = np.zeros((10*2*2,), dtype=np.float32) # <-- ADDED dtype=np.float32
    sorted_projectiles = sorted(np.array(state.projectiles), 
                                key=lambda x: norm_of_diff(x.position, state.position))
    if len(sorted_projectiles) > 0:
        farthest_projectile_value = norm_of_diff(sorted_projectiles[-1].position, state.position)
        if farthest_projectile_value == 0:
            farthest_projectile_value = 1
    for idx, projectile in enumerate(sorted_projectiles):
        if idx >= 10: break # Ensure we don't go out of bounds if more than 10 projectiles
        nearest_projectiles[idx*4] = np.float32(norm_of_diff(projectile.position, state.position))/farthest_projectile_value
        nearest_projectiles[idx*4+1] = np.float32(norm_of_diff(projectile.position, state.position))/farthest_projectile_value
        nearest_projectiles[idx*4+2] = np.float32(projectile.direction[0])
        nearest_projectiles[idx*4+3] = np.float32(projectile.direction[1])

    # 5. nearest_towers: np.zeros() defaults to float64
    nearest_towers = np.zeros((5*2*2,), dtype=np.float32) # <-- ADDED dtype=np.float32
    sorted_towers = sorted(np.array(state.towers), 
                           key=lambda x: abs(x.position[0] - state.position[0]) + abs(x.position[1] - state.position[1]))
    farthest_tower_value = np.inf
    if len(sorted_towers) > 0:
        farthest_tower_value = norm_of_diff(sorted_towers[-1].position, state.position)
    for idx, tower in enumerate(sorted_towers):
        if idx >= 5: break # Ensure we don't go out of bounds if more than 5 towers
        nearest_towers[idx*4] = np.float32(norm_of_diff(tower.position, state.position))/farthest_tower_value
        nearest_towers[idx*4+1] = np.float32(norm_of_diff(tower.position, state.position))/farthest_tower_value
        nearest_towers[idx*4+2] = np.float32(tower.health)/100
        nearest_towers[idx*4+3] = np.float32(tower.health <= 0) # Boolean to float is fine

    # Create a simplified observation of the game state for the agent
    observation = {
        'grid_map': grid_state,
        'vector_state': vector_state,
        'nearest_projectiles': nearest_projectiles,
        'nearest_towers': nearest_towers,
    }
    
    return observation

def norm_of_diff(a, b):
    return np.linalg.norm(np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32))

def get_action_space(state: RoundState) -> spaces.Dict:
    return spaces.Dict({
        'action_type': spaces.Discrete(N_ACTIONS),
        'target_position': spaces.Box(low=np.array([0, 0], dtype=np.float32),
                                      high=np.array([state.grid_width - 1, state.grid_height - 1], dtype=np.float32),
                                      dtype=np.float32),
        'target_id': spaces.Discrete(len(state.towers))
    })

def get_observation_space(state: RoundState) -> spaces.Dict:
    return spaces.Dict({
        'grid_map': spaces.Box(low=0.0, high=1.0,
                               shape=(state.grid_height, state.grid_width, 6),
                                dtype=np.float32),
        'vector_state': spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(13,),
                                   dtype=np.float32),
        'nearest_projectiles': spaces.Box(low=0, high=1,
                                          shape=(10*2*2,),
                                          dtype=np.float32),
        'nearest_towers': spaces.Box(low=0, high=1,
                                     shape=(5*2*2,),
                                     dtype=np.float32),
    })

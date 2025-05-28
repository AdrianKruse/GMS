import gymnasium as gym
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from .round_logic.state import RoundState, BlockType
from .round_logic.events import Event, RoundOverEvent
from .round_logic.step import step
import logging
import random
from agents.observation_adapter import get_observation_from_round_state, get_observation_space, get_action_space
from Training.ActionWrapper import ActionWrapper

logger = logging.getLogger('core.env')
class GMSEnv(gym.Env):
    """
    Gymnasium environment for the Arrow Block Demo.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Initialize configuration
        self.config = config
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        # Initialize round state properly from the provided state
        round_states = config.get('round_states')
        if len(round_states) == 0:
            logger.error(f"Expected a list of RoundState objects in config['round_states'], got {type(round_states)}")
        map_key = random.choice(list(round_states.keys()))
        self.map_name = map_key
        round_state = round_states[map_key]
        if not isinstance(round_state, RoundState):
            logger.error(f"Expected a RoundState object in config['round_state'], got {type(round_state)}")


        self.round_state = round_state
        self.tick_count = 0
        # Create a deep copy of the state for reset
        import copy
        self.start_states = copy.deepcopy(round_states)
        print("starting states: ", self.start_states)
        print("starting state: ",self.round_state)

        self.observation_space = get_observation_space(self.round_state)
        self.action_space = get_action_space(self.round_state)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        import copy


        round_states = copy.deepcopy(self.start_states)
        map_key = random.choice(list(round_states.keys()))
        self.map_name = map_key
        self.round_state = round_states[map_key]

        self.round_state = self.round_state.random_transform()
        starting_positions = [self.round_state.position]
        for row in self.round_state.grid_layout:
            for field in row:
                if field == BlockType.START:
                    starting_positions.append((row.index(field), row.index(field)))
        self.round_state.position = random.choice(starting_positions)
        print("starting state: ",self.round_state)
        self.tick_count = 0

        self.observation_space = get_observation_space(self.round_state)
        self.action_space = get_action_space(self.round_state)

        obs = get_observation_from_round_state(self.round_state)

        return obs, {}
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        """
        self.round_state, events, reward = step(self.round_state, action)
        self.tick_count += 1
        terminated = any(isinstance(event, RoundOverEvent) for event in events)
        truncated = False
        if self.tick_count >= self.max_episode_steps:
            truncated = True
        obs = get_observation_from_round_state(self.round_state)
        return obs, reward, terminated, truncated, {'events': events}
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        """
        pass

class EnvFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_env(self) -> gym.Env:
        return ActionWrapper(GMSEnv(self.config))

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch 
from agents.base import AgentBase

class PPOAgent(AgentBase):
    def __init__(self, state: RoundState):
        super().__init__(state)
        self.


    def act(self, observation: spaces.Dict) -> spaces.Dict:

        return super().act(observation)


"""
Abstract base class for agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class Agent(ABC):
    """
    Abstract base class for all agents.
    
    This follows a standard interface similar to Gym environments to make
    future integration with RL libraries easier.
    """
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Any:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Dictionary containing the current observation
            
        Returns:
            An action to be taken by the agent
        """
        pass
    
    def observe(self, observation: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> None:
        """
        Process the result of an action.
        
        This method is primarily used by RL agents to update their policy.
        Default implementation does nothing.
        
        Args:
            observation: Dictionary containing the next observation
            reward: Reward received from the environment
            done: Whether the episode is complete
            info: Additional information from the environment
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the agent's internal state.
        
        Default implementation does nothing.
        """
        pass 
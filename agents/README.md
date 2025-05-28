# Agents Module

This module contains agent implementations for the Arrow Block Demo game.

## Agent Interface

All agents implement the following interface defined in `base.py`:

- `act(observation)`: Select an action based on the current observation
- `observe(observation, reward, done, info)`: Process the result of an action (for learning agents)
- `reset()`: Reset the agent's internal state between episodes

## Included Agents

### SimpleAgent

A basic agent that targets and destroys towers. It uses a simple heuristic:
1. Find the nearest non-destroyed tower
2. If adjacent to the tower, attack it
3. Otherwise, move to an adjacent position

### RLAgent (Placeholder)

A placeholder for future reinforcement learning agents. Currently, it uses the same simple heuristic as `SimpleAgent`, but with hooks for RL functionality:
- Tracks rewards over time
- Has placeholder methods for experience collection
- Will be expanded to use reinforcement learning algorithms in the future

## Usage

To use an agent in the game:

```python
from agents.simple_agent import SimpleAgent

# Create agent at a starting position
agent = SimpleAgent(starting_position=(5, 5))

# Get observation from environment
observation = {...}  # Game state representation

# Get action from agent
action = agent.act(observation)

# Apply action to environment
# ...

# For learning agents, provide feedback
agent.observe(next_observation, reward, done, info)
```

## Creating a Custom Agent

To create a custom agent, extend the `Agent` base class:

```python
from agents.base import Agent

class MyCustomAgent(Agent):
    def __init__(self, starting_position):
        self.pos = starting_position
        # Initialize other state
    
    def act(self, observation):
        # Implement decision-making logic
        return action
    
    def observe(self, observation, reward, done, info):
        # Optional: implement learning logic
        pass
    
    def reset(self):
        # Reset internal state
        pass
``` 
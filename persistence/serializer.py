import json
import os
from typing import Dict, Any

from core.state.game_state import GameState


def _game_state_to_dict(game_state: GameState) -> Dict[str, Any]:
    """Convert GameState to a serializable dictionary."""
    return {
        "grid_width": game_state.grid_width,
        "grid_height": game_state.grid_height,
        "difficulty": game_state.difficulty,
        "currency": game_state.currency,
        "lives": game_state.lives,
        "wave_counter": game_state.wave_counter
    }


def _dict_to_game_state(data: Dict[str, Any]) -> GameState:
    """Create a GameState from a dictionary."""
    return GameState(
        grid_width=data.get("grid_width", 16),
        grid_height=data.get("grid_height", 16),
        difficulty=data.get("difficulty", 1),
        currency=data.get("currency", 0),
        lives=data.get("lives", 3),
        wave_counter=data.get("wave_counter", 0)
    )


def save_game_state(game_state: GameState, filename: str) -> bool:
    """
    Save the game state to a JSON file.
    
    Args:
        game_state: The GameState to save
        filename: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Convert to serializable dict
        data = _game_state_to_dict(game_state)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving game state: {e}")
        return False


def load_game_state(filename: str) -> GameState:
    """
    Load a game state from a JSON file.
    
    Args:
        filename: Path to the save file
        
    Returns:
        The loaded GameState or a new default GameState if loading fails
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return _dict_to_game_state(data)
    except Exception as e:
        print(f"Error loading game state: {e}")
        return GameState(grid_width=16, grid_height=16) 

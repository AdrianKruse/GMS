from enum import Enum, auto


class Phase(Enum):
    """Game phases that control the flow of the game."""
    BUILD = auto()    # Player can issue commands
    ROUND = auto()    # Block is moving automatically
    TRAINING = auto()    # Block is moving automatically
    SUMMARY = auto()  # Round has ended, showing results
    GAME_OVER = auto() # Game has ended 

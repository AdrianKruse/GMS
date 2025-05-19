"""
Arrow Block Demo - A simple terminal game with a block that moves across a grid.
"""

# Main package initialization
from . import cli
from . import controller
from . import core
from . import maps
from . import persistence
from . import sprites
from . import ui

__all__ = ['cli', 'controller', 'core', 'maps', 'persistence', 'sprites', 'ui']

__version__ = "0.1.0" 
#!/usr/bin/env python3
import curses
import sys
import traceback
import locale
import logging
import logging_setup
from typing import Optional

from cli.args import parse_args
from controller.mainloop import run_game
from persistence.serializer import load_game_state


def check_terminal_size() -> Optional[str]:
    """Check if terminal is large enough for the game."""
    try:
        # Get terminal size without initializing curses
        import os
        rows, cols = os.popen('stty size', 'r').read().split()
        rows, cols = int(rows), int(cols)
        
        if rows < 15 or cols < 50:
            return f"Terminal too small: {rows}x{cols}. Need at least 15x50."
    except Exception:
        # If we can't check size, we'll let curses initialization fail later
        pass
    
    return None


def main():
    """Main entry point for the Arrow Block Demo game."""
    logging_setup.setup_logs()
    logger = logging.getLogger('game')
    logger.debug("game started")
    # Set up locale for proper character display
    locale.setlocale(locale.LC_ALL, '')
    
    # Parse command line arguments
    args, tick_rate, grid_width, grid_height = parse_args()
    
    # Check terminal size before entering curses mode
    size_error = check_terminal_size()
    if size_error:
        print(size_error)
        print("Please resize your terminal window and try again.")
        return 1
    
    # Run the game with curses
    try:
        curses.wrapper(run_game, tick_rate, grid_width, grid_height)
        logger.debug("game ended")
        return 0
    except KeyboardInterrupt:
        print("Game terminated by user.")
        return 0
    except curses.error as e:
        print(f"Curses error: {e}")
        print("This might be due to a terminal window that's too small or doesn't support required features.")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return 1




if __name__ == "__main__":
    sys.exit(main()) 

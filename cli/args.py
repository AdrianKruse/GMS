import argparse
from typing import Tuple


def parse_args() -> Tuple[argparse.Namespace, float, int, int]:
    """
    Parse command line arguments for game configuration.
    
    Returns:
        args: Parsed arguments
        tick_rate: Game update rate in seconds
        grid_width: Width of the game grid
        grid_height: Height of the game grid
    """
    parser = argparse.ArgumentParser(description='Arrow Block Demo - A simple terminal game')
    
    parser.add_argument('--tick-rate', type=float, default=0.2,
                        help='Game update rate in seconds (default: 0.2)')
    parser.add_argument('--grid-width', type=int, default=16,
                        help='Width of the game grid (default: 20)')
    parser.add_argument('--grid-height', type=int, default=16,
                        help='Height of the game grid (default: 20)')
    parser.add_argument('--replay', type=str, 
                        help='Path to replay file')
    
    args = parser.parse_args()
    
    return args, args.tick_rate, args.grid_width, args.grid_height 

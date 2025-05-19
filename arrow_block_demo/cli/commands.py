from core.actions import (
    Action, StartRound, Quit, ErrorAction, 
    PlaceTower, ShowStats, ShowTowers, ShowPosition
)
import re
import logging

logger = logging.getLogger('game')

def parse_command(command: str) -> Action:
    """
    Parse a command string and return an appropriate Action object.
    
    Commands:
    - start: Begin movement
    - tower x y dir: Place a tower at position (x,y) with direction dir (0=up, 1=right, 2=down, 3=left)
    - show stats: Show standard game statistics
    - show towers: Show tower information
    - show x y: Show information about position (x,y)
    - quit/exit: Exit the game
    """
    cmd = command.strip().lower()
    logger.debug(f"Parsing command: {command}")

    # Action commands
    if cmd in ["start"]:
        logger.debug("Parsed as StartRound command")
        return StartRound()
    elif cmd in ["quit", "exit", "q"]:
        logger.debug("Parsed as Quit command")
        return Quit()
    
    # Tower command - "tower x y dir"
    elif cmd.startswith("tower "):
        match = re.match(r"tower\s+(\d+)\s+(\d+)\s+(\d)", cmd)
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                dir = int(match.group(3))
                if dir not in [0, 1, 2, 3]:
                    logger.warning(f"Invalid tower direction: {dir}")
                    return ErrorAction(message="Tower direction must be 0 (up), 1 (right), 2 (down), or 3 (left)")
                logger.debug(f"Parsed as PlaceTower command: x={x}, y={y}, dir={dir}")
                return PlaceTower(x=x, y=y, direction=dir)
            except ValueError:
                logger.warning(f"Invalid coordinates in tower command: {cmd}")
                return ErrorAction(message="Invalid coordinates for tower")
        else:
            logger.warning(f"Invalid tower command format: {cmd}")
            return ErrorAction(message="Invalid tower command format. Use: tower x y dir")
    
    # Show stats command
    elif cmd == "show stats":
        logger.debug("Parsed as ShowStats command")
        return ShowStats()
    
    # Show towers command
    elif cmd == "show towers":
        logger.debug("Parsed as ShowTowers command")
        return ShowTowers()
    
    # Show position command - "show x y"
    elif cmd.startswith("show "):
        match = re.match(r"show\s+(\d+)\s+(\d+)", cmd)
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                logger.debug(f"Parsed as ShowPosition command: x={x}, y={y}")
                return ShowPosition(x=x, y=y)
            except ValueError:
                logger.warning(f"Invalid coordinates in show command: {cmd}")
                return ErrorAction(message="Invalid coordinates")
        else:
            # Check if it's a different show command that didn't match our patterns
            if not cmd in ["show stats", "show towers"]:
                logger.warning(f"Invalid show command format: {cmd}")
                return ErrorAction(message="Invalid show command format. Use: show stats, show towers, or show x y")
    
    # Invalid command
    else:
        logger.error(f"Unknown command: {command}")
        return ErrorAction(message=f"Unknown command: {command}. Available commands: start, tower x y dir, show stats, show towers, show x y, quit") 

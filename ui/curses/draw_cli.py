import curses
from typing import List, Optional


def render(win, command_history: List[str], current_command: str, error_message: Optional[str] = None) -> None:
    """
    Draw the CLI panel with command history and current input.
    """
    if win is None:
        return
    
    # Use erase instead of clear to reduce flickering
    win.erase()
    win.box()
    win.addstr(0, 2, "Command")
    
    rows, cols = win.getmaxyx()
    max_history_items = (rows - 4) // 2  # Account for spacing and prompt
    
    # Draw command history (most recent at the bottom) with empty lines between entries
    history_to_show = command_history[-max_history_items:] if len(command_history) > max_history_items else command_history
    for i, cmd in enumerate(history_to_show):
        y_pos = 1 + i * 2  # Double spacing
        if y_pos < rows - 2:
            # Truncate command if needed to avoid drawing errors
            safe_cmd = cmd[:cols-4] if cols > 4 else ""
            win.addstr(y_pos, 2, safe_cmd)
    
    # Draw error message if any, with spacing
    if error_message and rows > 5:  # Ensure we have room for error + prompt
        error_row = rows - 4  # Leave room for prompt
        # Truncate error message if needed
        safe_error = error_message[:cols-4] if cols > 4 else ""
        try:
            win.addstr(error_row, 2, safe_error, curses.A_BOLD)
        except curses.error:
            # Handle potential curses errors by being more cautious
            win.addstr(error_row, 2, safe_error[:cols-6])
    
    # Draw prompt and current command
    if rows > 2:
        prompt_row = rows - 2
        prefix = "> "
        # Ensure we don't try to draw beyond window bounds
        max_cmd_len = cols - len(prefix) - 4
        safe_cmd = current_command[:max_cmd_len] if max_cmd_len > 0 else ""
        win.addstr(prompt_row, 2, prefix + safe_cmd)
        
        # Position cursor at the end of the current command
        try:
            curses.curs_set(1)  # Show cursor
            win.move(prompt_row, 4 + min(len(current_command), max_cmd_len))
        except curses.error:
            # Handle potential cursor position errors
            pass
    
    # Queue for refresh but don't refresh yet
    win.noutrefresh() 
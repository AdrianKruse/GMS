import curses
from typing import Dict


def make_windows(stdscr) -> Dict[str, curses.window]:
    """
    Create the window layout for the game:
    - info: top-right panel showing state information
    - cli: bottom-left panel for command input
    - field: right panel showing the game grid
    """
    rows, cols = stdscr.getmaxyx()
    
    # Calculate dimensions with safety bounds
    left_w = max(16, cols // 3)
    info_h = max(5, rows * 2 // 3)
    cli_h = max(3, rows - info_h)
    field_w = max(16, cols - left_w)
    
    # Ensure we don't try to create windows beyond screen bounds
    if left_w + field_w > cols:
        field_w = cols - left_w
    
    if info_h + cli_h > rows:
        cli_h = rows - info_h
    
    # Create windows with safe dimensions
    info_win = curses.newwin(info_h, left_w, 0, 0)
    cli_win = curses.newwin(cli_h, left_w, info_h, 0)
    field_win = curses.newwin(rows, field_w, 0, left_w)
    
    # Set up borders
    info_win.box()
    cli_win.box()
    field_win.box()
    
    # Add titles (safely)
    try:
        info_win.addstr(0, 2, "Info")
        cli_win.addstr(0, 2, "Command")
        field_win.addstr(0, 2, "Game Field")
    except curses.error:
        # Handle potential errors when drawing on small terminals
        pass
    
    # Enable immediate keyboard input mode
    stdscr.nodelay(True)
    
    # Refresh all windows
    info_win.noutrefresh()
    cli_win.noutrefresh()
    field_win.noutrefresh()
    
    return {
        "info": info_win,
        "cli": cli_win,
        "field": field_win
    } 

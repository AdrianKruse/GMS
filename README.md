# Arrow Block Demo

A simple terminal-based game where you control a block with an arrow that can move across a grid.

## Features

- Terminal-based UI with separate windows for info, game field, and command input
- Control a block by changing its facing direction with arrow commands
- Send the block moving across the grid until it reaches the edge
- Clean architecture with separation of concerns (MVC-like pattern)
- Serializable game state for saving/loading
- Optional replay system for recording and playback
- Optimized rendering to minimize flickering and CPU usage

## Installation

1. Clone the repository
2. Make sure you have Python 3.6+ installed
3. No additional dependencies required (uses built-in curses library)

## Requirements

- A terminal with a minimum size of 15x50 characters
- Support for curses/ncurses
- Python 3.6 or higher

## Usage

Run the game with:

```
python -m arrow_block_demo
```

Or directly from the source directory:

```
python main.py
```

### Command Line Arguments

- `--tick-rate`: Game update rate in seconds (default: 0.2)
- `--grid-width`: Width of the game grid (default: 10)
- `--grid-height`: Height of the game grid (default: 10)
- `--replay`: Path to replay file (optional)

### In-Game Commands

- Direction: `up`, `down`, `left`, `right` (or `u`, `d`, `l`, `r` for short)
- Begin movement: `start` or `go`
- Quit: `quit` or `q`

## Architecture

The project follows a clean architecture with clear separation of concerns:

- `core/`: Pure domain logic with no dependencies on UI or external systems
- `ui/`: UI components based on the curses library
- `controller/`: Game loop and phase management
- `cli/`: Command parsing and validation
- `persistence/`: State serialization and replay functionality

## Performance Notes

The game is optimized to:
- Minimize screen flickering by using selective redrawing
- Reduce CPU usage with efficient rendering
- Handle terminal resizing gracefully
- Validate terminal capabilities before launching
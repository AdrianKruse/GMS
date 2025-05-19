import curses
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from core.state.game_state import GameState, Tower
from core.state.round_state import RoundState, Projectile
from core.actions import (
    Action, StartRound, Quit, ErrorAction,
    PlaceTower, ShowStats, ShowTowers, ShowPosition
)
from core.events import Event, RoundEnded
from core.systems.init_round import init_round
from core.systems.step_round import step_round
from core.agent_actions import AgentAction, MoveTo, Stand
from core.entities import Agent, AgentState
from controller.phases import Phase
from cli.commands import parse_command
from ui.curses import windows, draw_field, draw_info, draw_cli
from ui.info_display import InfoDisplayManager
from persistence.serializer import save_game_state
from maps.loader import get_map_loader
import logging


class GameController:
    def __init__(self, logger, stdscr, tick_rate: float = 0.03, grid_width: int = 16, grid_height: int = 16):
        self.stdscr = stdscr
        self.tick_rate = tick_rate
        self.windows = windows.make_windows(stdscr)
        self.logger = logger
        
        # Initialize states
        self.game_state = GameState(grid_width=grid_width, grid_height=grid_height)
        
        # Load map
        map_loader = get_map_loader()
        default_map = map_loader.get_default_map()
        self.game_state.current_map = default_map
        if default_map:
            # Update grid dimensions to match map if available
            self.game_state.grid_width = default_map.width
            self.game_state.grid_height = default_map.height

        logger.debug("loaded map: ")

        self.round_state: Optional[RoundState] = None
        
        # Game control
        self.phase = Phase.BUILD
        self.running = True
        
        # Command line state
        self.command_history: List[str] = []
        self.current_command = ""
        self.error_message: Optional[str] = None
        
        # Initialize round
        self.round_state = init_round(self.game_state)
        
        # Agent instance (for now, always at round_state.agent_pos)
        self.agent = Agent(self.round_state.agent_pos)
        
        # Force full redraw on first frame
        self.needs_full_redraw = True
        self.last_game_state = None
        self.last_round_state = None
        self.last_phase = None
    
    def log_game_state(self, event_name: str = "GameState Update") -> None:
        """Logs the current game state in a structured way."""
        if not self.game_state:
            self.logger.debug(f"{event_name}: GameState is None")
            return
        
        towers_info = []
        if self.game_state.towers:
            for i, tower in enumerate(self.game_state.towers):
                towers_info.append(
                    f"  Tower {i+1}: Pos=({tower.position[0]},{tower.position[1]}), Dir={tower.direction}, Tick={tower.tick}, Rate={tower.rate}"
                )
        else:
            towers_info.append("  No towers present.")
            
        map_info = "None"
        if self.game_state.current_map:
            map_info = f"Name='{self.game_state.current_map.name}', Dimensions=({self.game_state.current_map.width}x{self.game_state.current_map.height})"

        state_details = (
            f"{event_name}:\\n"
            f"  Grid Dimensions: ({self.game_state.grid_width}x{self.game_state.grid_height})\\n"
            f"  Difficulty: {self.game_state.difficulty}\\n"
            f"  Currency: {self.game_state.currency}\\n"
            f"  Lives: {self.game_state.lives}\\n"
            f"  Wave Counter: {self.game_state.wave_counter}\\n"
            f"  Current Map: {map_info}\\n"
            f"  Towers:\\n" + "\\n".join(towers_info)
        )
        self.logger.info(state_details)

    def log_round_state(self, event_name: str = "RoundState Update") -> None:
        """Logs the current round state in a structured way."""
        if not self.round_state:
            self.logger.debug(f"{event_name}: RoundState is None")
            return

        projectiles_info = []
        if self.round_state.projectiles:
            for i, p in enumerate(self.round_state.projectiles):
                projectiles_info.append(
                    f"  Projectile {i+1}: Pos=({p.position[0]:.2f},{p.position[1]:.2f}), Dir=({p.direction[0]:.2f},{p.direction[1]:.2f})"
                )
        else:
            projectiles_info.append("  No projectiles present.")
            
        state_details = (
            f"{event_name}:\\n"
            f"  Agent Position: ({self.round_state.agent_pos[0]},{self.round_state.agent_pos[1]})\\n"
            f"  Agent Health: {self.round_state.agent_health}\\n"
            f"  Is Moving: {self.round_state.is_moving}\\n"
            f"  Tick Index: {self.round_state.tick_index}\\n"
            f"  Projectiles:\\n" + "\\n".join(projectiles_info)
        )
        self.logger.info(state_details)

    def process_action(self, action: Action) -> List[Event]:
        """Process a player action and return events."""
        events = []
        
        if isinstance(action, StartRound):
            if self.phase == Phase.BUILD and not self.round_state.is_moving:
                self.round_state.is_moving = True
                self.phase = Phase.ROUND
                self.needs_full_redraw = True
                self.logger.debug("Starting new round")
            else:
                self.error_message = "Cannot start movement during active round"
                self.logger.warning("Attempted to start round during active round")
        
        elif isinstance(action, Quit):
            self.running = False
            save_game_state(self.game_state, "latest_game.json")
            self.logger.info("Game quit, state saved")
            
        elif isinstance(action, PlaceTower):
            if self.phase != Phase.BUILD:
                self.error_message = "Can only place towers during build phase"
                self.logger.warning(f"Attempted to place tower during {self.phase} phase")
                return events
                
            # Check if coordinates are within the grid
            if not self.game_state.is_position_valid(action.x, action.y):
                self.error_message = "Cannot place tower at that position"
                self.logger.warning(f"Invalid tower position attempted: ({action.x}, {action.y})")
                return events
                
            # Check if there's already a tower at this position
            for tower in self.game_state.towers:
                if tower.position == (action.x, action.y):
                    self.error_message = "There's already a tower at this position"
                    self.logger.warning(f"Tower already exists at ({action.x}, {action.y})")
                    return events
            
            # Create direction vector based on direction parameter
            direction_vector = (0.0, 0.0)
            if action.direction == 0:  # Up
                direction_vector = (0.0, -1.0)
            elif action.direction == 1:  # Right
                direction_vector = (1.0, 0.0)
            elif action.direction == 2:  # Down
                direction_vector = (0.0, 1.0)
            elif action.direction == 3:  # Left
                direction_vector = (-1.0, 0.0)
            
            # Place the tower
            self.game_state.towers.append(Tower(
                position=(action.x, action.y),
                direction=direction_vector
            ))
            self.logger.info(f"Tower placed at ({action.x}, {action.y}) with direction {direction_vector}")
            self.needs_full_redraw = True
            
            # If the info display is showing towers, update it
            info_manager = InfoDisplayManager()
            if info_manager.display_state.is_towers_view:
                info_manager.show_towers()  # Refresh the view
        
        elif isinstance(action, ShowStats):
            # Switch to stats view
            info_manager = InfoDisplayManager()
            info_manager.show_stats()
            self.needs_full_redraw = True
            
        elif isinstance(action, ShowTowers):
            # Switch to towers view
            info_manager = InfoDisplayManager()
            info_manager.show_towers()
            self.needs_full_redraw = True
            
        elif isinstance(action, ShowPosition):
            # Show information about a specific position
            info_manager = InfoDisplayManager()
            info_manager.show_position_info(action.x, action.y)
            self.needs_full_redraw = True
        
        elif isinstance(action, ErrorAction):
            self.error_message = action.message
        
        return events
    
    def handle_input(self) -> None:
        """Handle keyboard input."""
        self.stdscr.timeout(0)  # Non-blocking input
        try:
            key = self.stdscr.getch()
            if key == curses.ERR:
                return  # No input available
            
            # Handle special keys
            if key == curses.KEY_BACKSPACE or key == 127:
                self.current_command = self.current_command[:-1]
            elif key == curses.KEY_ENTER or key == 10 or key == 13:
                # Process command
                if self.current_command:
                    self.command_history.append(self.current_command)
                    
                    # Parse and process command
                    action = parse_command(self.current_command)
                    events = self.process_action(action)
                    
                    # Log states after processing the action
                    self.log_game_state(f"After action: {self.current_command}")
                    self.log_round_state(f"After action: {self.current_command}")
                    
                    # Process events (will be expanded in future)
                    for event in events:
                        if isinstance(event, RoundEnded):
                            self.phase = Phase.SUMMARY
                            self.game_state.wave_counter += 1
                            self.needs_full_redraw = True
                    
                    self.current_command = ""
                    self.error_message = None
            elif key == 27:  # ESC key
                self.running = False
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.current_command += chr(key)
                
        except Exception as e:
            self.error_message = f"Error: {str(e)}"
    
    def update(self) -> None:
        """Update game state based on current phase."""
        if self.phase == Phase.ROUND and self.round_state.is_moving:
            # Create an AgentState (placeholder for now)
            agent_state = AgentState()
            # Update agent's position to match round_state
            self.agent.pos = self.round_state.agent_pos
            # Get agent action using the agent's action method
            agent_action = self.agent.action(agent_state)
            
            # Store old state for comparison
            old_pos = self.round_state.agent_pos
            
            # Update state
            new_round_state, events = step_round(self.game_state, self.round_state, agent_action)
            self.round_state = new_round_state
            self.log_round_state("After step_round") # Log round state after update
            
            # If position changed, we need to redraw
            if old_pos != self.round_state.agent_pos:
                self.needs_full_redraw = True
            
            # Process events
            for event in events:
                if isinstance(event, RoundEnded):
                    self.phase = Phase.SUMMARY
                    self.round_state.is_moving = False
                    self.game_state.wave_counter += 1
                    self.needs_full_redraw = True
            
            self.log_game_state("After processing events in update") # Log game state
            self.log_round_state("After processing events in update") # Log round state
        
        elif self.phase == Phase.SUMMARY:
            # Wait for any key to continue
            if self.current_command:
                self.phase = Phase.BUILD
                self.round_state = init_round(self.game_state)
                self.current_command = ""
                self.needs_full_redraw = True
                self.log_game_state("After SUMMARY phase, new round initialized") # Log game state
                self.log_round_state("After SUMMARY phase, new round initialized") # Log round state
    
    def render(self) -> None:
        """Render all windows."""
        # We optimize rendering now - always render the field for better animation,
        # but be selective about the info pane
        
        # Only redraw info pane if something relevant changed
        game_state_changed = (self.last_game_state is None or 
                             self.game_state.grid_width != self.last_game_state.grid_width or
                             self.game_state.grid_height != self.last_game_state.grid_height or
                             self.game_state.wave_counter != self.last_game_state.wave_counter or
                             self.game_state.lives != self.last_game_state.lives or
                             self.game_state.currency != self.last_game_state.currency)
                             
        round_state_changed = (self.last_round_state is None or
                              self.round_state.agent_pos != self.last_round_state.agent_pos or
                              self.round_state.is_moving != self.last_round_state.is_moving or
                              self.round_state.agent_health != self.last_round_state.agent_health)
                              
        phase_changed = (self.last_phase != self.phase)
        
        draw_info_pane = (self.needs_full_redraw or game_state_changed or round_state_changed or phase_changed)
        
        # Always draw CLI as input might have changed
        draw_cli.render(self.windows["cli"], self.command_history, self.current_command, self.error_message)
        
        # Always draw the field for smooth animation
        draw_field.render(self.windows["field"], self.game_state, self.round_state)
            
        # Only redraw info pane when needed
        if draw_info_pane:
            draw_info.render(self.windows["info"], self.game_state, self.round_state, self.phase)
        
        # Update screen in one go
        curses.doupdate()
        
        # Save current state for next frame comparison
        self.last_game_state = GameState(
            grid_width=self.game_state.grid_width,
            grid_height=self.game_state.grid_height,
            wave_counter=self.game_state.wave_counter,
            currency=self.game_state.currency,
            lives=self.game_state.lives
        )
        
        if self.round_state:
            self.last_round_state = RoundState(
                agent_pos=self.round_state.agent_pos,
                agent_health=self.round_state.agent_health,
                is_moving=self.round_state.is_moving,
                projectiles=self.round_state.projectiles.copy()
            )
        
        self.last_phase = self.phase
        self.needs_full_redraw = False


def run_game(stdscr, tick_rate: float = 0.03, grid_width: int = 16, grid_height: int = 16) -> None:
    """
    Main game loop.
    """
    # Set up curses
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.refresh()
    
    # Set up logging
    logger = logging.getLogger('game')
    
    # Create and run game controller
    controller = GameController(logger, stdscr, tick_rate, grid_width, grid_height)
    
    # Frame rate control
    RENDERING_FPS = 30  # Target 30 FPS for smooth rendering but less CPU usage
    frame_duration = 1.0 / RENDERING_FPS
    
    # Main game loop
    last_update = time.time()
    last_frame_time = time.time()
    frame_count = 0
    fps_timer = time.time()
    
    while controller.running:
        frame_start_time = time.time()
        
        # Handle input
        controller.handle_input()
        
        # Update game state at the specified tick rate
        current_time = time.time()
        if current_time - last_update >= controller.tick_rate:
            controller.update()
            last_update = current_time
            
            # Always render immediately after game update to reduce perceived lag
            controller.render()
            frame_count += 1
        # Render at the target frame rate when no game update happened
        elif current_time - last_frame_time >= frame_duration:
            controller.render()
            frame_count += 1
            last_frame_time = current_time
        
        # Calculate and log FPS every second
        if current_time - fps_timer >= 1.0:
            fps = frame_count / (current_time - fps_timer)
            logger.debug(f"FPS: {fps:.1f}")
            frame_count = 0
            fps_timer = current_time
        
        # Dynamic sleep to maintain frame rate without using 100% CPU
        frame_time = time.time() - frame_start_time
        sleep_time = max(0, frame_duration - frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        # Update last_frame_time after sleep for accurate frame timing
        last_frame_time = time.time()

import curses
import time
import uuid
from typing import Dict, List, Optional, Tuple
from Training.TrainingManager import TrainingManager
from Training.ActionWrapper import ActionWrapper
from core.round_state_generator import generate_round_state
from Training.FeatureExtractor import DictPolicy
from stable_baselines3 import PPO

from core.state.game_state import GameState, Tower
from core.env import GMSEnv
from controller.actions import (
    Action, StartRound, Quit, ErrorAction,
    PlaceTower, ShowStats, ShowTowers, ShowPosition, TrainAction
)
from core.events import Event, RoundEnded
from core.round_logic.events import RoundOverEvent, TowerDestroyedEvent, AgentDamagedEvent
from core.round_logic.state import RoundState as CoreRoundState
from core.adapters.controller_adapter import ControllerAdapter
from agents.simple_agent import SimpleAgent
from agents.base import Agent as AgentBase
from agents.observation_adapter import get_observation_from_round_state
from controller.phases import Phase
from cli.commands import parse_command
from ui.curses import windows, draw_field, draw_info, draw_cli
from ui.info_display import InfoDisplayManager
from persistence.serializer import save_game_state
from maps.loader import get_map_loader
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.policies")

# Get the appropriate loggers
system_logger = logging.getLogger('system')
graphics_logger = logging.getLogger('graphics')
cli_logger = logging.getLogger('cli')


class GameController:
    """
    Main game controller that manages the game state and UI interaction.
    """
    def __init__(self, stdscr, tick_rate: float = 0.03, grid_width: int = 16, grid_height: int = 16):
        self.stdscr = stdscr
        self.tick_rate = tick_rate
        self.windows = None
        if stdscr:  # Only create windows if stdscr is provided (graphical mode)
            self.windows = windows.make_windows(stdscr)
        # Initialize loggers
        self.system_logger = system_logger
        self.graphics_logger = graphics_logger
        self.cli_logger = cli_logger
        
        # Initialize states
        self.map_name = "cross"
        self.round_state, self.game_state = generate_round_state(self.map_name)
        self.last_observation = get_observation_from_round_state(self.round_state)
        round_states = {
            self.map_name: self.round_state
        }
        env_config = {
            'round_states': round_states,
            'max_episode_steps': 1000
        }
        self.env = ActionWrapper(GMSEnv(env_config))
        
        # Game control
        self.phase = Phase.BUILD
        self.running = True
        
        # Command line state
        self.command_history: List[str] = []
        self.current_command = ""
        self.error_message: Optional[str] = None
        
        # Initialize round state and agent
        self._initialize_round()
        
        # Force full redraw on first frame
        self.needs_full_redraw = True
        self.last_game_state = None
        self.last_round_state = None
        self.last_phase = None
    
    def _initialize_round(self):
        """Initialize the round state and agent."""
        # Create a new controller round state using the adapter
        self.round_state = ControllerAdapter.initialize_round_state(self.game_state)
        round_states = {
            self.map_name: self.round_state
        }
        env_config = {
            'round_states': round_states,
            'max_episode_steps': 1000
        }
        self.env = ActionWrapper(GMSEnv(env_config))
        # Create a new agent at the agent position
        #self.agent = PPO.load("Training/models/p6g174ek/model.zip", custom_objects={"policy_class": DictPolicy}, verbose=0)
        self.agent = SimpleAgent()
        self.last_observation = get_observation_from_round_state(self.round_state)
        
        self.system_logger.info(f"Round initialized with agent at {self.round_state.position}")
    
    def process_action(self, user_action: Action) -> List[Event]:
        """Process a player action and return events."""

        if isinstance(user_action, StartRound):
            if self.phase == Phase.BUILD:
                self.phase = Phase.ROUND
                self.needs_full_redraw = True
                self.system_logger.debug("Starting new round")
            else:
                self.error_message = "Cannot start movement during active round"
                self.cli_logger.warning("Attempted to start round during active round")
        
        elif isinstance(user_action, Quit):
            self.running = False
            #save_game_state(self.game_state, "latest_game.json")
            self.system_logger.info("Game quit, state saved")
        
        elif isinstance(user_action, TrainAction):
            # Run training mode
            self.system_logger.info(f"Starting training mode with {user_action.iterations} iterations")
            self.error_message = None
            
            # Train using the training manager
            training_result = TrainingManager.run_training(
                self.game_state, 
                self.agent,
                user_action.iterations
            )
            
            # Update the game state if needed
            # (training runs on a copy, so original state is preserved)
            
            # Mark for redraw
            self.needs_full_redraw = True
            
        elif isinstance(user_action, PlaceTower):
            self._handle_place_tower(user_action)
            
        elif isinstance(user_action, ShowStats):
            # Switch to stats view
            info_manager = InfoDisplayManager()
            info_manager.show_stats()
            self.needs_full_redraw = True
            
        elif isinstance(user_action, ShowTowers):
            # Switch to towers view
            info_manager = InfoDisplayManager()
            info_manager.show_towers()
            self.needs_full_redraw = True
            
        elif isinstance(user_action, ShowPosition):
            # Show information about a specific position
            info_manager = InfoDisplayManager()
            info_manager.show_position_info(user_action.x, user_action.y)
            self.needs_full_redraw = True
        
        elif isinstance(user_action, ErrorAction):
            self.error_message = user_action.message
    
    def _handle_place_tower(self, action: PlaceTower) -> None:
        """Handle placing a tower action."""
        
        if self.phase != Phase.BUILD:
            self.error_message = "Can only place towers during build phase"
            self.cli_logger.warning(f"Attempted to place tower during {self.phase} phase")
            return
            
        # Check if coordinates are within the grid
        if not self.game_state.is_position_valid(action.x, action.y):
            self.error_message = "Cannot place tower at that position"
            self.cli_logger.warning(f"Invalid tower position attempted: ({action.x}, {action.y})")
            return
            
        # Check if there's already a tower at this position
        for tower in self.game_state.towers:
            if tower.position == (action.x, action.y):
                self.error_message = "There's already a tower at this position"
                self.cli_logger.warning(f"Tower already exists at ({action.x}, {action.y})")
                return
        
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
        
        # Place the tower with proper initialization of all fields
        tower_id = str(uuid.uuid4())
        self.game_state.towers.append(Tower(
            position=(action.x, action.y),
            direction=direction_vector,
            tower_id=tower_id,
        ))
        self.system_logger.info(f"Tower placed at ({action.x}, {action.y}) with direction {direction_vector}, id={tower_id}")
        self.needs_full_redraw = True
        
        # If the info display is showing towers, update it
        info_manager = InfoDisplayManager()
        if info_manager.display_state.is_towers_view:
            info_manager.show_towers()  # Refresh the view
            
        return
    
    def handle_input(self) -> Optional[Action]:
        """Handle keyboard input."""
        if not self.stdscr:  # Skip if in non-graphical mode
            return None
            
        self.stdscr.timeout(0)  # Non-blocking input
        try:
            key = self.stdscr.getch()
            if key == curses.ERR:
                return None  # No input available
            
            # Handle special keys
            if key == curses.KEY_BACKSPACE or key == 127:
                self.current_command = self.current_command[:-1]
            elif key == curses.KEY_ENTER or key == 10 or key == 13:
                # Process command
                if self.current_command:
                    self.command_history.append(self.current_command)
                    
                    # Parse command
                    action = parse_command(self.current_command)
                    
                    # Clear command line
                    self.current_command = ""
                    self.error_message = None
                    
                    return action
            elif key == 27:  # ESC key
                return Quit()
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.current_command += chr(key)
                
        except Exception as e:
            self.error_message = f"Error: {str(e)}"
            self.cli_logger.error(f"Input error: {str(e)}")
            
        return None
    
    def step_game(self) -> None:
        """
        Process one game step/tick based on current phase.
        This is the main update function for the game loop.
        """
        if self.phase == Phase.ROUND:
            # Get the current position for redraw check
            old_position = self.round_state.position

            action = self.agent.act(self.last_observation)
            system_logger.info(f"Action: {action}")
            action = self.env.reverse_action(action)
            system_logger.info(f"Action: {action}")

            obs, reward, done, truncated, info = self.env.step(action)

            self.last_observation = obs
            self.round_state = (self.env).get_round_state()

            # Process events
            events = info.get('events', [])
            for event in events:
                if isinstance(event, RoundOverEvent):
                    # Synchronize tower states from round_state to game_state
                    # This ensures destroyed towers keep their sprite_name_override
                    for rt in self.round_state.towers:
                        for gt in self.game_state.towers:
                            if gt.tower_id == rt.tower_id:
                                gt.health = rt.health
                                if hasattr(rt, 'sprite_name_override') and rt.sprite_name_override:
                                    gt.sprite_name_override = rt.sprite_name_override
                    
                    self.phase = Phase.SUMMARY
                    self.game_state.wave_counter += 1
                    self.system_logger.info("Round ended, transitioning to SUMMARY phase")
                    self.needs_full_redraw = True
                elif isinstance(event, TowerDestroyedEvent):
                    # Set sprite_name_override for destroyed towers
                    tower_id = event.tower_id
                    for tower in self.round_state.towers:
                        if tower.tower_id == tower_id:
                            tower.sprite_name_override = "tower_rubble"
                            # Also update game_state towers for rendering
                            for game_tower in self.game_state.towers:
                                if game_tower.tower_id == tower_id:
                                    game_tower.health = 0
                                    game_tower.sprite_name_override = "tower_rubble"
                            break
                elif isinstance(event, AgentDamagedEvent):
                    # Log agent damage events
                    self.system_logger.info(f"Agent damaged! Health: {event.health_remaining}")
                    self.needs_full_redraw = True
            
            # If position changed or phase changed, we need to redraw
            if old_position != self.round_state.position or done:
                self.needs_full_redraw = True
    
    def render(self) -> None:
        """Render all windows."""
        if not self.windows:  # Skip rendering in non-graphical mode
            return
            
        # Only redraw info pane if something relevant changed
        game_state_changed = (self.last_game_state is None or 
                              self.game_state.grid_width != self.last_game_state.grid_width or
                              self.game_state.grid_height != self.last_game_state.grid_height or
                              self.game_state.wave_counter != self.last_game_state.wave_counter or
                              self.game_state.lives != self.last_game_state.lives or
                              self.game_state.currency != self.last_game_state.currency)
                             
        round_state_changed = (self.last_round_state is None or
                               self.round_state.position != self.last_round_state.position or
                               self.round_state.is_moving != self.last_round_state.is_moving or
                               self.round_state.health != self.last_round_state.health)
                              
        phase_changed = (self.last_phase != self.phase)
        
        draw_info_pane = (self.needs_full_redraw or game_state_changed or 
                          round_state_changed or phase_changed)
        
        # Always draw CLI as input might have changed
        draw_cli.render(self.windows["cli"], self.command_history, 
                        self.current_command, self.error_message)
        
        # Always draw the field for smooth animation
        draw_field.render(self.windows["field"], self.game_state, self.round_state)
            
        # Only redraw info pane when needed
        if draw_info_pane:
            draw_info.render(self.windows["info"], self.game_state, 
                             self.round_state, self.phase)
        
        # Update screen in one go
        curses.doupdate()
        
        # Save current state for next frame comparison
        self._save_state_for_comparison()
    
    def _save_state_for_comparison(self):
        """Save current state for comparison in the next frame."""
        self.last_game_state = GameState(
            grid_width=self.game_state.grid_width,
            grid_height=self.game_state.grid_height,
            wave_counter=self.game_state.wave_counter,
            currency=self.game_state.currency,
            lives=self.game_state.lives
        )
        
        if self.round_state:
            self.last_round_state = CoreRoundState(
                grid_width=self.round_state.grid_width,
                grid_height=self.round_state.grid_height,
                grid_layout=self.round_state.grid_layout,
                position=self.round_state.position,
                health=self.round_state.health,
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
    
    # Create game controller
    controller = GameController(stdscr, tick_rate, grid_width, grid_height)
    
    # Frame rate control
    RENDERING_FPS = 30  # Target 30 FPS for smooth rendering
    frame_duration = 1.0 / RENDERING_FPS
    
    # Timing variables
    last_update = time.time()
    last_frame_time = time.time()
    frame_count = 0
    fps_timer = time.time()
    
    # Performance tracking
    input_time_total = 0
    update_time_total = 0
    render_time_total = 0
    
    system_logger.info(f"Starting game loop with tick_rate={tick_rate}, target FPS={RENDERING_FPS}")
    
    # Main game loop
    while controller.running:
        frame_start_time = time.time()
        
        # Handle input
        input_start = time.time()
        action = controller.handle_input()
        input_time = time.time() - input_start
        input_time_total += input_time
        
        # Process action if one was returned
        if action:
            events = controller.process_action(action)
        
        # Update game state at the specified tick rate
        current_time = time.time()
        if current_time - last_update >= controller.tick_rate:
            update_start = time.time()
            controller.step_game()
            update_time = time.time() - update_start
            update_time_total += update_time
            last_update = current_time
            
            # Render immediately after update
            render_start = time.time()
            controller.render()
            render_time = time.time() - render_start
            render_time_total += render_time
            frame_count += 1
        
        # Render at target frame rate when no update happened
        elif current_time - last_frame_time >= frame_duration:
            render_start = time.time()
            controller.render()
            render_time = time.time() - render_start
            render_time_total += render_time
            frame_count += 1
            last_frame_time = current_time
        
        # Log performance stats periodically
        if current_time - fps_timer >= 1.0:
            fps = frame_count / (current_time - fps_timer)
            avg_input_time = input_time_total / max(1, frame_count) * 1000  # ms
            avg_update_time = update_time_total / max(1, frame_count) * 1000  # ms
            avg_render_time = render_time_total / max(1, frame_count) * 1000  # ms
            
            graphics_logger.debug(
                f"Performance: FPS={fps:.1f}, "
                f"Input={avg_input_time:.2f}ms, "
                f"Update={avg_update_time:.2f}ms, "
                f"Render={avg_render_time:.2f}ms, "
                f"Total={avg_input_time + avg_update_time + avg_render_time:.2f}ms"
            )
            
            # Reset tracking
            frame_count = 0
            fps_timer = current_time
            input_time_total = 0
            update_time_total = 0
            render_time_total = 0
        
        # Sleep to maintain frame rate
        frame_time = time.time() - frame_start_time
        sleep_time = max(0, frame_duration - frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

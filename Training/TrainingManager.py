from core.state.game_state import GameState, Tower
from agents.base import Agent
import logging
from core.adapters.controller_adapter import ControllerAdapter
from agents.observation_adapter import get_observation_from_round_state
from core.env import GMSEnv
from typing import Dict
import time

system_logger = logging.getLogger("system")

class TrainingManager:
    """
    Manager for training agents without UI interaction.
    """
    @staticmethod
    def run_training(game_state: GameState, agent: Agent, iterations: int) -> Dict:
        """
        Run training on a copy of the provided game state using the specified agent class.
        
        Args:
            game_state: Game state to use as template
            iterations: Number of iterations to run
           
        Returns:
            Dictionary with training results
        """
        system_logger.info(f"Starting training mode for {iterations} iterations")
        system_logger.info(f"Using game state with {len(game_state.towers)} towers")
        
        # Create a copy of the game state
        training_game_state = GameState(
            grid_width=game_state.grid_width,
            grid_height=game_state.grid_height,
            wave_counter=game_state.wave_counter,
            currency=game_state.currency,
            lives=game_state.lives,
            current_map=game_state.current_map,
        )
        
        # Copy towers
        training_game_state.towers = [
            Tower(
                position=tower.position,
                direction=tower.direction,
                health=100,  # Reset to full health
                rate=tower.rate,
                tick=tower.tick,
                tower_id=tower.tower_id,
                sprite_name_override=None
            ) for tower in game_state.towers
        ]
        
        # Training statistics
        training_stats = {
            "iterations_completed": 0,
            "total_reward": 0,
            "avg_steps_per_round": 0,
            "win_rate": 0,
        }
        
        total_steps = 0
        wins = 0
        round_state = ControllerAdapter.initialize_round_state(training_game_state)
        env_config = {
            'round_state': round_state,
            'max_episode_steps': 1000
        }
        env = GMSEnv(env_config)

        try:
            # Run the specified number of iterations
            for i in range(iterations):
                system_logger.info(f"Starting training iteration {i+1}/{iterations}")
               
                # Reset towers for this iteration
                for tower in training_game_state.towers:
                    tower.health = 100
               
                agent.reset()
                env.reset()
               
                # Log initial state
                system_logger.info(f"Iteration {i+1} starting with {len(training_game_state.towers)} towers")
                for idx, tower in enumerate(training_game_state.towers):
                    system_logger.info(f"Tower {idx+1} at {tower.position}: health={tower.health}, destroyed={tower.health <= 0}")
               
                # Run until the round is over
                done = False
                iteration_reward = 0
                tick_count = 0
                max_ticks = 1000  # Safety limit to prevent infinite loops
               
                while not done and tick_count < max_ticks:

                    observation = get_observation_from_round_state(round_state)
                    # Process step in core logic
                    next_round_state, reward, done, info = env.step(agent.act(observation))
                   
                    round_state = next_round_state
                   
                    # Log movement for debugging
                    if tick_count % 10 == 0:
                        system_logger.info(f"Agent at {round_state.position}")
                   
                    # Update agent with feedback
                    agent.observe(get_observation_from_round_state(round_state), reward, done, info)
                   
                    # Update for next iteration
                    iteration_reward += reward
                   
                    tick_count += 1
                   
                    # Small delay to prevent 100% CPU usage
                    time.sleep(0.0001)
               
                # Update training statistics
                total_steps += tick_count
                training_stats["total_reward"] += iteration_reward
               
                # Determine if this was a win (all towers destroyed)
                all_towers_destroyed = all(tower.health <= 0 for tower in round_state.towers)
                if all_towers_destroyed:
                    wins += 1
               
                # Log completion of this iteration
                system_logger.info(f"Completed training iteration {i+1}/{iterations} after {tick_count} ticks")
                system_logger.info(f"Reward: {iteration_reward}, Win: {all_towers_destroyed}")
               
                # Log the final state
                system_logger.info(f"Final agent position: {round_state.position}, health: {round_state.health}")
                for idx, tower in enumerate(round_state.towers):
                    system_logger.info(f"Tower {idx+1} at {tower.position}: health={tower.health}, destroyed={tower.health <= 0}")
               
                # Briefly wait between iterations
                time.sleep(0.0001)
           
            # Compute final statistics
            training_stats["iterations_completed"] = iterations
            training_stats["avg_steps_per_round"] = total_steps / iterations if iterations > 0 else 0
            training_stats["win_rate"] = wins / iterations if iterations > 0 else 0
           
            system_logger.info(f"Training completed: {iterations} iterations, {wins} wins")
            system_logger.info(f"Average steps per round: {training_stats['avg_steps_per_round']}")
           
        except Exception as e:
            system_logger.error(f"Error during training: {str(e)}")
            import traceback
            system_logger.error(traceback.format_exc())
        
        return {}
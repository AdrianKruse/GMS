import logging
import os

def setup_logging():
    """Set up logging configuration with separate loggers for different components."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Common formatter for all loggers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # System logger (core game mechanics, state management, etc.)
    system_logger = logging.getLogger('system')
    system_handler = logging.FileHandler(os.path.join(logs_dir, 'system.log'), mode='w')
    system_handler.setFormatter(formatter)
    system_logger.addHandler(system_handler)
    system_logger.setLevel(logging.DEBUG)

    # Graphics logger (rendering, sprites, UI updates)
    graphics_logger = logging.getLogger('graphics')
    graphics_handler = logging.FileHandler(os.path.join(logs_dir, 'graphics.log'), mode='w')
    graphics_handler.setFormatter(formatter)
    graphics_logger.addHandler(graphics_handler)
    graphics_logger.setLevel(logging.DEBUG)

    # CLI logger (command processing, user input)
    cli_logger = logging.getLogger('cli')
    cli_handler = logging.FileHandler(os.path.join(logs_dir, 'cli.log'), mode='w')
    cli_handler.setFormatter(formatter)
    cli_logger.addHandler(cli_handler)
    cli_logger.setLevel(logging.DEBUG)

    # Agent logger (agent actions, observations, etc.)
    agent_logger = logging.getLogger('agent')
    agent_handler = logging.FileHandler(os.path.join(logs_dir, 'agent.log'), mode='w')
    agent_handler.setFormatter(formatter)
    agent_logger.addHandler(agent_handler)
    agent_logger.setLevel(logging.DEBUG)

    core_logger = logging.getLogger('core')
    core_handler = logging.FileHandler(os.path.join(logs_dir, 'core.log'), mode='w')
    core_handler.setFormatter(formatter)
    core_logger.addHandler(core_handler)
    core_logger.setLevel(logging.DEBUG)

    # Root logger for any uncategorized logs
    root_logger = logging.getLogger()
    root_handler = logging.FileHandler(os.path.join(logs_dir, 'game.log'), mode='w')
    root_handler.setFormatter(formatter)
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.DEBUG)

    # Prevent log propagation to avoid duplicate entries
    system_logger.propagate = False
    graphics_logger.propagate = False
    cli_logger.propagate = False

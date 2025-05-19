import logging


def setup_logs():
    logging.basicConfig(
        filename='game.log',
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w'
    )

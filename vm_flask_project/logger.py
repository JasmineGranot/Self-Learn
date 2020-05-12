import logging
from pathlib import Path

LOG_FILE_PATH = Path('file.log')


def create_logger(name):
    logger = logging.getLogger(name)

    # remove existing log file
    # Path(LOG_FILE_PATH).unlink(missing_ok=True)

    # Create file handler
    f_handler = logging.FileHandler(LOG_FILE_PATH)
    f_handler.setLevel(logging.INFO)
    # f_handler.setLevel(logging.ERROR)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

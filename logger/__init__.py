import logging
from datetime import datetime

# Configure the logger
logging.basicConfig(level=logging.INFO, format="INFO: %(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def info(message):
    """Log an informational message with the current time."""
    logging.info(message)



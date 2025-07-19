"""
This module stores the main logger object used to tracking the code.
"""
import logging
from enum import Enum

from torch.utils.tensorboard import SummaryWriter


# Enums.
class LevelName(Enum):
    """
    Enum that stores all level name available.
    """
    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


# Functions.
def get_logger(name, level = "info"):
    """
    Creates a logger object.

    Args:
        name (str): Name of the logger.
        level (str): Level name to be used. (Default: info)
    
    Returns:
        logging.Logger: Logger object.
    """
    # Define formatter for logger.
    message_fmt = "[%(levelname)-8s]: %(asctime)s - %(message)s"
    date_fmt = None
    fmt = logging.Formatter(fmt=message_fmt, datefmt=date_fmt)

    # Define handler.
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(fmt=fmt)

    # Define logger.
    log = logging.Logger(name=name, level=LevelName[level].value)
    log.addHandler(hdlr=hdlr)

    return log


# Classes.
class Tensorboard:


    # Special methods.
    def __init__(self, log_dir):
        """
        Initializes the Tensorboard writer.
        
        Args:
            log_dir (str): Directory where the logs will be saved.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

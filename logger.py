"""
Main module that contains the logger class of the project.
"""
import logging
import logging.handlers
import os


# Classes.
class Logger(logging.Logger):


    # Create logs directory.
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(name=LOG_DIR, exist_ok=True)


    # Special methods.
    def __init__(self, name, **kwargs):
        """
        Initializes a logger object used to track all execution steps of the project.

        Args:
            name (str): The name of the logger.
        """
        super().__init__(name=name)

        # Create handler.
        max_bytes = kwargs.get("max_bytes", 1024 ** 2 * 2) # 2 MB.

        handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(self.LOG_DIR, "%s.log" % name), 
            mode="w", 
            maxBytes=max_bytes, 
            backupCount=5
        )
        handler.setLevel(level=logging.DEBUG)

        # Set formatter.
        fmt = logging.Formatter(fmt="[%(levelname)-8s]: %(asctime)s - %(message)s")
        handler.setFormatter(fmt)

        # Add handler to logger.
        self.addHandler(handler)
    
        # Set logger level.
        self.setLevel(level=logging.DEBUG)

"""
This module stores the main logger object used to tracking the code.
"""
import logging
import os
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw
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
        log_dir = os.path.join(log_dir, "tensorboard")
        self.writer = SummaryWriter(log_dir=log_dir)


    # Methods.
    def add_train_loss(self, loss, step):
        """
        Adds the training loss to the Tensorboard writer.

        Args:
            loss (float): Training loss value.
            step (int): Step number.
        """
        self.writer.add_scalar(tag="train_loss", scalar_value=loss, global_step=step)


    def add_valid_loss(self, loss, step):
        """
        Adds the validation loss to the Tensorboard writer.

        Args:
            loss (float): Validation loss value.
            step (int): Step number.
        """
        self.writer.add_scalar(tag="valid_loss", scalar_value=loss, global_step=step)


    def add_image(self, samples, step):
        """
        Displays the predictions of the model on the target
        image.

        Args:
            samples (List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]): List of tuples containing the image, label, and prediction.
            step (int): Step number.
        """
        # Prepare the images for Tensorboard.
        tb_samples = []
        for (img, label, prediction) in samples:

            # Get the image dimensions.
            height, width = img.shape[:2]

            # Compute real coordinates.
            label[:, 0::2] *= width
            label[:, 1::2] *= height
            prediction[:, 0::2] *= width
            prediction[:, 1::2] *= height

            prediction[:, 2] = prediction[:, 0] + prediction[:, 2]
            prediction[:, 3] = prediction[:, 1] + prediction[:, 3]
            label[:, 2] = label[:, 0] + label[:, 2]
            label[:, 3] = label[:, 1] + label[:, 3]

            # Draw the rectangles on the image.
            pil_img = Image.fromarray((img * 255).astype("uint8"))
            draw = ImageDraw.Draw(im=pil_img)

            for box in label:
                draw.rectangle(xy=tuple(box), outline="green", width=2)
            for box in prediction:
                try:
                    draw.rectangle(xy=tuple(box), outline="red", width=2)
                except ValueError:
                    # If the box is invalid, skip it.
                    continue

            # Append the image to the samples.
            tb_samples.append(pil_img)

        # Add the images to the Tensorboard writer.
        for idx, sample in enumerate(iterable=tb_samples):
            self.writer.add_image(tag=f"sample_{idx}", img_tensor=np.asarray(sample), global_step=step, dataformats="HWC")

    def close(self):
        """
        Closes the Tensorboard writer.
        """
        self.writer.close()

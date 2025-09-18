"""
This module stores the main logger object used to tracking the code.
"""
import logging
import os
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter

from data.daug import PrepareImage


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
    def add_current_lr(self, lr, step):
        """
        Adds the current learning rate to the Tensorboard writer.

        Args:
            lr (float): Current learning rate value.
            step (int): Step number.
        """
        self.writer.add_scalar(tag="lr", scalar_value=lr, global_step=step)


    def add_train_losses(self, loss, bbox_loss, presence_loss, step):
        """
        Adds the training loss to the Tensorboard writer.

        Args:
            loss (float): Total training loss value.
            bbox_loss (float): Training L1 loss value.
            presence_loss (float): Training presence loss value.
            step (int): Step number.
        """
        self.writer.add_scalar(tag="train_loss", scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag="train_l1_loss", scalar_value=bbox_loss, global_step=step)
        self.writer.add_scalar(tag="train_presence_loss", scalar_value=presence_loss, global_step=step)


    def add_train_giou_accuracy(self, acc, step, th):
        """
        Adds the training GIoU accuracy to the Tensorboard writer.

        Args:
            acc (float): Total training accuracy value.
            step (int): Step number.
            th (float): Threshold used for the GIoU accuracy.
        """
        tag = "train_GIoU@%f" % th
        self.writer.add_scalar(tag=tag, scalar_value=acc, global_step=step)


    def add_train_f1_accuracy(self, acc, step, th):
        """
        Adds the training F1 accuracy to the Tensorboard writer.

        Args:
            acc (float): Total training accuracy value.
            step (int): Step number.
            th (float): Threshold used for the F1 accuracy.
        """
        tag = "train_F1@%f" % th
        self.writer.add_scalar(tag=tag, scalar_value=acc, global_step=step)


    def add_valid_losses(self, loss, bbox_loss, presence_loss, step):
        """
        Adds the validation loss to the Tensorboard writer.

        Args:
            loss (float): Total validation loss value.
            bbox_loss (float): Validation L1 loss value.
            presence_loss (float): Validation presence loss value.
            step (int): Step number.
        """
        self.writer.add_scalar(tag="valid_loss", scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag="valid_l1_loss", scalar_value=bbox_loss, global_step=step)
        self.writer.add_scalar(tag="valid_presence_loss", scalar_value=presence_loss, global_step=step)


    def add_valid_giou_accuracy(self, acc, step, th):
        """
        Adds the validation GIoU accuracy to the Tensorboard writer.

        Args:
            acc (float): Total validation accuracy value.
            step (int): Step number.
            th (float): Threshold used for the GIoU accuracy.
        """
        tag = "valid_GIoU@%f" % th
        self.writer.add_scalar(tag=tag, scalar_value=acc, global_step=step)


    def add_valid_f1_accuracy(self, acc, step, th):
        """
        Adds the validation F1 accuracy to the Tensorboard writer.

        Args:
            acc (float): Total validation accuracy value.
            step (int): Step number.
            th (float): Threshold used for the F1 accuracy.
        """
        tag = "valid_F1@%f" % th
        self.writer.add_scalar(tag=tag, scalar_value=acc, global_step=step)


    def add_image(self, samples, step, giou_th):
        """
        Displays the predictions of the model on the target
        image.

        Args:
            samples (List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]): List of tuples containing the image, label, and prediction.
            step (int): Step number.
            giou_th (float): GIoU threshold used for the predictions.
        """
        tb_samples = []
        # Prepare the images for Tensorboard.
        size = 5
        for (img, caption, label, prediction) in samples:

            # Draw the rectangles on the image.
            height, width = img.shape[:2]
            img = PrepareImage.de_normalize(normalized_image=img)
            pil_img = Image.fromarray(img.astype("uint8"))
            draw = ImageDraw.Draw(im=pil_img)

            for bbox in label:
                draw.rectangle(xy=tuple(bbox), outline="lime", width=2)
            for bbox_array in prediction:
                for bbox in bbox_array:
                    try:
                        draw.rectangle(xy=tuple(bbox), outline="red", width=2)
                    except ValueError:
                        # If the bbox is invalid, skip it.
                        continue

            # Write the caption on the image.
            font = ImageFont.load_default()
            n_digits = len(caption)
            draw.rectangle(xy=(0, 0, 10 + n_digits * 7, 20), fill="black")
            draw.text((5, 2), caption, fill="white", font=font)

            # Append the image to the samples.
            tb_samples.append(pil_img)

        # Add the images to the Tensorboard writer.
        for idx, sample in enumerate(iterable=tb_samples):
            self.writer.add_image(tag=f"[{idx} - GIoU@{giou_th}]", img_tensor=np.asarray(sample), global_step=step, dataformats="HWC")

    def close(self):
        """
        Closes the Tensorboard writer.
        """
        self.writer.close()

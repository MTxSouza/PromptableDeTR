"""
Main module that starts the UI data label for PromptVision project. This UI allows 
the developer to label image text with a bounding box and assign it to a prompt.
"""
import os
import sys
import tkinter as tk
from configparser import ConfigParser

sys.path.append(os.path.abspath(path=os.path.dirname(p=__file__.split(sep=os.sep)[-2])))

from data.label_ui.image_frame.image_frame import ImageFrame
from data.label_ui.side_bar.side_bar import SideBar


# Functions.
def load_config_file(config_filepath):
    """
    Load the a config file for any software.

    Args:
        config_filepath (str): Path to the .cnf file.
    
    Returns:
        configparser.ConfigParser: Config parser object.
    """
    # Define parser.
    config = ConfigParser()

    # Load file.
    config.read(filenames=config_filepath)

    return config


# Classes.
class LabelUI(tk.Tk):


    def __init__(self, config):
        """
        Initialize the main UI for data labeling in PromptVision project.

        Args:
            config (configparser.ConfigParser): Config parser object with window settings.
        """
        super().__init__()

        # Define setting variables.
        main_window_setting = config["mainWindowSettings"]
        WINDOW_NAME = main_window_setting["windowName"]
        MIN_WINDOW_WIDTH_SIZE = main_window_setting.getint("minWidthSize")
        MIN_WINDOW_HEIGHT_SIZE = main_window_setting.getint("minHeightSize")

        window_palette_color = config["windowPaletteColor"]
        FRAME_COLOR = window_palette_color["frameColor"]

        top_bar_settings = config["topBarSettings"]
        TOP_BAR_HEIGHT_SIZE = top_bar_settings.getint("heightSize")

        # Configure main window.
        self.title(string=WINDOW_NAME)
        self.geometry(newGeometry="%dx%d" % (MIN_WINDOW_WIDTH_SIZE, MIN_WINDOW_HEIGHT_SIZE))
        self.resizable(width=True, height=True)
        self.grid_rowconfigure(index=[0, 1], weight=0)
        self.grid_rowconfigure(index=2, weight=1)
        self.grid_columnconfigure(index=1, weight=1)

        # Define side bar.
        self.side_bar_frame = SideBar(config=config, master=self)
        self.side_bar_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky="nsw")

        # Define top bar label.
        self.image_dir_var_label = tk.StringVar()
        self.update_image_directory_path(dirpath="*no directory selected")
        tk.Label(
            textvariable=self.image_dir_var_label, 
            height=TOP_BAR_HEIGHT_SIZE, 
            bg=FRAME_COLOR, 
            anchor="w"
        ).grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        self.output_dir_var_label = tk.StringVar()
        self.update_output_directory_path(dirpath="*no directory selected")
        tk.Label(
            textvariable=self.output_dir_var_label, 
            height=TOP_BAR_HEIGHT_SIZE, 
            bg=FRAME_COLOR, 
            anchor="w"
        ).grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        # Define main image frame.
        self.image_frame = ImageFrame(config=config, master=self)
        self.image_frame.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")


    # Methods.
    def update_image_directory_path(self, dirpath):
        """
        Update the text where the image directory path is 
        displayed.

        Args:
            dirpath (str): Directory path.
        """
        self.image_dir_var_label.set(value="Image directory: %s" % dirpath)


    def update_output_directory_path(self, dirpath):
        """
        Update the text where the output directory path is 
        displayed.

        Args:
            dirpath (str): Directory path.
        """
        self.output_dir_var_label.set(value="Output directory: %s" % dirpath)


if __name__=="__main__":

    # Load window configs.
    config_filepath = os.path.join(os.path.abspath(path=os.path.dirname(p=__file__)), "ui.cfg")
    config = load_config_file(config_filepath=config_filepath)

    # Run interface.
    label_ui = LabelUI(config=config)
    label_ui.mainloop()

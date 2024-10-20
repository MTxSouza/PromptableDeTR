"""
"""
import tkinter as tk
from tkinter import filedialog

from data.label_ui.side_bar.directory_browser import (DirectoryBrowser,
                                                      DirectoryOutput)


# Classes.
class SideBar(tk.Frame):


    def __init__(self, config, *args, **kwargs):
        """
        Initialize the side bar to be displayed on main window.

        Args:
            config (configparser.ConfigParser): Config parser object with window settings.
        """
        # Define settings variables.
        side_bar_setting = config["sideBarSettings"]
        MIN_SIDE_BAR_WIDTH_SIZE = side_bar_setting.getint("minWidthSize")
        
        window_palette_color = config["windowPaletteColor"]
        FRAME_COLOR = window_palette_color["frameColor"]

        super().__init__(
            width=MIN_SIDE_BAR_WIDTH_SIZE, 
            bg=FRAME_COLOR, 
            *args, 
            **kwargs
        )

        # Define directory browser.
        self.directory_browser = DirectoryBrowser(config=config, master=self)
        self.directory_browser.grid(row=0, column=0, padx=5, pady=5, sticky="new")

        # Define directory output.
        self.directory_output = DirectoryOutput(config=config, master=self)
        self.directory_output.grid(row=0, column=1, padx=5, pady=5, sticky="new")

        # Display file names.
        tk.Label(master=self, text="Images:", bg=FRAME_COLOR, anchor="w").grid(row=1, column=0, sticky="new")
        self.file_name_list = tk.Listbox(master=self, listvariable=tk.Variable(value=[]))
        self.file_name_list.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="new")

        self.display_image_selected_bt = tk.Button(
            master=self, 
            text="Label selected image", 
            state="disabled", 
            command=self.show_selected_image
        )
        self.display_image_selected_bt.grid(row=3, column=0, columnspan=2, padx=5, pady=1, sticky="new")


    # Methods.
    def check_directories(self):
        """
        Check if both image directory and output directory 
        were selected.
        """
        # Check if both were defined.
        if not self.directory_browser.current_directory_content or not self.directory_output.current_directory:

            # Disable annotation buttons.
            self.display_image_selected_bt["state"] = "disabled"
            return

        # Enable annotation buttons.
        self.master.image_frame.update_image(image_index=0)
        self.display_image_selected_bt["state"] = "normal"


    def show_selected_image(self):
        """
        Update the current image on canvas by the 
        selected image.
        """
        # Get current index.
        for index in self.file_name_list.curselection():
            
            # Update canvas.
            self.master.image_frame.update_image(image_index=index)

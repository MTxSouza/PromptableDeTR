"""
"""
import json
import os
import re
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from tkinter import filedialog


# Enums.
class ImageFormat(Enum):
    """
    Define all available image formats.
    """
    jpg = "jpg"
    png = "png"
    jpeg = "jpeg"


# Structures.
@dataclass
class ImageData:
    """
    Structure that stores information for all images 
    retrived from any location in computer.
    """
    file_name: str
    full_path: str
    file_format: ImageFormat
    size: int


@dataclass
class Boxes:
    """
    Sctructure that stores the coordinates of a bounding 
    box.
    """
    cx: int
    cy: int
    width: int
    height: int


@dataclass
class Annotation:
    """
    Structure that stores the prompt and all bounding 
    boxes of that prompt.
    """
    prompt: str
    boxes: list[Boxes]


@dataclass
class ImageAnnotation:
    """
    Sctructure that stores the annotations of an 
    image.
    """
    file_name: str
    full_path: str
    image_name: str
    image_path: str
    annots: list[Annotation]


# Classes.
class _BaseDirectoryBrowser:


    # Methods.
    def _open_browser(self, initial_directory):
        """
        Open a dialog screen to select the directory to 
        load the images.

        Args:
            initial_directory (str); Initial directory to starts the browser.

        Returns:
            tuple[list, str]: List with all files inside the directory and the selected directory path
        """
        # Check if the dirpath is empty.
        if not initial_directory:
            initial_directory = os.getcwd()

        # Update current directory.
        current_directory = filedialog.askdirectory(
            title="Directory browser", 
            initialdir=initial_directory
        )

        # Load content of directory.
        if not current_directory:
            return [], ""

        return os.listdir(path=current_directory), current_directory


class DirectoryBrowser(tk.Button, _BaseDirectoryBrowser):


    def __init__(self, config, *args, **kwargs):
        """
        Initializes the directory browser object that filters 
        all images available inside a directory.

        Args:
            config (configparser.ConfigParser): Config parser object with window settings.
        """
        # Define settings variables.
        side_bar_setting = config["sideBarSettings"]

        BUTTON_WIDTH_SIZE = side_bar_setting.getint("browserWidthSize")

        super().__init__(
            text="Open image directory", 
            width=BUTTON_WIDTH_SIZE, 
            command=self._open_browser, 
            *args, 
            **kwargs
        )

        # Attributes.
        self.__current_directory = ""
        self.__current_directory_content = []


    def __getitem__(self, index):
        """
        Get the corresponding file from `current_directory_content` 
        given it index.

        Returns:
            ImageData: Image file data.
        """
        return self.current_directory_content[index]


    # Properties.
    @property
    def current_directory(self):
        """
        Get the current directory selected.

        Returns:
            str: Directory path selected.
        """
        return self.__current_directory


    @property
    def current_directory_content(self):
        """
        Get the content of current directory selected. Note 
        that, it will filter only image files.

        Returns:
            List[ImageData]: All images available inside the current directory.
        """
        return self.__current_directory_content


    # Methods.
    def _open_browser(self):
        """
        Open a dialog screen to select the directory to 
        load the images.
        """
        # Get content of directory.
        directory_content, current_directory = super()._open_browser(initial_directory=self.current_directory)
        self.__current_directory = current_directory

        # Retrieve images from directory.
        self.__current_directory_content.clear()
        for content_name in directory_content:

            # Get full path.
            full_path = os.path.join(self.current_directory, content_name)

            # Check content type.
            if not os.path.isfile(path=full_path):
                continue

            # Check file format.
            file_format = content_name.split(sep=".")[-1].lower()
            if not file_format in ImageFormat._value2member_map_:
                continue

            self.__current_directory_content.append(ImageData(
                file_name=content_name, 
                full_path=full_path, 
                file_format=file_format,
                size=os.path.getsize(filename=full_path)
            ))

        # Update current directory text.
        self.master.master.update_image_directory_path(dirpath=self.current_directory)

        # Update file list.
        self.master.file_name_list.delete(first=0, last=self.master.file_name_list.size() - 1)
        for index, content in enumerate(iterable=self.current_directory_content):
            self.master.file_name_list.insert(index, content.file_name)

        # Update image frame.
        self.master.master.image_frame.load_images_to_canvas(image_data_list=self.current_directory_content)

        # Enable/Disable annotation buttons.
        self.master.check_directories()


class DirectoryOutput(tk.Button, _BaseDirectoryBrowser):


    def __init__(self, config, *args, **kwargs):
        """
        Initializes the directory output object that saves 
        all annotations.

        Args:
            config (configparser.ConfigParser): Config parser object with window settings.
        """
        # Define settings variables.
        side_bar_setting = config["sideBarSettings"]

        BUTTON_WIDTH_SIZE = side_bar_setting.getint("outputWidthSize")

        super().__init__(
            text="Save at", 
            width=BUTTON_WIDTH_SIZE, 
            *args, 
            **kwargs
        )

        # Attributes.
        self.__annot_pattern = re.compile(pattern=r"^[a-z0-9_]+_\[annot\]\.json$")
        self.__current_directory = ""
        self.__current_annotations = []


    # Properties.
    @property
    def current_directory(self):
        """
        Get the current output directory selected.

        Returns:
            str: Directory path selected.
        """
        return self.__current_directory


    @property
    def current_annotations(self):
        """
        Get the current annotations inside the current 
        directory selected.

        Returns:
            list[ImageAnnotation]: List with all annotations inside the directory.
        """
        return self.__current_annotations


    # Methods.
    def _open_browser(self):
        """
        Open a dialog screen to select the output directory 
        to save the annotations.
        """
        # Get content of directory.
        directory_content, current_directory = super()._open_browser(initial_directory=self.current_directory)
        self.__current_directory = current_directory

        # Retrieve annotation from directory.
        self.__current_annotations.clear()
        for content_name in directory_content:

            # Get full path.
            full_path = os.path.join(self.current_directory, content_name)

            # Check content type.
            if not os.path.isfile(path=full_path):
                continue

            # Check file format.
            file_format = content_name.split(sep=".")[-1].lower()
            if not file_format == "json":
                continue

            # Load annotation.
            with open(file=full_path, mode="r") as file_buffer:
                annot_content = json.load(fp=file_buffer)

            
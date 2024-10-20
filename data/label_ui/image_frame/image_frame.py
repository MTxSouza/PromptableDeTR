"""
"""
import tkinter as tk

from PIL import Image, ImageTk


# Classes.
class ImageFrame(tk.Frame):


    def __init__(self, config, *args, **kwargs):
        """
        Initialize the image frame used to draw bounding boxes 
        on images.

        Args:
            config (configparser.ConfigParser): Config parser object with window settings.
        """
        # Define settings variables.
        window_palette_color = config["windowPaletteColor"]
        FRAME_COLOR = window_palette_color["frameColor"]

        super().__init__(
            bg="black", 
            *args, 
            **kwargs
        )

        # Configure frame.
        self.grid_rowconfigure(index=0, weight=1)
        self.grid_columnconfigure(index=0, weight=1)

        # Define main frame for image display.
        self.image_frame_display = tk.Canvas(master=self, bg=FRAME_COLOR)
        self.image_frame_display.pack(expand=True, padx=5, pady=5, fill="both", anchor="center")

        # Define mouse events.
        self.image_frame_display.bind(sequence="<ButtonPress-1>", func=self.__on_mouse_press)
        self.image_frame_display.bind(sequence="<B1-Motion>", func=self.__on_mouse_move)
        self.image_frame_display.bind(sequence="<ButtonRelease-1>", func=self.__on_mouse_release)

        # Attributes.
        self.__images = []
        self.__image_on_canvas = []

        self.__start_x = None
        self.__start_y = None
        self.__rect = None


    # Methods.
    def load_images_to_canvas(self, image_data_list):
        """
        Load multiple images to be viewed on canvas.

        Args:
            image_data_list (List[ImageData]): List with all images to be loaded.
        """
        # Get canvas dimension.
        c_width, c_height = self.image_frame_display.winfo_width(), self.image_frame_display.winfo_height()

        # Add images.
        self.__images.clear()
        for image in image_data_list:

            # Load image.
            with Image.open(fp=image.full_path, mode="r") as pil_img:

                # Resize image to fit canvas.
                i_width, i_height = pil_img.size
                new_width = min(c_width, i_width)
                new_height = min(c_height, i_height)
                pil_img = pil_img.resize(size=(new_width, new_height))

                self.__images.append(ImageTk.PhotoImage(image=pil_img))

        # Create image.
        half_width, half_height = self.image_frame_display.winfo_width() // 2, self.image_frame_display.winfo_height() // 2
        self.__image_on_canvas = self.image_frame_display.create_image(
            half_width, 
            half_height, 
            anchor="center", 
            image=self.__images[0] if self.__images else None
        )


    def update_image(self, image_index):
        """
        Display an image on canvas.

        Args:
            image_index (int): Index of image already loaded to be displayed.
        """
        # Check image list size.
        if not self.__images:
            return

        # Display new image.
        self.image_frame_display.itemconfigure(tagOrId=self.__image_on_canvas, image=self.__images[image_index])


    # Mouse methods.
    def __on_mouse_press(self, event):
        """
        Used to trigger any event when the mouse was pressed 
        over the canvas.

        Args:
            event (tkinter.Event): Tkinter event object.
        """
        # Check if there is any image on canvas.
        if not self.master.side_bar_frame.file_name_list.size():
            return

        # Save current coordinates.
        self.__start_x = event.x
        self.__start_y = event.y

        # Draw rectangle.
        if not self.__rect:
            self.__rect = self.image_frame_display.create_rectangle(
                self.__start_x, 
                self.__start_y, 
                self.__start_x, 
                self.__start_y, 
                outline="lime", 
                width=3
            )


    def __on_mouse_move(self, event):
        """
        Used to trigger any event when the mouse was moved 
        over the canvas.

        Args:
            event (tkinter.Event): Tkinter event object.
        """
        # Get current mouse coordinates.
        current_x = self.image_frame_display.canvasx(screenx=event.x)
        current_y = self.image_frame_display.canvasy(screeny=event.y)

        # Check canvas limit.
        c_w, c_h = self.image_frame_display.winfo_width() - 3, self.image_frame_display.winfo_height() - 3
        current_x = min(c_w, current_x) if current_x > 0 else 1
        current_y = min(c_h, current_y) if current_y > 0 else 1

        # Update rectangle shape.
        self.image_frame_display.coords(self.__rect, self.__start_x, self.__start_y, current_x, current_y)


    def __on_mouse_release(self, event):
        """
        Used to trigger any event when the mouse was released 
        over the canvas.

        Args:
            event (tkinter.Event): Tkinter event object.
        """
        # Release current rectangle.
        self.__rect = None

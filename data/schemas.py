"""
This module stores all schemas and structures used to load and store the data 
used during training.
"""
import os
from dataclasses import dataclass
from enum import Enum, EnumMeta


# Enums.
class CustomEnumMeta(EnumMeta):


    def __contains__(self, value):
        """
        Check if the value is in the enum.

        Args:
            value (str): The value to check.
        """
        return any(value == item.name for item in self)


class _ObjectCategory(Enum, metaclass=CustomEnumMeta):
    """
    Define all object categories available on dataset.
    """
    apple           = "Apple"
    airplane        = "Airplane"
    bed             = "Bed"
    bus             = "Bus"
    bear            = "Bear"
    bird            = "Bird"
    boat            = "Boat"
    bowl            = "Bowl"
    book            = "Book"
    bench           = "Bench"
    banana          = "Banana"
    bottle          = "Bottle"
    bicycle         = "Bicycle"
    backpack        = "Backpack"
    broccoli        = "Broccoli"
    baseball_bat    = "Baseball Bat"
    baseball_glove  = "Baseball Glove"
    car             = "Car"
    cat             = "Cat"
    cow             = "Cow"
    cup             = "Cup"
    cake            = "Cake"
    chair           = "Chair"
    clock           = "Clock"
    couch           = "Couch"
    carrot          = "Carrot"
    cell_phone      = "Cell Phone"
    dog             = "Dog"
    donut           = "Donut"
    dining_table    = "Dining Table"
    elephant        = "Elephant"
    fork            = "Fork"
    frisbee         = "Frisbee"
    fire_hydrant    = "Fire Hydrant"
    giraffe         = "Giraffe"
    horse           = "Horse"
    handbag         = "Handbag"
    hot_dog         = "Hot Dog"
    hair_drier      = "Hair Drier"
    kite            = "Kite"
    knife           = "Knife"
    keyboard        = "Keyboard"
    laptop          = "Laptop"
    mouse           = "Mouse"
    microwave       = "Microwave"
    motorcycle      = "Motorcycle"
    oven            = "Oven"
    orange          = "Orange"
    pizza           = "Pizza"
    person          = "Person"
    potted_plant    = "Potted Plant"
    parking_meter   = "Parking Meter"
    remote          = "Remote"
    refrigerator    = "Refrigerator"
    skis            = "Skis"
    sink            = "Sink"
    sheep           = "Sheep"
    spoon           = "Spoon"
    sandwich        = "Sandwich"
    scissors        = "Scissors"
    suitcase        = "Suitcase"
    skateboard      = "Skateboard"
    snowboard       = "Snowboard"
    stop_sign       = "Stop Sign"
    surfboard       = "Surfboard"
    sports_ball     = "Sports Ball"
    tv              = "TV"
    tie             = "Tie"
    train           = "Train"
    truck           = "Truck"
    toilet          = "Toilet"
    toaster         = "Toaster"
    teddy_bear      = "Teddy Bear"
    toothbrush      = "Toothbrush"
    tennis_racket   = "Tennis Racket"
    traffic_light   = "Traffic Light"
    umbrella        = "Umbrella"
    vase            = "Vase"
    wine_glass      = "Wine Glass"
    zebra           = "Zebra"

    # Super categories.
    animal      = "Animal"
    accessory   = "Accessory"
    appliance   = "Appliance"
    electronic  = "Electronic"
    food        = "Food"
    furniture   = "Furniture"
    indoor      = "Indoor"
    kitchen     = "Kitchen"
    outdoor     = "Outdoor"
    sports      = "Sports"
    vehicle     = "Vehicle"


class LabelCategory(Enum):
    """
    Define all type of categories for annotations.
    """
    object  = "Object"
    color   = "Color"
    shape   = "Shape"


# Structures.
@dataclass
class Boxes:
    """
    Structure that store the coordinates of a bounding box 
    on image.
    """
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class Annotation:
    """
    A complete schema that stores all informations and metadata 
    for a particular sample.
    """
    text: str
    image_id: int
    image_filepath: str
    category: LabelCategory
    annotations: list[Boxes]


    def __post_init__(self):
        """
        Validate the schema after instantiation.
        """
        # Check if the image exists.
        if not os.path.isfile(self.image_filepath):
            raise ValueError("Invalid image path %s." % self.image_filepath)

        # Check if the category is valid.
        if not self.category in LabelCategory:
            raise ValueError("Invalid category %s." % self.category)

        elif self.category == LabelCategory.object:
            # Check if the category is a list of ObjectCategory.
            if not self.text in _ObjectCategory:
                raise ValueError("Invalid object category %s." % self.text)

        # Check if the annotations is a list of Boxes.
        if not all(isinstance(ann, Boxes) for ann in self.annotations):
            raise ValueError("Invalid annotations. It must be a list of Boxes.")


    def model_dump(self):
        """
        Dump the annotation to a dictionary.
        """
        return {
            "text": self.text,
            "image_id": self.image_id,
            "image_filepath": self.image_filepath,
            "category": self.category,
            "annotations": [
                {
                    "cx": ann.cx,
                    "cy": ann.cy,
                    "width": ann.width,
                    "height": ann.height
                }
                for ann in self.annotations
            ]
        }

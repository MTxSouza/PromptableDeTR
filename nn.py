"""
Main module that contains the PromptableDeTR model class.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image

from data.daug import PrepareImage
from data.schemas import Boxes, Output
from models.detector import PromptableDeTR as _PromptableDeTR
from models.tokenizer import Tokenizer


# Classes.
class PromptableDeTRModel:
    """
    A class representing the PromptableDeTR model used for inference.
    """
    
    # Special methods.
    def __init__(self, model_weights: str, vocab_path: str, force_cpu: bool = False) -> None:
        """
        Initializes the PromptableDeTR model.

        Args:
            model_weights (str): Path to the model weights file.
            vocab_path (str): Path to the vocabulary file.
        """
        # Load weight file to get the model
        # config.
        weights_data = torch.load(f=model_weights, map_location="cpu")
        assert "configs" in weights_data, "Model config not found in the weights file."
        model_config = weights_data["configs"]

        # Create the model.
        self.__estimator = _PromptableDeTR(**model_config)
        self.__estimator.load_weights(model_weights=model_weights)

        # Create the tokenizer.
        self.__tokenizer = Tokenizer(vocab_filepath=vocab_path)

        # Atributes.
        self.__device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        self.__description = None
        self.__description_embedding = None
        self.__image_size = model_config["image_size"]

        if force_cpu:
            self.__device = torch.device(device="cpu")
        self.__estimator = self.__estimator.to(self.__device)
        self.__estimator.eval()

        image_formatter = PrepareImage()
        self.__img_mean = image_formatter.mean.to(device=self.__device)
        self.__img_std = image_formatter.std.to(device=self.__device)
        del image_formatter

    def __call__(self, *args, **kwds):
        return self.infer_image(*args, **kwds)

    # Properties.
    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model is running.

        Returns:
            torch.device: The device (CPU or GPU).
        """
        return self.__device

    @property
    def emb(self):
        return self.__description_embedding

    # Methods.
    def set_description(self, description: str) -> None:
        """
        Set the description for the model.

        Args:
            description (str): The description to set.
        """
        # Tokenize description.
        desc_tokens = self.__tokenizer.encode(texts=description)
        desc_tokens = torch.tensor(desc_tokens)
        desc_tokens = desc_tokens.to(device=self.__device)

        # Generate description embedding.
        self.__description_embedding = self.__estimator.text_encoder(desc_tokens)
        self.__description = description

    def infer_image(self, image: str | np.ndarray | torch.Tensor, confidence_threshold: float = 0.5) -> list:
        """
        Perform inference on a single image.

        Args:
            image (str | np.ndarray | torch.Tensor): The input image, which can be a file path, a NumPy array, or a PyTorch tensor.
            confidence_threshold (float): The confidence threshold for filtering detections. (Default: 0.5)

        Returns:
            list: A list of detected objects with their bounding boxes and labels.
        """
        # Check if description is set.
        if self.__description is None:
            raise RuntimeError("Description is not set. Please set the description before inference.")

        # Prepare image.
        if isinstance(image, str):
            with Image.open(fp=image) as pil_img:
                pil_img = pil_img.convert(mode="RGB")
                image = np.asarray(a=pil_img)
        if isinstance(image, np.ndarray):
            image = torch.tensor(data=image).permute(2, 0, 1)  # Change HWC to CHW
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        assert image.ndim == 4 and image.size(1) == 3, "Input image must have shape (B, 3, H, W)"
        image = image.to(device=self.__device)

        original_height, original_width = image.size(2), image.size(3)
        image = tf.resize(img=image, size=[self.__image_size, self.__image_size], antialias=True).float()
        image = (image - self.__img_mean) / self.__img_std

        # Perform inference.
        with torch.no_grad():
            img_embedding = self.__estimator.image_encoder(image)
            joint_out, _, _ = self.__estimator.joiner(img_embedding, self.__description_embedding)
            bbox = F.sigmoid(self.__estimator.bbox_predictor(joint_out))
            presence = self.__estimator.presence_predictor(joint_out)

        # Post-process output.
        logits = torch.cat(tensors=(bbox, presence), dim=-1)
        logits[:, :, 4:] = logits[:, :, 4:].softmax(dim=-1)
        logits = logits[logits[:, :, 5] >= confidence_threshold]
        logits = logits.cpu().numpy().tolist()

        # Convert logits to Output format.
        outputs = Output(description=self.__description, boxes=[])
        for logit in logits:
            cx, cy, w, h, _, conf = logit
            box = Boxes(
                x1=(cx - 0.5 * w) * original_width,
                y1=(cy - 0.5 * h) * original_height,
                x2=(cx + 0.5 * w) * original_width,
                y2=(cy + 0.5 * h) * original_height,
                confidence=conf
            )
            outputs.boxes.append(box)

        # Process output.
        return outputs

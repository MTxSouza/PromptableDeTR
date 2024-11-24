"""
This script is used to display the model architecture.
"""
import argparse

# Define arguments.
parser = argparse.ArgumentParser(description="Display the model architecture.")
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="The verbosity level of the model architecture."
)

# Parse the arguments.
args = parser.parse_args()


# Display the model architecture.
if __name__=="__main__":
    # Imports.
    import os
    import sys

    # Get `models/__init__.py` path.
    app_path = os.path.join(os.path.abspath(path=os.path.dirname(p=__file__)), "__init__.py")
    assert os.path.exists(path=app_path), "Could not localize the `__init__.py` module."

    # Append the path to the system.
    sys.path.append(os.path.abspath(path=os.path.dirname(p=__file__.split(sep=os.sep)[-1])))

    # Imports.
    from warnings import warn

    import torch
    from torchinfo import summary

    from models import PromptVisionTrainer

    # Check current device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("The model is being run on the CPU. This will be slow.")

    # Create the model object.
    model = PromptVisionTrainer(
        num_text_encoder_layers=4, 
        num_image_encoder_layers=6, 
        text_encoder_hidden_dim=1024, 
        image_encoder_hidden_dim=2048, 
        num_heads=8, 
        embedding_dim=768, 
        context_length=196, 
        image_size=224, 
        num_patches=14, 
        num_points=8, 
        padding_idx=0
    )
    model.to(device=device)

    # Display the model architecture.
    image_tensor = torch.randn(1, 3, 224, 224)
    tokenized_text_tensor = torch.randint(0, 100, (1, 196))
    summary(
        model=model, 
        input_data=(tokenized_text_tensor, image_tensor), 
        device=device,
        verbose=args.verbose
    )

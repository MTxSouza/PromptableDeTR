"""
This script is used to display the model architecture.
"""
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
        text_encoder_name="bert-base-uncased",
        num_image_decoder_blocks=6,
        num_image_decoder_heads=8,
        image_decoder_hidden_dim=2048
    )

    # Display the model architecture.
    image_tensor = torch.randn(1, 3, 196, 768)
    tokenized_text_tensor = torch.randint(0, 100, (1, 196))
    summary(
        model=model, 
        input_data=(image_tensor, tokenized_text_tensor), 
        device=device
    )

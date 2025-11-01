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
parser.add_argument(
    "--image-encoder-weights",
    "--image-weights",
    "-imgw",
    type=str,
    default=None,
    help="Path to the image encoder weights."
)
parser.add_argument(
    "--text-encoder-weights",
    "--text-weights",
    "-txtw",
    type=str,
    default=None,
    help="Path to the text encoder weights."
)

# Parse the arguments.
args = parser.parse_args()


# Display the model architecture.
if __name__=="__main__":

    # Imports.
    import os
    import sys
    import time
    from warnings import warn

    import torch
    from torchsummary import summary

    # Get `models/__init__.py` path.
    app_path = os.path.join(os.path.abspath(path=os.path.dirname(p=__file__)), "__init__.py")
    assert os.path.exists(path=app_path), "Could not localize the `__init__.py` module."

    # Append the path to the system.
    sys.path.append(os.path.abspath(path=os.path.dirname(p=__file__.split(sep=os.sep)[-1])))

    from models import PromptableDeTR

    # Check current device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        warn("The model is being run on the CPU. This will be slow.")

    # Create the model object.
    image_size = 320
    text_size = 50
    model = PromptableDeTR(image_size=image_size)
    model.load_base_weights(
        image_encoder_weights=args.image_encoder_weights, 
        text_encoder_weights=args.text_encoder_weights
    )
    model.to(device=device)

    # Display the model architecture.
    image_shape = (1, 3, image_size, image_size)
    text_shape = (1, text_size)
    summary(
        model=model, 
        input_size=(image_shape, text_shape), 
        device=device, 
        verbose=args.verbose
    )

    # Compute inference time.
    @torch.no_grad()
    def infer(image, text):
        model(image=image, prompt=text)

    def compute_inference_time(image_size, text_size):

        base_image = torch.randn(size=image_size).to(device=device)
        base_text = torch.randint(low=0, high=30522, size=text_size).to(device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        infer(image=base_image, text=base_text)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time

        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return inference_time

    print("Warm-up the model...")
    compute_inference_time(image_size=image_shape, text_size=text_shape)
    time.sleep(1)
    print("Warm-up completed.\n")

    print("Computing single inference time...")
    infer_time = compute_inference_time(image_size=image_shape, text_size=text_shape)
    print(f"Single inference time: {infer_time:.4f} seconds.\n")

    print("Compute batch inference time...")
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        image_shape = (batch_size, 3, image_size, image_size)
        text_shape = (batch_size, text_size)
        print(f"Batch size: {batch_size}")
        try:
            infer_time = compute_inference_time(image_size=image_shape, text_size=text_shape)
        except torch.OutOfMemoryError:
            print("Could not compute the inference time due to memory error.\n")
            break
        else:
            print(f"Batch inference time: {infer_time:.4f} seconds\n")

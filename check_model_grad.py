from pathlib import Path

import torch

from data.daug import PrepareSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.detector import PromptableDeTRTrainer

if __name__=="__main__":
    
    # Instantiate the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    model = PromptableDeTRTrainer().to(device)
    model.load_base_weights(
        image_encoder_weights="/home/mtxsouza/Downloads/mobilenetv3.pth", 
        text_encoder_weights="/home/mtxsouza/Downloads/mobilebert.pth"
    )

    # Prepare mock sample.
    n_batch = 16
    sample_files = list(Path("/home/mtxsouza/workspace/PromptableDeTR/.samples").glob("*.json"))
    loader = PromptableDeTRDataLoader(
        sample_file_paths=sample_files,
        batch_size=n_batch,
        transformations=[
            PrepareSample(vocab_file="/home/mtxsouza/Downloads/mobilebert.vocab"),
            ReshapeImage(image_size=320)
        ]
    )
    for batch in loader:
        images, captions, mask, extra_data = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch, max_len=model.image_context_length)
        points = extra_data["points"]
        images, captions, mask, points = images.to(device), captions.to(device), mask.to(device), points.to(device)
        print("Batch image shape:", images.size())
        print("Batch captions shape:", captions.size())
        print("Batch mask shape:", mask.size())
        print("Batch labels shape:", points.size())
        break

    # Forward pass.
    logits = model(images, captions)
    print("Model outputs:", logits.size())

    # Compute gradients.
    loss_map = model.compute_loss_and_accuracy(logits, points)
    loss = loss_map["loss"]
    print("Loss:", loss.item())
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"Layer: {name}, Gradient shape: {param.grad.shape} | Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

"""
This script simply checks the output of the data loader of the Aligner model. It will log the following 
information:
- Image shape.
- Input tokens.
- Input text.
- Target tokens.
- Target text.
"""
import os
import sys

import torch

# Add the project directory to the path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.daug import MaskCaption, PrepareAlignerSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.aligner import Aligner
from models.tokenizer import Tokenizer
from params import get_args

if __name__=="__main__":

    # Get the arguments.
    args = get_args()

    # Prepare datasets.
    print("Creating train and validation splits.")
    train_files, valid_files = PromptableDeTRDataLoader.get_train_val_split(
        sample_directory=args.dataset_dir,
        val_split=args.valid_split,
        shuffle_samples=True,
        seed=args.seed
    )

    # Instantiate the data loader.
    print("Creating the data loader for the training set.")
    dataset = PromptableDeTRDataLoader(
        sample_file_paths=train_files,
        image_directory=args.image_dir,
        batch_size=args.batch_size,
        transformations=[
            PrepareAlignerSample(vocab_file=args.vocab_file),
            ReshapeImage(image_size=args.image_size),
            MaskCaption(vocab_file=args.vocab_file, mask_token=103, mask_ratio=args.mask_ratio),
        ],
        aligner=True,
    )

    # Instantiate the model.
    print("Creating the model.")
    model = Aligner(
        image_tokens=args.image_tokens,
        vocab_size=30522,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        emb_dropout_rate=args.emb_dropout_rate,
    )

    # Instantiate Tokenizer.
    tokenizer = Tokenizer(vocab_filepath=args.vocab_file)

    # Load the model weights.
    print("Loading the model weights.")
    model.load_base_weights(
        image_encoder_weights=args.image_encoder_weights,
        text_encoder_weights=args.text_encoder_weights,
    )

    # Get the first batch.
    print("Getting the first batch.")
    for batch in dataset:
        if isinstance(batch, (list, tuple)):
            batch = batch[1]
        break

    # Perform a forward pass.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    with torch.no_grad():

        image = batch.image.unsqueeze(dim=0).to(device=device)
        caption_tokens = batch.caption_tokens.unsqueeze(dim=0).to(device=device)
        masked_caption_tokens = batch.masked_caption_tokens.unsqueeze(dim=0).to(device=device)

        mask = caption_tokens.clone()
        mask[masked_caption_tokens == 0] = 103 # Mask token.

        caption = batch.caption
        masked_caption = tokenizer.decode(indices=mask.squeeze(dim=0).cpu().tolist())

        print("Image shape:", image.size())
        print("Input tokens:", caption_tokens.size(), caption_tokens)
        print("Masked tokens:", masked_caption_tokens.size(), masked_caption_tokens)
        print("Input text:", caption)
        print("Masked text:", masked_caption)

        out = model(image=image, prompt=masked_caption_tokens)
        print("Output tokens:", out.size())

        out_caption = out.argmax(dim=-1)
        out_caption = tokenizer.decode(indices=out_caption.squeeze(dim=0).cpu().tolist())
        print("Output text:", out_caption)

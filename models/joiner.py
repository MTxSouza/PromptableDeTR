"""
This module contains the main block used to unify both text and image embeddings 
to be feed into the transformer model.
"""
import torch
import torch.nn as nn


# Classes.
class PositionalEncoding(nn.Module):


    # Special methods.
    def __init__(
            self, 
            high_feat_resolution = (40, 40), 
            emb_dim = 512
        ):
        """
        Positional encoding layer used to encode different levels of image features 
        with positional information.

        Args:
            high_feat_resolution (tuple): Resolution of the high-level image feature. (Default: (40, 40))
            emb_dim (int): Dimension of the embedding. (Default: 512)
        """
        super().__init__()

        # Check if the embedding dimension is divisible 
        # by 2.
        if emb_dim % 2 != 0:
            raise ValueError("Embedding dimension must be divisible by 2.")
        half_dim = emb_dim // 2

        col_pe_len, row_pe_len = high_feat_resolution

        # Layers.
        self.col_emb = nn.Embedding(num_embeddings=col_pe_len, embedding_dim=half_dim)
        self.row_emb = nn.Embedding(num_embeddings=row_pe_len, embedding_dim=half_dim)


    # Methods.
    def forward(self, image_feature):
        """
        Forward pass of the positional encoding layer.

        Args:
            image_feature (torch.Tensor): Image feature tensor.

        Returns:
            torch.Tensor: Positional encoding of the image feature tensor.
        """
        # Get shape of the image feature tensor.
        if not len(image_feature.size()) == 4:
            raise ValueError("Image feature tensor must be 4D.")
        B, _, H, W = image_feature.size()

        # Compute range of positional encodings.
        col_range = torch.arange(W, device=image_feature.device)
        row_range = torch.arange(H, device=image_feature.device)

        # Project positional encodings.
        col_emb = self.col_emb(col_range)
        row_emb = self.row_emb(row_range)

        # Concatenate positional encodings.
        pe = torch.cat(tensors=[
            col_emb.unsqueeze(0).repeat(H, 1, 1), 
            row_emb.unsqueeze(1).repeat(1, W, 1)
        ], dim=-1)

        # Reshape to match the image feature 
        # size.
        pe = pe.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        return pe


class Joiner(nn.Module):


    # Special methods.
    def __init__(self, *image_token_dim, emb_dim = 512, context_window_size = 64, high_feat_resolution = (40, 40)):
        super().__init__()

        # Layers.
        self.token_projection = nn.ModuleList(modules=[
            nn.Linear(in_features=feat, out_features=context_window_size) for feat in image_token_dim
        ])
        self.img_pe = PositionalEncoding(
            high_feat_resolution=high_feat_resolution, 
            emb_dim=emb_dim
        )

        num_image_feature_levels = len(image_token_dim)
        self.level_emb = nn.Parameter(data=torch.Tensor(num_image_feature_levels, emb_dim))


    # Methods.
    def forward(self, *image_features, text_embedding):
        """
        Forward pass of the joiner block.

        Args:
            image_features (tuple): Tuple containing image embedding tensors.
            text_embedding (torch.Tensor): Text embedding tensor.

        Returns:
            torch.Tensor: Joined embedding tensor.
        """
        # Apply level embedding to the image features.
        processed_image_features = []
        for level, image_feature in enumerate(iterable=image_features):

            # Flatten feature.
            flt_image_feature = image_feature.flatten(2)

            # Compute positional encoding for the
            # current feature map.
            pe = self.img_pe(image_feature)
            pe = pe.flatten(2)

            # Compute level embedding.
            level_emb = self.level_emb[level].view(1, -1, 1)

            # Add level embedding and positional encoding 
            # to the feature map.
            flt_image_feature = flt_image_feature + pe + level_emb

            processed_image_features.append(flt_image_feature)

        # Project image embeddings.
        projected_image_embeddings = [
            projection(image_embedding) for projection, image_embedding in zip(self.token_projection, processed_image_features)
        ]
        projected_image_embeddings = torch.cat(projected_image_embeddings, dim=-1)
        projected_image_embeddings = projected_image_embeddings.permute(0, 2, 1)

        # Concatenate text and image embeddings.
        joined_embedding = torch.cat(tensors=[text_embedding, projected_image_embeddings], dim=1)

        return joined_embedding

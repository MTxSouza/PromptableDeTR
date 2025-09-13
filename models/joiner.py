"""
This module contains the main block used to unify both text and image embeddings 
to be feed into the transformer model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger

# Logger.
logger = Logger(name="model")


# Classes.
class PositionalEncoding(nn.Module):


    # Special methods.
    def __init__(
            self, 
            n_positions,
            emb_dim = 512
        ):
        """
        Positional encoding layer used to encode different levels of image features 
        with positional information.

        Args:
            n_positions (int): Number of positions to encode.
            emb_dim (int): Dimension of the embeddings. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.pe = nn.Embedding(num_embeddings=n_positions, embedding_dim=emb_dim)

        self.register_buffer(name="indices", tensor=torch.arange(n_positions))


    # Methods.
    def forward(self, image_feature):
        """
        Forward pass of the positional encoding layer.

        Args:
            image_feature (torch.Tensor): Image feature tensor.

        Returns:
            torch.Tensor: Encoded image feature tensor.
        """
        logger.debug(msg="Calling `PositionalEncoding` forward method.")
        logger.debug(msg="Input shape: %s" % (image_feature.shape,))

        # Get shape of the image feature tensor.
        if not len(image_feature.size()) == 4:
            logger.error(msg="Image feature tensor must be 4D.")
            raise ValueError("Image feature tensor must be 4D.")
        
        # Flatten image feature tensor.
        image_flatten = image_feature.flatten(2).permute(0, 2, 1)
        logger.debug(msg="Flattened image feature shape: %s" % (image_flatten.shape,))

        # Compute positional encoding.
        pe = self.pe(self.indices).unsqueeze(0)
        logger.debug(msg="Positional encoding shape: %s" % (pe.shape,))
        image_flatten = image_flatten + pe

        return image_flatten


class Attention(nn.Module):


    # Special methods.
    def __init__(self, emb_proj, emb_dim = 512):
        """
        Attention layer used to compute attention scores and weights.

        Args:
            emb_proj (int): Projected dimension of the embeddings.
            emb_dim (int): Dimension of the embeddings. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.query = nn.Linear(in_features=emb_dim, out_features=emb_proj)
        self.key = nn.Linear(in_features=emb_dim, out_features=emb_proj)
        self.value = nn.Linear(in_features=emb_dim, out_features=emb_proj)
    
        # Attributes.
        self.__attention = None


    # Properties.
    @property
    def attention(self):
        """
        Get the attention weights.

        Returns:
            torch.Tensor: Attention weights tensor.
        """
        return self.__attention


    # Methods.
    def forward(self, query, key, value):
        """
        Forward pass of the attention layer.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Attended value tensor.
        """
        logger.debug(msg="Calling `Attention` forward method.")
        logger.debug(msg="Query shape: %s" % (query.shape,))
        logger.debug(msg="Key shape: %s" % (key.shape,))
        logger.debug(msg="Value shape: %s" % (value.shape,))

        # Project query, key and value tensors.
        query = self.query(query)
        logger.debug(msg="Projected query shape: %s" % (query.shape,))
        key = self.key(key)
        logger.debug(msg="Projected key shape: %s" % (key.shape,))
        value = self.value(value)
        logger.debug(msg="Projected value shape: %s" % (value.shape,))

        # Compute attention scores.
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / (key.size(-1) ** 0.5)
        logger.debug(msg="Attention scores shape: %s" % (attention_scores.shape,))

        # Compute attention weights.
        attention_weights = torch.softmax(input=attention_scores, dim=-1)
        self.__attention = attention_weights

        # Compute attended value.
        attended_value = attention_weights @ value
        logger.debug(msg="Attended value shape: %s" % (attended_value.shape,))

        return attended_value


class MultiHeadAttention(nn.Module):


    # Special methods.
    def __init__(self, num_heads = 8, emb_dim = 512):
        """
        Multi-head attention layer used to compute multiple attention heads.

        Args:
            num_heads (int): Number of heads. (Default: 8)
            emb_dim (int): Dimension of the embeddings. (Default: 512)
        """
        super().__init__()

        # Check if the embedding dimension is divisible 
        # by the number of heads.
        if emb_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")
        emb_proj = emb_dim // num_heads

        # Layers.
        self.attention_heads = nn.ModuleList(modules=[
            Attention(emb_proj=emb_proj, emb_dim=emb_dim) for _ in range(num_heads)
        ])
        self.projection = nn.Linear(in_features=emb_dim, out_features=emb_dim)
    

    # Properties.
    @property
    def attentions(self):
        """
        Get the attention weights of all attention heads.

        Returns:
            List[torch.Tensor]: List containing attention weights tensors.
        """
        return [attention_head.attention for attention_head in self.attention_heads]


    # Methods.
    def forward(self, query, key, value):
        """
        Forward pass of the multi-head attention layer.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Multi-head attended value tensor.
        """
        logger.debug(msg="Calling `MultiHeadAttention` forward method.")
        logger.debug(msg="Query shape: %s" % (query.shape,))
        logger.debug(msg="Key shape: %s" % (key.shape,))
        logger.debug(msg="Value shape: %s" % (value.shape,))

        # Compute attention heads.
        attention_heads = [
            attention_head(query, key, value) for attention_head in self.attention_heads
        ]
        attention_heads = torch.cat(tensors=attention_heads, dim=-1)
        logger.debug(msg="Attention heads shape: %s" % (attention_heads.shape,))

        # Project attention heads.
        projected_attention_heads = self.projection(attention_heads)
        logger.debug(msg="Projected attention heads shape: %s" % (projected_attention_heads.shape,))

        return projected_attention_heads


class FeedForward(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 512, ff_dim = 2048):
        """
        Feed-forward layer used to merge text and image embeddings.

        Args:
            emb_dim (int): Dimension of the embeddings. (Default: 512)
            ff_dim (int): Dimension of the feed-forward layer. (Default: 2048)
        """
        super().__init__()

        # Layers.
        self.ff = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=ff_dim),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=emb_dim)
        )


    # Methods.
    def forward(self, x):
        """
        Forward pass of the feed-forward layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        logger.debug(msg="Calling `FeedForward` forward method.")
        logger.debug(msg="Input shape: %s" % (x.shape,))

        return self.ff(x)



class JoinerBlock(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 512, num_heads = 8, ff_dim = 2048):
        """
        Joiner block used to join text and image embeddings. It consists of a multi-head 
        attention layer and a feed-forward layer used to merge both text and image embeddings.

        Args:
            emb_dim (int): Dimension of the embeddings. (Default: 512)
            num_heads (int): Number of heads in the multi-head attention layer. (Default: 8)
            ff_dim (int): Dimension of the feed-forward layer. (Default: 204
        """
        super().__init__()

        # Layers.
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, emb_dim=emb_dim)
        self.feed_forward = FeedForward(emb_dim=emb_dim, ff_dim=ff_dim)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_dim)


    # Methods.
    def forward(self, query_embedding, text_embedding, image_embedding):
        """
        Forward pass of the joiner block.

        Args:
            query_embedding (torch.Tensor): Detection embedding tensor. It will be used as the query tensor.
            text_embedding (torch.Tensor): Text embedding tensor. It will be used as the key tensor.
            image_embedding (torch.Tensor): Image embedding tensor. It will be used as the key and value tensors.

        Returns:
            torch.Tensor: Output tensor.
        """
        logger.debug(msg="Calling `JoinerBlock` forward method.")
        logger.debug(msg="Query embedding shape: %s" % (query_embedding.shape,))
        logger.debug(msg="Text embedding shape: %s" % (text_embedding.shape,))
        logger.debug(msg="Image embedding shape: %s" % (image_embedding.shape,))

        # Compute multi-head attention.
        mha_out = self.multi_head_attention(query_embedding, text_embedding, image_embedding)
        attention_output = self.norm1(query_embedding + mha_out)
        logger.debug(msg="Attention output shape: %s" % (attention_output.shape,))

        # Compute feed-forward layer.
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.norm2(attention_output + feed_forward_output)
        logger.debug(msg="Feed-forward output shape: %s" % (feed_forward_output.shape,))

        return feed_forward_output



class Joiner(nn.Module):


    # Special methods.
    def __init__(self, image_tokens, emb_dim = 512, num_heads = 8, ff_dim = 1024, num_joins = 3):
        super().__init__()

        # Prepare query vector.
        num_queries = sum(image_tokens)
        self.query_vector = nn.Parameter(data=torch.Tensor(num_queries, emb_dim))
        self.query_layer = nn.Linear(in_features=emb_dim, out_features=emb_dim)

        # Layers.
        self.img_pe = nn.ModuleList(modules=[
            PositionalEncoding(n_positions=image_token, emb_dim=emb_dim) for image_token in image_tokens
        ])
        self.joiner_blocks = nn.ModuleList(modules=[
            JoinerBlock(emb_dim=emb_dim, num_heads=num_heads, ff_dim=ff_dim) for _ in range(num_joins)
        ])

        num_image_feature_levels = len(image_tokens)
        self.level_emb = nn.Parameter(data=torch.Tensor(num_image_feature_levels, emb_dim))


    # Properties.
    @property
    def attentions(self):
        """
        Get the attention weights of all joiner blocks.

        Returns:
            List[torch.Tensor]: List containing attention weights tensors.
        """
        return [joiner_block.multi_head_attention.attentions for joiner_block in self.joiner_blocks]


    # Methods.
    def forward(self, image_features, text_embedding):
        """
        Forward pass of the joiner block.

        Args:
            image_features (list): List containing image embedding tensors.
            text_embedding (torch.Tensor): Text embedding tensor.

        Returns:
            torch.Tensor: Joined embedding tensor.
        """
        logger.debug(msg="Calling `Joiner` forward method.")

        text_embedding = text_embedding.last_hidden_state
        image_features = [
            image_features.high_resolution_feat, 
            image_features.mid_resolution_feat
        ]

        logger.debug(msg="Text embedding shape: %s" % (text_embedding.shape,))

        # Apply level embedding to the image features.
        processed_image_features = []
        for level, image_feature in enumerate(iterable=image_features):

            logger.debug(msg="Image feature shape: %s" % (image_feature.shape,))

            # Compute positional encoding.
            image_pe = self.img_pe[level](image_feature)
            logger.debug(msg="Positional encoded image feature shape: %s" % (image_pe.shape,))

            # Compute level embedding.
            level_emb = self.level_emb[level].view(1, 1, -1)
            logger.debug(msg="Level embedding shape: %s" % (level_emb.shape,))

            # Add level embedding to the feature map.
            feature = image_pe + level_emb

            processed_image_features.append(feature)

        # Project image embeddings.
        processed_image_features = torch.cat(processed_image_features, dim=1)
        logger.debug(msg="Processed image features shape: %s" % (processed_image_features.shape,))

        # Project query vector.
        query_vector = self.query_vector.unsqueeze(0).expand(text_embedding.size(0), -1, -1)
        logger.debug(msg="Query vector shape: %s" % (query_vector.shape,))
        query_vector = self.query_layer(query_vector)

        # Join text and image embeddings.
        memory = torch.cat(tensors=[text_embedding, processed_image_features], dim=1)
        for joiner_block in self.joiner_blocks:
            query_vector = joiner_block(query_vector, memory, memory)
            logger.debug(msg="Text embedding shape: %s" % (query_vector.shape,))

        return query_vector

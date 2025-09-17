

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedAttentionHead(nn.Module):
    """Single Attention Head with Masking Support.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value, with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.

    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(MaskedAttentionHead, self).__init__()

        # Initialize the linear layers required for the query, key, and value projections.
        self.wq = nn.Linear(d_model, d_q)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights with optional causal mask.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """
        # Implement the scaled dot-product attention mechanism, now with masking.
        dim_k = k.size(-1)
        scores = q @ k.transpose(1, 2) / math.sqrt(dim_k)
        masked_scores = scores.masked_fill(mask == 0, float("-inf")) if mask is not None else scores
        weights = torch.softmax(masked_scores, dim = -1)
        output = weights @ v
        return output, weights

    def forward(self, x, mask=None):
        """Forward pass for the attention head with optional causal mask.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # Implement the forward pass for the attention head, now with masking.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        return output


class MultiHeadMaskedAttention(nn.Module):
    """Multi-Head Attention mechanism with Masking Support.

    This class implements the multi-head attention mechanism, allowing
    the model to focus on different parts of the input sequence at each layer,
    with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadMaskedAttention, self).__init__()
        
        # Define the heads and linear layer
        self.num_heads = num_attention_heads
        self.d_k = d_model // num_attention_heads

        self.heads = nn.ModuleList([MaskedAttentionHead(d_model, self.d_k, self.d_k, self.d_k) for _ in range(num_attention_heads)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_state, mask=None):
        """Forward pass for the multi-head attention layer with optional causal mask.

        Args:
            hidden_state (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Implement the forward pass for the multi-head attention layer, now with masking.
        if hidden_state.size(-1) % self.num_heads != 0:
            raise RuntimeError("d_model must be divisible by num_attention_heads")
    
        head_outputs = [head(hidden_state, mask) for head in self.heads]
        concat_outputs = torch.cat(head_outputs, dim = -1)
        x = self.output_linear(concat_outputs)
        return x


class FeedForward(nn.Module):
    """FeedForward module for the Transformer.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): The first linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): The second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    This class implements a single layer of the Transformer decoder, consisting
    of a masked multi-head self-attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization before self-attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization before feed-forward network.
        self_attention (MultiHeadAttention): Masked multi-head self-attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()

        # Initialize the sub-layers
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadMaskedAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Implement the forward pass for the Transformer decoder layer
        x_ = self.attention(x, mask)
        x = self.layer_norm_1(x_ + x)

        x_ = self.feed_forward(x)
        x = self.layer_norm_2(x_ + x)

        return x


class Embeddings(nn.Module):
    """Embeddings module for the Transformer.

      This module combines token embeddings and positional embeddings and applies
      layer normalization.

      Args:
          vocab_size (int): The size of the vocabulary.
          max_position_embeddings (int): The maximum number of positions for positional embeddings.
          d_model (int): The dimension of the input embeddings.

      Attributes:
          token_embeddings (nn.Embedding): Embedding layer for token embeddings.
          position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
          layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
      """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int, dropout: float = 0.0):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        # Generate position IDs based on the input sequence length
        device = input_ids.device
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = self.dropout(self.layer_norm(token_embeddings + position_embeddings))

        return embeddings


class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    This class implements the decoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer decoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer decoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer decoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerDecoder, self).__init__()
        # Define the embeddings layer and the decoder layers
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Implement the forward pass for the Transformer decoder
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).expand(batch_size, -1, -1)
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerForLanguageModeling(nn.Module):
    """Transformer model with a language modeling head for text generation.

    Args:
        vocab_size (int): Vocabulary size.
        max_position_embeddings (int): Maximum number of position embeddings.
        d_model (int): Hidden size of the Transformer model.
        num_attention_heads (int): Number of attention heads in each decoder layer.
        intermediate_size (int): Intermediate size of the feed-forward network.
        num_hidden_layers (int): Number of Transformer decoder layers.

    Attributes:
        transformer_decoder (TransformerDecoder): The Transformer decoder.
        lm_head (nn.Linear): Linear layer mapping hidden states to vocabulary logits.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerForLanguageModeling, self).__init__()
        # Define the Transformer decoder and the language modeling head
        self.transformer_decoder = TransformerDecoder(vocab_size, max_position_embeddings, d_model, num_attention_heads, intermediate_size, num_hidden_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer model with language modeling head.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Implement the forward pass for the Transformer model
        logits = self.lm_head(self.transformer_decoder(input_ids))
        return logits

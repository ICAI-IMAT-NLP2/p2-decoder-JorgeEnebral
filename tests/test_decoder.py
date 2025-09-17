import torch
import pytest

from src.decoder import MaskedAttentionHead, MultiHeadMaskedAttention, FeedForward, TransformerDecoderLayer, TransformerDecoder, TransformerForLanguageModeling

@pytest.fixture
def seed():
    torch.manual_seed(0)

@pytest.mark.order(1)
def test_attention_head():
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = d_q = d_v = 8

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)

    attention_head = MaskedAttentionHead(d_model, d_k, d_q, d_v)
    output = attention_head(x, mask)

    assert output.shape == (batch_size, seq_len, d_v), "Output shape mismatch in AttentionHead"

@pytest.mark.order(2)
def test_multi_head_attention():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)

    multi_head_attention = MultiHeadMaskedAttention(d_model, num_heads)
    output = multi_head_attention(x, mask)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in MultiHeadAttention"

@pytest.mark.order(3)
def test_feed_forward():
    batch_size = 2
    seq_len = 4
    d_model = 8
    intermediate_size = 16

    x = torch.rand(batch_size, seq_len, d_model)
    feed_forward = FeedForward(d_model, intermediate_size)
    output = feed_forward(x)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in FeedForward"

@pytest.mark.order(4)
def test_transformer_decoder_layer():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    # Create an input tensor with known values
    x = torch.rand((batch_size, seq_len, d_model))

    # Create a mask that prevents attending to future tokens (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (1, seq_len, seq_len)

    # Instantiate the decoder layer
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, intermediate_size)

    # Pass the input through the decoder layer with the mask
    output_with_mask = decoder_layer(x, mask)

    # Pass the input through the decoder layer without the mask (full attention)
    output_no_mask = decoder_layer(x, None)

    # The outputs should be different if the mask is applied correctly
    assert not torch.allclose(output_with_mask, output_no_mask), "Mask is not being applied; outputs are identical."

@pytest.mark.order(5)
def test_transformer_decoder():
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    max_position_embeddings = 10
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder = TransformerDecoder(vocab_size, max_position_embeddings, d_model,
                                num_heads, intermediate_size, num_layers)
    output = decoder(input_ids)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch in TransformerDecoder"

@pytest.mark.order(6)
def test_transformer_for_language_modeling():
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    max_position_embeddings = 10
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = TransformerForLanguageModeling(vocab_size, max_position_embeddings, d_model,
                                            num_heads, intermediate_size, num_layers)
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch in TransformerForLanguageModeling"

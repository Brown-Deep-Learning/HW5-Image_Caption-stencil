import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMatrix(nn.Module):

    def __init__(self, use_mask=False):
        super().__init__()
        # Mask is [batch_size x window_size_queries x window_size_keys]
        self.use_mask = use_mask

    def forward(self, K, Q):
        """
        STUDENT MUST WRITE:

        Computes attention weights given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix [batch_size x window_size_queries x window_size_keys]
        """
        window_size_queries = Q.size(1)   # window size of queries
        window_size_keys    = K.size(1)   # window size of keys
        embedding_size_keys = K.size(2)

        # TODO:
        # 1) Compute raw attention scores via scaled dot-product:
        #       scores = Q @ K^T  / sqrt(embedding_size_keys)
        #    Hint: use torch.bmm for batched matrix multiplication.
        # 2) If use_mask == True, apply a causal mask so that position i
        #    cannot attend to positions j > i.  Set those positions to -inf
        #    before softmax.
        # 3) Apply softmax along the key dimension and return.

        raise NotImplementedError("AttentionMatrix Not Implemented Yet")


class AttentionHead(nn.Module):
    def __init__(self, input_size, output_size, is_self_attention):
        super().__init__()
        self.use_mask = is_self_attention

        # TODO:
        # Initialize three nn.Linear layers (no bias) that project inputs into
        # Key, Value, and Query spaces of size output_size.
        # Also create an AttentionMatrix with the appropriate mask setting.


    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Runs a single attention head.

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size]
        """

        # TODO:
        # 1) Project inputs into K, V, Q using the linear layers.
        # 2) Compute attention weights via AttentionMatrix.
        # 3) Return the weighted combination of V.

        raise NotImplementedError("AttentionHead Not Implemented Yet")


class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_sz, use_mask):
        super().__init__()

        # TODO:
        # Create 3 AttentionHead instances, each with output size emb_sz // 3.
        # After concatenating their outputs (giving a tensor of size emb_sz),
        # add a final nn.Linear(emb_sz, emb_sz) layer to combine them.


    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Runs multiheaded attention.

        Requirements:
            - 3 attention heads, each of output size emb_sz // 3
            - Concatenate the three head outputs along the last dimension
            - Pass through a final linear layer

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x emb_sz]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x emb_sz]
        """

        raise NotImplementedError("MultiHeadedAttention Not Implemented Yet")


class TransformerBlock(nn.Module):
    def __init__(self, emb_sz, multiheaded=False):
        super().__init__()

        # TODO:
        # 1) Create a self-attention layer (masked) and a cross-attention layer
        #    (unmasked).  Use AttentionHead when multiheaded=False, or
        #    MultiHeadedAttention when multiheaded=True.
        # 2) Create three nn.LayerNorm(emb_sz) layers.
        # 3) Create a two-layer feed-forward network:
        #       Linear(emb_sz, 2048) -> ReLU -> Linear(2048, emb_sz)
        #    Hint: see Section 3.3 of "Attention Is All You Need".


    def forward(self, inputs, context_sequence):
        """
        Runs one Transformer decoder block.

        :param inputs:           tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE]
        :return:                 tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]

        The block should:
          1. Self-attend over `inputs`  (masked), add residual, layer-norm.
          2. Cross-attend using context_sequence as keys/values, `inputs` as
             queries (unmasked), add residual, layer-norm.
          3. Pass through feed-forward network, add residual, layer-norm.
        """

        raise NotImplementedError("TransformerBlock Not Implemented Yet")


def positional_encoding(length, depth):
    """
    STUDENT MUST WRITE:

    Generates a sinusoidal positional encoding matrix.

    :param length: number of positions (sequence length)
    :param depth:  embedding dimension (must be even)
    :return:       torch.FloatTensor of shape [length x depth]

    Hint: use alternating sin/cos at different frequencies.
    See https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    """
    ## TODO:
    raise NotImplementedError("positional_encoding Not Implemented Yet")


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Sinusoidal positional encoding – precomputed and stored as a buffer
        # (not a trainable parameter).
        # HINT: call positional_encoding(length=window_size, depth=embed_size)
        pos_enc = positional_encoding(length=window_size, depth=embed_size)
        self.register_buffer('pos_encoding', pos_enc[:window_size, :])

    def forward(self, x):
        """
        STUDENT MUST WRITE:

        :param x: integer tensor of token ids [BATCH_SIZE x WINDOW_SIZE]
        :return:  float tensor [BATCH_SIZE x WINDOW_SIZE x EMBED_SIZE]

        Steps:
          1. Embed x with self.embedding.
          2. Scale the embeddings by sqrt(embed_size).
          3. Add self.pos_encoding (broadcast over the batch dimension).
        """
        ## TODO:
        raise NotImplementedError("PositionalEncoding Not Implemented Yet")

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformer import TransformerBlock, PositionalEncoding
except ImportError:
    try:
        from .transformer import TransformerBlock, PositionalEncoding
    except Exception as e:
        print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, window_size):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Define the following layers:
        #
        # self.image_embedding  – projects the 2048-D ResNet feature vector
        #                         down to hidden_size (a small feed-forward block
        #                         is fine).
        #
        # self.embedding        – nn.Embedding that maps token ids → hidden_size
        #                         vectors.
        #
        # self.decoder          – a GRU (or LSTM) that takes embedded tokens as
        #                         input and uses the image embedding as its initial
        #                         hidden state.  Use batch_first=True.
        #
        # self.classifier       – maps each GRU output step to a logit over the
        #                         vocabulary.


    def forward(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions:       tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits  of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]

        Steps:
          1. Project encoded_images through self.image_embedding to get the
             initial hidden state h_0 of shape [1 x BATCH_SIZE x hidden_size].
          2. Embed captions with self.embedding.
          3. Pass the embedded captions + h_0 through the GRU.
          4. Pass the GRU output sequence through self.classifier.
          5. Return logits (NOT probabilities).
        """

        # TODO:

        raise NotImplementedError("RNNDecoder Not Implemented Yet")

########################################################################################

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, window_size):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Define the following layers:
        #
        # self.image_embedding  – projects the 2048-D ResNet feature vector
        #                         to hidden_size (a small feed-forward block).
        #
        # self.encoding         – PositionalEncoding(vocab_size, hidden_size,
        #                         window_size) that embeds caption token ids.
        #
        # self.decoder          – TransformerBlock(hidden_size) that attends over
        #                         both the caption sequence and the image context.
        #
        # self.classifier       – nn.Linear(hidden_size, vocab_size) that maps
        #                         each decoder output to a logit over the vocab.
        #
        # NOTE: embedding/hidden_size must be the same for PositionalEncoding and
        # TransformerBlock – do not mix different sizes here.


    def forward(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions:       tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits  of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]

        Steps:
          1. Project encoded_images through self.image_embedding.
          2. Reshape the image embedding to [BATCH_SIZE x 1 x hidden_size] so
             it acts as a context sequence of length 1.
          3. Pass captions through self.encoding to get
             [BATCH_SIZE x WINDOW_SIZE x hidden_size].
          4. Pass the caption embeddings + image context through self.decoder.
          5. Pass the decoder output through self.classifier.
          6. Return logits (NOT probabilities).
        """

        # TODO

        raise NotImplementedError("TransformerDecoder Not Implemented Yet")

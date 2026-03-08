import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCaptionModel(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, encoded_images, captions):
        return self.decoder(encoded_images, captions)

    def compile(self, optimizer, loss, metrics):
        """
        Stores optimizer and loss/metric functions on the model so that
        train_model() and test() can use them without extra arguments.
        """
        self.optimizer        = optimizer
        self.loss_function    = loss
        self.accuracy_function = metrics[0]

    def train_epoch(self, train_captions, train_image_features, padding_index, batch_size=30):
        """
        TODO: Runs through one epoch over all training examples.

        :param train_captions:       integer tensor [N x (WINDOW_SIZE+1)] – full
                                     caption sequences including <start> and <end>.
        :param train_image_features: float tensor   [N x 2048]
        :param padding_index:        int – token id of *PAD*; used for the mask.
        :param batch_size:           int
        :return: None

        Implementation notes
        --------------------
        - The decoder input should be captions with the LAST token removed:
              [<start> w1 w2 ... wN <end>] --> [<start> w1 w2 ... wN]
        - The training labels should be captions with the FIRST token removed:
              [<start> w1 w2 ... wN <end>] --> [w1 w2 ... wN <end>]
        - Build a boolean mask (True where label != padding_index) so that
          padding positions do not contribute to the loss.
        - Shuffle training examples at the start of each epoch.
        - For each batch:
              1. Zero gradients (self.optimizer.zero_grad()).
              2. Forward pass: logits = self(batch_images, decoder_input).
              3. Compute loss: self.loss_function(logits, labels, mask).
              4. loss.backward()
              5. self.optimizer.step()
        """

        # Switch model to training mode
        super().train()

        ## TODO – implement the training loop above
        raise NotImplementedError("train_epoch Not Implemented Yet")

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        """
        DO NOT CHANGE

        Runs through one epoch over all testing examples.

        :param test_captions:        integer tensor [N x (WINDOW_SIZE+1)]
        :param test_image_features:  float tensor   [N x 2048]
        :param padding_index:        int
        :param batch_size:           int
        :returns: (perplexity, per-symbol-accuracy) on the test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0

        # Switch model to evaluation mode
        self.eval()
        with torch.no_grad():
            for index, end in enumerate(range(batch_size, len(test_captions) + 1, batch_size)):

                start = end - batch_size
                batch_image_features = test_image_features[start:end, :]
                decoder_input  = test_captions[start:end, :-1]
                decoder_labels = test_captions[start:end,  1:]

                probs = self(batch_image_features, decoder_input)
                mask  = decoder_labels != padding_index
                num_predictions = mask.float().sum()
                loss     = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)

                total_loss    += loss.item()
                total_seen    += num_predictions.item()
                total_correct += num_predictions.item() * accuracy

                avg_loss = float(total_loss / total_seen)
                avg_acc  = float(total_correct / total_seen)
                avg_prp  = np.exp(avg_loss)
                print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        return avg_prp, avg_acc


def accuracy_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Computes the batch accuracy.

    :param prbs:   float tensor  [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]  (logits)
    :param labels: integer tensor [BATCH_SIZE x WINDOW_SIZE]
    :param mask:   bool tensor    [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar float – accuracy between 0 and 1
    """
    predictions = torch.argmax(prbs, dim=-1)
    correct  = (predictions == labels) & mask
    accuracy = correct.float().sum() / mask.float().sum()
    return accuracy.item()


def loss_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Calculates the model cross-entropy loss.
    Uses reduce_sum (not reduce_mean) so that per-symbol accuracy
    can be computed correctly in the calling code.

    :param prbs:   float tensor  [batch_size x window_size x vocab_size]  (logits)
    :param labels: integer tensor [batch_size x window_size]
    :param mask:   bool tensor    [batch_size x window_size]
    :return: scalar loss tensor
    """
    B, T, V  = prbs.shape
    prbs_flat   = prbs.reshape(B * T, V)
    labels_flat = labels.reshape(B * T).long()
    mask_flat   = mask.reshape(B * T).float()

    loss_unreduced = F.cross_entropy(prbs_flat, labels_flat, reduction='none')
    loss = (loss_unreduced * mask_flat).sum()
    return loss

---
title: 'HW5 Programming: Image Captioning'
---

# HW5 Programming: Image Captioning

:::info
Programming assignment due **Tuesday, March 23th, 2025 at 6:00 PM EST**
:::

## Getting the stencil
Please click <ins>[here](https://classroom.github.com/a/Y6AX2y0i)</ins> to get the stencil code. Reference this <ins>[guide](https://hackmd.io/gGOpcqoeTx-BOvLXQWRgQg)</ins> for more information about GitHub and GitHub Classroom.

:::danger
**Do not change the stencil except where specified**. Changing the stencil's method signatures or removing pre-defined functions could result in incompatibility with the autograder and result in a low grade.

**This assignment uses PyTorch.** Make sure your environment has PyTorch installed (`pip install torch torchvision`).
:::


## Assignment Overview
In this assignment, we will be generating English language captions for images using the Flickr 8k dataset. This dataset contains over 8000 images with 5 captions apiece. Provided is a slightly-altered version of the dataset, with uncommon words removed and all sentences limited to their first 20 words to make training easier.

Follow the instructions in `preprocessing.py` to download and process the data. This should take around 10 minutes the first time and will generate a 100 MB file. The dataset file itself is around 1.1 GB and can be found [here](https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download).
Note: You may need to sign in with Google and create an account on Kaggle. Make sure to download the dataset and put the `Images/` folder and `captions.txt` in your `data/` directory before running `preprocessing.py`.


You will implement **two** different types of image captioning models that first encode an image and then decode the encoded image to generate its English language caption. The first model is based on **Recurrent Neural Networks**, while the second is based on **Transformers**. Since both models solve the same task, they share the same preprocessing, training, testing, and loss function.

To simplify the assignment and improve model quality, all images have been passed through <ins>[ResNet-50](https://arxiv.org/abs/1512.03385)</ins> to get 2048-D feature vectors. You will use these feature vectors in both models.

**Note**: A major aspect of this assignment is implementing attention and then using your attention heads to build a Transformer decoder block. Although these are the last two steps, they will probably be the most time-consuming, so plan accordingly.


## Roadmap

### Step 1: Training the Model
Both models perform the same task; they differ only in their decoder architecture. The `ImageCaptionModel` class in `model.py` wraps any decoder and provides shared `test()`, `accuracy_function()`, and `loss_function()` utilities.

**Your task**: implement `train_model()` in `assignment.py`.

**ImageCaptionModel** (`model.py`)
- `forward`: calls the decoder — already provided, do not change.
- `test`: runs evaluation over the test set — already provided, do not change.
- `compile`: stores optimizer and loss/metric functions — already provided.

**`train_model()` in `assignment.py`** — **this is your main TODO for Step 1**.

The training loop should:
1. For each epoch, shuffle the training data.
2. Iterate over mini-batches of size `batch_size`.
3. For each batch:
   - Slice decoder **input**: captions with the **last** token removed —
     `[<start> w1 w2 ... wN <end>]` → `[<start> w1 w2 ... wN]`
   - Slice decoder **labels**: captions with the **first** token removed —
     `[<start> w1 w2 ... wN <end>]` → `[w1 w2 ... wN <end>]`
   - Build a boolean mask: `True` where label `!= padding_index`.
   - Forward pass → compute loss with `model.loss_function()`.
   - `loss.backward()` + `optimizer.step()`.
4. Print per-batch statistics (loss, accuracy, perplexity).

After Steps 1 and 2 you will be able to run your RNN model.

### Step 2: RNN Image Captioning
Build an RNN decoder that takes a sequence of word IDs and an image embedding and outputs vocabulary logits for each timestep, following <ins>[Show and Tell](https://arxiv.org/pdf/1411.4555.pdf)</ins>.

The 2048-D ResNet embeddings are pre-computed. Pass the image embedding to your RNN as its **initial hidden state** — but first project it to `hidden_size` with a small feed-forward layer.

Use **teacher forcing**: at each step, give the decoder the previous *correct* word and have it predict the next word. In `decoder.py`, fill out `RNNDecoder`:

**RNNDecoder**
- `__init__`: define `self.image_embedding`, `self.embedding`, `self.decoder`, `self.classifier`.
- `forward`: project the image, embed captions, run the GRU, classify each timestep.
- **Return logits, not probabilities.**

**Useful PyTorch modules**:
- [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) or [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — use `batch_first=True`
- [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) for word embeddings
- [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) for feed-forward layers

#### Running your RNN Model

```bash
python assignment.py --type rnn --task both --data ../data/data.p --epochs 1 --chkpt_path ../rnn_model
```

You can also call `parse_args` from a notebook by passing a list:

```python
from assignment import parse_args, main
args = parse_args(['--type', 'rnn', '--task', 'both', '--data', '../data/data.p', '--epochs', '1'])
main(args)
```

**Your RNN model should not take more than 5 minutes to train.** Target validation perplexity ≤ 20 and per-symbol accuracy > 30 %. Our reference implementation reaches ~15 perplexity within 5 minutes without a GPU.

#### Hyperparameters for RNN
We recommend embedding / hidden sizes between 32–256 and 2–4 training epochs.

Note: pass `--chkpt_path` to save model weights needed by the notebook visualizations.


### Step 3: Attention and Transformer Blocks
Since 2017, **Transformers** have been state-of-the-art for sequence tasks. Here you will implement attention from scratch, following a simplified version of <ins>[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)</ins>.

Attention turns a sequence of embeddings into Queries, Keys, and Values. Each timestep has a query (Q), key (K), and value (V). Queries are compared to every key to produce an attention matrix, which creates new contextualised representations.

**TODO** — fill out the following components in `transformer.py`:

- **`AttentionMatrix`**: computes scaled dot-product attention weights given K and Q.
- **`AttentionHead`**: holds the K, V, Q linear projections; calls `AttentionMatrix`.
- **`MultiHeadedAttention`**: 3 heads of size `emb_sz // 3`; concatenates results; final linear layer.
- **`TransformerBlock`**: self-attention → cross-attention (over image context) → feed-forward. Apply layer norm and residual connections after each sub-layer.
- **`positional_encoding`**: sinusoidal positional encoding function.
- **`PositionalEncoding`**: embeds tokens, scales by `√embed_size`, adds positional encoding.

**Useful PyTorch modules**:
- [`nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) for layer normalization in your Transformer block.
- [`torch.bmm`](https://pytorch.org/docs/stable/generated/torch.bmm.html) for batched matrix multiplication.
- [`F.softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) for attention weights.

See Section 3.3 of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) for the feed-forward sub-layer design.
The <ins>[Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)</ins> is also an excellent resource.

**Hint for positional encoding**: alternate sin / cos across embedding dimensions using different frequencies. See the sinusoidal scheme described [here](https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer).


### Step 4: Transformer Image Captioning
This step is similar to Step 2. The key differences:

- Use a `TransformerBlock` instead of a GRU.
- Instead of passing the image as the GRU's initial hidden state, pass it as the **context sequence** for cross-attention. Reshape the image vector from `[B, hidden_size]` to `[B, 1, hidden_size]` (a sequence of length 1).

In `decoder.py`, fill out `TransformerDecoder`:

**TransformerDecoder**
- `__init__`: define `self.image_embedding`, `self.encoding` (PositionalEncoding), `self.decoder` (TransformerBlock), `self.classifier`.
- `forward`: project image → unsqueeze → embed captions → transformer block → classify.
- **Return logits, not probabilities.**

> **Note**: the embedding / hidden size must be the same for `PositionalEncoding` and `TransformerBlock`.

#### Running your Transformer

```bash
python assignment.py --type transformer --task both --data ../data/data.p --epochs 1 --chkpt_path ../transformer_model
```

Target validation perplexity ~15–18. Our reference implementation reaches this in ~4 epochs.

#### Mandatory Hyperparameters for Transformer
Similar to RNN. Training should finish in 5–10 minutes. Target validation perplexity ≤ 18 and per-symbol accuracy ~35 %.

`MultiHeadedAttention` must use 3 heads of output size `emb_sz // 3`, concatenated and projected back to `emb_sz`.


## Grading
**Code**: Graded primarily on functionality. RNN perplexity ≤ 20; Transformer perplexity ≤ 18 (test set).

**README**: Include your perplexity, accuracy, and any known bugs.


## Autograder
The autograder loads your saved model checkpoint and runs it directly — you do not need to retrain during grading. Transformer components are also tested in pseudo-isolation via the unit tests in `tests/`.To run the autograder tests locally (requires the `csci1470` conda environment):

#### TestingRun the unit tests locally:
```bash
python run_tests_local.py
```
This temporarily symlinks `solution/ → student/` and runs `pytest tests/` to validate your implementation against the reference solution.
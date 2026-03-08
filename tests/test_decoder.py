"""
Unit tests for decoder.py – RNNDecoder and TransformerDecoder (PyTorch).
"""

# ── Gradescope-compatible decorator stubs (overwritten if package is present) ─
def number(test_id):
    def decorator(fn):
        fn.test_id = test_id
        return fn
    return decorator

def weight(points):
    def decorator(fn):
        fn.weight = points
        return fn
    return decorator

try:
    from gradescope_utils.autograder_utils.decorators import number, weight  # noqa: F811
except ImportError:
    pass  # use stubs above

# ── Main imports ──────────────────────────────────────────────────────────────
from grader_utils import raise_file_paths_incorrect_error

try:
    import traceback
    import unittest
    import numpy as np
    import torch
    import torch.nn.functional as F

    from student.decoder import RNNDecoder         as StudentRNNDecoder
    from student.decoder import TransformerDecoder  as StudentTransformerDecoder

    from solution.decoder import RNNDecoder         as SolutionRNNDecoder
    from solution.decoder import TransformerDecoder  as SolutionTransformerDecoder

except Exception as e:
    print(f"Import error: {e}")
    raise_file_paths_incorrect_error()

test_num = 1


class TestDecoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cpu')
        cls.repeat = 3
        torch.manual_seed(42)

    # ── 1.1 RNNDecoder shape ─────────────────────────────────────────────────
    @number(f"{test_num}.1")
    @weight(2)
    def test_RNN_decoder_shape(self):
        """RNN Decoder Shape Test"""
        MAX_BATCH  = 100
        MAX_VOCAB  = 1000
        MAX_HIDDEN = 500
        MAX_WIN    = 40
        IMG_DIM    = 2048

        for _ in range(self.repeat):
            batch_size  = np.random.randint(10, MAX_BATCH)
            vocab_size  = np.random.randint(100, MAX_VOCAB)
            hidden_size = np.random.randint(10, MAX_HIDDEN)
            window_size = np.random.randint(10, MAX_WIN)

            encoded_images = torch.randn(batch_size, IMG_DIM)
            caption_inputs = torch.randint(0, vocab_size, (batch_size, window_size))

            args = (vocab_size, hidden_size, window_size)

            try:
                stu_dec = StudentRNNDecoder(*args)
                sol_dec = SolutionRNNDecoder(*args)

                stu_dec.eval()
                sol_dec.eval()

                with torch.no_grad():
                    stu_out = stu_dec(encoded_images, caption_inputs)
                    sol_out = sol_dec(encoded_images, caption_inputs)

                # Shape check
                self.assertEqual(stu_out.shape, sol_out.shape,
                                 f"RNN shape {stu_out.shape} != solution {sol_out.shape}")
                self.assertEqual(stu_out.shape, (batch_size, window_size, vocab_size),
                                 f"Expected {(batch_size, window_size, vocab_size)}, got {stu_out.shape}")

                # Logits check – softmax probs must NOT already sum to window_size
                probs_sum = torch.sum(F.softmax(stu_out, dim=-1), dim=-1)
                self.assertTrue(torch.all(torch.abs(probs_sum - window_size) > 0.1),
                                "RNN decoder should return logits, not probabilities")

                # Required attributes
                self.assertTrue(hasattr(stu_dec, 'image_embedding'),
                                "RNNDecoder missing self.image_embedding")
                self.assertTrue(hasattr(stu_dec, 'embedding'),
                                "RNNDecoder missing self.embedding")
                self.assertTrue(hasattr(stu_dec, 'decoder'),
                                "RNNDecoder missing self.decoder")
                self.assertTrue(hasattr(stu_dec, 'classifier'),
                                "RNNDecoder missing self.classifier")

            except Exception as e:
                self.fail(f"test_RNN_decoder_shape:\n{traceback.format_exc()}")

    # ── 1.2 TransformerDecoder shape ─────────────────────────────────────────
    @number(f'{test_num}.2')
    @weight(3)
    def test_transformer_decoder_shape(self):
        """Transformer Decoder Shape Test"""
        MAX_BATCH  = 100
        MAX_VOCAB  = 1000
        MAX_HIDDEN = 500
        MAX_WIN    = 40
        IMG_DIM    = 2048

        for _ in range(self.repeat):
            batch_size  = np.random.randint(10, MAX_BATCH)
            vocab_size  = np.random.randint(100, MAX_VOCAB)
            hidden_size = np.random.randint(10, MAX_HIDDEN)
            hidden_size = hidden_size - (hidden_size % 2)   # must be even for positional encoding
            window_size = np.random.randint(10, MAX_WIN)

            encoded_images = torch.randn(batch_size, IMG_DIM)
            caption_inputs = torch.randint(0, vocab_size, (batch_size, window_size))

            args = (vocab_size, hidden_size, window_size)

            try:
                stu_dec = StudentTransformerDecoder(*args)
                sol_dec = SolutionTransformerDecoder(*args)

                stu_dec.eval()
                sol_dec.eval()

                with torch.no_grad():
                    stu_out = stu_dec(encoded_images, caption_inputs)
                    sol_out = sol_dec(encoded_images, caption_inputs)

                # Shape check
                self.assertEqual(stu_out.shape, sol_out.shape,
                                 f"Transformer shape {stu_out.shape} != solution {sol_out.shape}")
                self.assertEqual(stu_out.shape, (batch_size, window_size, vocab_size),
                                 f"Expected {(batch_size, window_size, vocab_size)}, got {stu_out.shape}")

                # Logits check
                probs_sum = torch.sum(F.softmax(stu_out, dim=-1), dim=-1)
                self.assertTrue(torch.all(torch.abs(probs_sum - window_size) > 0.1),
                                "Transformer decoder should return logits, not probabilities")

                # Required attributes
                self.assertTrue(hasattr(stu_dec, 'image_embedding'),
                                "TransformerDecoder missing self.image_embedding")
                self.assertTrue(hasattr(stu_dec, 'encoding'),
                                "TransformerDecoder missing self.encoding")
                self.assertTrue(hasattr(stu_dec, 'decoder'),
                                "TransformerDecoder missing self.decoder")
                self.assertTrue(hasattr(stu_dec, 'classifier'),
                                "TransformerDecoder missing self.classifier")

            except Exception as e:
                self.fail(f"test_transformer_decoder_shape:\n{traceback.format_exc()}")

    # ── 1.3 Gradient flow ────────────────────────────────────────────────────
    @number(f'{test_num}.3')
    @weight(3)
    def test_decoder_gradients(self):
        """Decoder Gradient Flow Test"""
        batch_size  = 4
        vocab_size  = 100
        hidden_size = 128
        window_size = 20
        img_dim     = 2048

        encoded_images = torch.randn(batch_size, img_dim)
        caption_inputs = torch.randint(0, vocab_size, (batch_size, window_size))
        targets        = torch.randint(0, vocab_size, (batch_size, window_size))

        # RNN
        try:
            rnn_dec = StudentRNNDecoder(vocab_size, hidden_size, window_size)
            logits  = rnn_dec(encoded_images, caption_inputs)
            loss    = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()

            has_grad = any(p.grad is not None for p in rnn_dec.parameters())
            self.assertTrue(has_grad, "RNN decoder parameters have no gradients after backward()")
        except Exception as e:
            self.fail(f"RNN gradient test failed: {e}")

        # Transformer
        try:
            tf_dec  = StudentTransformerDecoder(vocab_size, hidden_size, window_size)
            logits  = tf_dec(encoded_images, caption_inputs)
            loss    = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()

            has_grad = any(p.grad is not None for p in tf_dec.parameters())
            self.assertTrue(has_grad,
                            "Transformer decoder parameters have no gradients after backward()")
        except Exception as e:
            self.fail(f"Transformer gradient test failed: {e}")


if __name__ == '__main__':
    unittest.main()

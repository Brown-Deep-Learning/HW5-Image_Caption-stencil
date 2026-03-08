"""
Unit tests for transformer.py attention components (PyTorch).
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
from grader_utils import raise_file_paths_incorrect_error, file_exists

try:
    import unittest
    import numpy as np
    import torch

    from student.transformer import AttentionHead         as Student_AttentionHead
    from solution.transformer import AttentionHead         as Solution_AttentionHead

    from student.transformer import MultiHeadedAttention  as Student_MultiHeadedAttention
    from solution.transformer import MultiHeadedAttention  as Solution_MultiHeadedAttention

    from student.transformer import AttentionMatrix        as Student_AttentionMatrix
    from solution.transformer import AttentionMatrix        as Solution_AttentionMatrix

    from student.transformer import TransformerBlock       as Student_TransformerBlock
    from solution.transformer import TransformerBlock       as Solution_TransformerBlock

    from student.transformer import PositionalEncoding     as Student_PositionalEncoding
    from solution.transformer import PositionalEncoding     as Solution_PositionalEncoding

except Exception:
    raise_file_paths_incorrect_error()

test_num = 2
is_grad  = file_exists("2470student", False)


class TestAttention(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repeat = 3
        cls.device = torch.device('cpu')

    @classmethod
    def tearDownClass(cls):
        torch.cuda.empty_cache()

    # ── 2.1 AttentionMatrix ──────────────────────────────────────────────────
    @number(f'{test_num}.1')
    @weight(2)
    def test_attention_matrix(self):
        """Attention Matrix Test"""
        MAX_BATCH = 100
        MAX_WIN   = 50
        MAX_EMBED = 300

        for _ in range(self.repeat):
            batch_size     = np.random.randint(10, MAX_BATCH)
            embed_size     = np.random.randint(10, MAX_EMBED)
            key_win_size   = np.random.randint(10, MAX_WIN)
            query_win_size = np.random.randint(10, MAX_WIN)

            K = torch.rand(batch_size, key_win_size,   embed_size) * 10
            Q = torch.rand(batch_size, query_win_size, embed_size) * 10

            try:
                stu_scores = Student_AttentionMatrix(use_mask=False)(K, Q)
                sol_scores = Solution_AttentionMatrix(use_mask=False)(K, Q)
                stu_masked = Student_AttentionMatrix(use_mask=True)(K, Q)
                sol_masked = Solution_AttentionMatrix(use_mask=True)(K, Q)

                self.assertEqual(stu_scores.shape, sol_scores.shape,
                                 "AttentionMatrix returned incorrect shape")
                self.assertEqual(stu_masked.shape, sol_masked.shape,
                                 "Masked AttentionMatrix returned incorrect shape")

                np.testing.assert_array_almost_equal(
                    stu_scores.detach().numpy(), sol_scores.detach().numpy(),
                    decimal=4, err_msg="AttentionMatrix returned incorrect values")
                np.testing.assert_array_almost_equal(
                    stu_masked.detach().numpy(), sol_masked.detach().numpy(),
                    decimal=4, err_msg="Masked AttentionMatrix returned incorrect values")

                # Verify softmax: rows must be in [0,1] and sum to 1
                arr = stu_scores.detach().numpy()
                for i in range(batch_size):
                    for j in range(query_win_size):
                        row = arr[i, j, :]
                        self.assertTrue(np.all((row >= 0) & (row <= 1)),
                                        "Attention weights outside [0, 1] – softmax missing?")
                        self.assertTrue(np.isclose(row.sum(), 1.0, atol=1e-5),
                                        "Attention rows do not sum to 1 – apply softmax")

            except Exception as e:
                self.fail(f"test_attention_matrix: {e}")

    # ── 2.2 AttentionHead shape ──────────────────────────────────────────────
    @number(f'{test_num}.2')
    @weight(3)
    def test_attention_head_shape(self):
        """Attention Head Shape Test"""
        MAX_BATCH = 100
        MAX_IN    = 200
        MAX_OUT   = 200
        MAX_WIN   = 50

        for _ in range(self.repeat):
            batch_size  = np.random.randint(10, MAX_BATCH)
            input_size  = np.random.randint(10, MAX_IN)
            output_size = np.random.randint(10, MAX_OUT)
            key_win     = np.random.randint(10, MAX_WIN)
            query_win   = np.random.randint(10, MAX_WIN)

            K = torch.rand(batch_size, key_win,   input_size) * 10
            V = torch.rand(batch_size, key_win,   input_size) * 10
            Q = torch.rand(batch_size, query_win, input_size) * 10

            try:
                stu_head = Student_AttentionHead(input_size, output_size, is_self_attention=False)
                sol_head = Solution_AttentionHead(input_size, output_size, is_self_attention=False)
                stu_out  = stu_head(K, V, Q)
                sol_out  = sol_head(K, V, Q)

                stu_self     = Student_AttentionHead(input_size, output_size, is_self_attention=True)
                sol_self     = Solution_AttentionHead(input_size, output_size, is_self_attention=True)
                stu_self_out = stu_self(K, V, Q)
                sol_self_out = sol_self(K, V, Q)

                self.assertEqual(stu_out.shape, sol_out.shape,
                                 "AttentionHead returned incorrect shape")
                self.assertEqual(stu_self_out.shape, sol_self_out.shape,
                                 "Self-AttentionHead returned incorrect shape")
                self.assertEqual(stu_out.shape, (batch_size, query_win, output_size),
                                 f"Expected {(batch_size, query_win, output_size)}, got {stu_out.shape}")

            except Exception as e:
                self.fail(f"test_attention_head_shape: {e}")

    # ── 2.3 MultiHeadedAttention shape ──────────────────────────────────────
    @number(f'{test_num}.3')
    @weight(4)
    def test_multiheaded_attention_shape(self):
        """Multi-head Attention Shape Test"""
        MAX_BATCH = 100
        MAX_EMBED = 200
        MAX_WIN   = 50

        for _ in range(self.repeat):
            batch_size = np.random.randint(10, MAX_BATCH)
            embed_size = np.random.randint(12, MAX_EMBED)
            embed_size = embed_size - (embed_size % 3)  # divisible by 3

            key_win   = np.random.randint(10, MAX_WIN)
            query_win = np.random.randint(10, MAX_WIN)

            K = torch.rand(batch_size, key_win,   embed_size) * 10
            V = torch.rand(batch_size, key_win,   embed_size) * 10
            Q = torch.rand(batch_size, query_win, embed_size) * 10

            try:
                stu_mha  = Student_MultiHeadedAttention(embed_size, use_mask=False)
                sol_mha  = Solution_MultiHeadedAttention(embed_size, use_mask=False)
                stu_out  = stu_mha(K, V, Q)
                sol_out  = sol_mha(K, V, Q)

                stu_smha = Student_MultiHeadedAttention(embed_size, use_mask=True)
                sol_smha = Solution_MultiHeadedAttention(embed_size, use_mask=True)
                stu_sout = stu_smha(K, V, Q)
                sol_sout = sol_smha(K, V, Q)

                self.assertEqual(stu_out.shape, sol_out.shape,
                                 "MultiHeadedAttention returned incorrect shape")
                self.assertEqual(stu_sout.shape, sol_sout.shape,
                                 "Masked MultiHeadedAttention returned incorrect shape")
                self.assertEqual(stu_out.shape, (batch_size, query_win, embed_size),
                                 f"Expected {(batch_size, query_win, embed_size)}, got {stu_out.shape}")

                # 3 heads × 3 weight matrices = 9 params with head dimensions.
                # Accept both (embed, embed//3) [nn.Parameter style] and
                # (embed//3, embed) [nn.Linear weight storage style].
                correct_head_shapes = {
                    (embed_size, embed_size // 3),
                    (embed_size // 3, embed_size),
                }
                correct_final = (embed_size, embed_size)
                head_count = 0
                has_final  = False
                for _, param in stu_mha.named_parameters():
                    if tuple(param.shape) in correct_head_shapes:
                        head_count += 1
                    elif tuple(param.shape) == correct_final:
                        has_final = True

                self.assertEqual(head_count, 9,
                                 f"Expected 9 head weight matrices, found {head_count}")
                self.assertTrue(has_final,
                                "MultiHeadedAttention missing final linear layer")

            except Exception as e:
                self.fail(f"test_multiheaded_attention_shape: {e}")

    # ── 2.4 TransformerBlock shape ───────────────────────────────────────────
    @number(f'{test_num}.4')
    @weight(3)
    def test_transformerblock_shape(self):
        """Transformer Block Shape Test"""
        MAX_BATCH   = 100
        MAX_IN_LEN  = 50
        MAX_CTX_LEN = 300
        MAX_EMBED   = 300

        for _ in range(self.repeat):
            batch_size  = np.random.randint(1,  MAX_BATCH)
            input_len   = np.random.randint(5,  MAX_IN_LEN)
            context_len = np.random.randint(5,  MAX_CTX_LEN)
            embed_size  = np.random.randint(12, MAX_EMBED)
            embed_size  = embed_size - (embed_size % 3)

            inputs  = torch.rand(batch_size, input_len,   embed_size) * 10
            context = torch.rand(batch_size, context_len, embed_size) * 10

            try:
                stu_block = Student_TransformerBlock(emb_sz=embed_size, multiheaded=False)
                sol_block = Solution_TransformerBlock(emb_sz=embed_size, multiheaded=False)
                stu_out   = stu_block(inputs, context)
                sol_out   = sol_block(inputs, context)

                self.assertEqual(stu_out.shape, sol_out.shape,
                                 "TransformerBlock returned incorrect shape")
                self.assertEqual(stu_out.shape, (batch_size, input_len, embed_size),
                                 f"Expected {(batch_size, input_len, embed_size)}, got {stu_out.shape}")

            except Exception as e:
                self.fail(f"test_transformerblock_shape: {e}")

    # ── 2.5 PositionalEncoding ───────────────────────────────────────────────
    @number(f'{test_num}.5')
    @weight(3)
    def test_positional_encoding(self):
        """Positional Encoding Test"""
        MAX_VOCAB = 600
        MAX_EMBED = 300
        MAX_WIN   = 50
        MAX_BATCH = 100

        for _ in range(self.repeat):
            vocab_size = np.random.randint(10, MAX_VOCAB)
            embed_size = np.random.randint(10, MAX_EMBED)
            embed_size = embed_size - (embed_size % 2)   # must be even
            win_size   = np.random.randint(10, MAX_WIN)
            batch_size = np.random.randint(10, MAX_BATCH)

            x = torch.randint(0, vocab_size, (batch_size, win_size))

            try:
                stu_enc = Student_PositionalEncoding(vocab_size, embed_size, win_size)
                sol_enc = Solution_PositionalEncoding(vocab_size, embed_size, win_size)

                # Share embedding weights for a fair values comparison
                stu_enc.embedding.weight.data.copy_(sol_enc.embedding.weight.data)

                stu_out = stu_enc(x)
                sol_out = sol_enc(x)

                self.assertEqual(stu_out.shape, sol_out.shape,
                                 f"PositionalEncoding shape mismatch: "
                                 f"student {stu_out.shape} vs solution {sol_out.shape}")

                np.testing.assert_array_almost_equal(
                    stu_out.detach().numpy(), sol_out.detach().numpy(),
                    decimal=4, err_msg="PositionalEncoding returned incorrect values")

                # Encoding must differ across positions
                pos_diff = torch.norm(stu_out[0, 0, :] - stu_out[0, 1, :])
                self.assertGreater(pos_diff.item(), 1e-3,
                                   "Positional encoding does not vary with position")

            except Exception as e:
                self.fail(f"test_positional_encoding: {e}")


if __name__ == '__main__':
    unittest.main()

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from seqio import seqio
from t5x import adafactor
from t5x import models
from t5x import test_utils
from models import network


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


def get_test_model(emb_dim,
                   head_dim,
                   num_heads,
                   mlp_dim,
                   dtype='float32',
                   vocab_size=32128,
                   num_encoder_layers=2,
                   num_decoder_layers=2):
  config = network.T5Config(
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      vocab_size=vocab_size,
      dropout_rate=0,
      emb_dim=emb_dim,
      num_heads=num_heads,
      head_dim=head_dim,
      mlp_dim=mlp_dim,
      dtype=dtype,
      mlp_activations=('gelu', 'linear'))
  module = network.FiDT5(config=config)
  vocab = seqio.test_utils.sentencepiece_vocab()
  optimizer_def = adafactor.Adafactor()
  return models.EncoderDecoderModel(
      module, vocab, vocab, optimizer_def=optimizer_def)


class NetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    batch_size, n_passages, max_decode_len, input_len = 2, 3, 4, 5
    self.input_shapes = {
        'encoder_input_tokens': (batch_size, n_passages, input_len),
        'decoder_input_tokens': (batch_size, max_decode_len)
    }
    np.random.seed(42)
    self.batch = {
        'encoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, n_passages, input_len)),
        'decoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        'decoder_target_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len))
    }

  def test_fidt5(self):
    np.random.seed(0)
    batch_size, n_passages, max_decode_len, input_len = 2, 3, 4, 5
    batch = {
        'encoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, n_passages, input_len)),
        'decoder_input_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        'decoder_target_tokens':
            np.random.randint(3, 10, size=(batch_size, max_decode_len))
    }
    model = get_test_model(
        emb_dim=13,
        head_dim=64,
        num_heads=8,
        mlp_dim=2048,
        vocab_size=10,
        num_encoder_layers=3)
    params = model.get_initial_variables(
        jax.random.PRNGKey(42), self.input_shapes)['params']
    loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
    self.assertAlmostEqual(loss, 21.607828, delta=0.05)

    predicted, scores = model.predict_batch_with_aux(params, batch)
    print(predicted)
    np.testing.assert_array_equal(predicted, [[1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(
        scores['scores'], [-1.573776, -1.54673], rtol=1e-2
    )


if __name__ == '__main__':
  absltest.main()
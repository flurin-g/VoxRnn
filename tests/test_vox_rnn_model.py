from unittest import TestCase
from vox_rnn_model import build_embedding_train_model


class TestVoxRnnModel(TestCase):
    def test_build_embedding_train_model(self):
        model = build_embedding_train_model(40, 128)

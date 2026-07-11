"""Dataset-free, hardware-free tests for replay_client.py's protocol/aggregation
logic. Uses a FakeDevice instead of real serial or a loaded TFLite model, so
these run in CI with no dataset and no ESP32 attached.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_client import Device, score_files  # noqa: E402


class FakeDevice(Device):
    """Returns a fixed score per call, or one score per call in sequence."""

    def __init__(self, scores):
        self._scores = iter(scores)

    def query(self, quantized_vector):
        return next(self._scores)


def test_score_files_averages_per_vector_scores_into_a_file_score():
    # One file with 2 vectors (scores 1.0, 3.0 -> file score 2.0), labeled normal;
    # one file with 1 vector (score 10.0), labeled anomalous.
    device = FakeDevice([1.0, 3.0, 10.0])
    files = [
        (0, [np.zeros(4, dtype=np.int8), np.zeros(4, dtype=np.int8)]),
        (1, [np.zeros(4, dtype=np.int8)]),
    ]
    auc, scores, labels = score_files(device, files)

    assert scores == [2.0, 10.0]
    assert labels == [0, 1]
    assert auc == 1.0  # the anomalous file scored strictly higher


def test_score_files_skips_empty_vector_lists():
    device = FakeDevice([5.0])
    files = [(0, []), (1, [np.zeros(4, dtype=np.int8)])]
    auc, scores, labels = score_files(device, files)

    assert scores == [5.0]
    assert labels == [1]
    # only one class present -> AUC is undefined
    assert auc != auc  # NaN check (nan != nan)

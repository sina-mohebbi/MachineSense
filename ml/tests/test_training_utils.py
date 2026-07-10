"""Fast tests for dataset limiting and feature normalization."""

import numpy as np

import train
from evaluate_tflite import Int8Autoencoder


def test_train_limit_is_balanced_across_machine_ids():
    files = [
        (machine_id, f"{machine_id}-{index}.wav")
        for machine_id in ("id_00", "id_02", "id_04", "id_06")
        for index in range(10)
    ]

    selected = train.limit_train_files(files, 12)
    counts = {machine_id: 0 for machine_id in ("id_00", "id_02", "id_04", "id_06")}
    for machine_id, _ in selected:
        counts[machine_id] += 1

    assert len(selected) == 12
    assert set(counts.values()) == {3}


def test_normalize_uses_supplied_statistics():
    vectors = np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    mean = np.array([2.0, 6.0], dtype=np.float32)
    std = np.array([1.0, 2.0], dtype=np.float32)

    normalized = train.normalize(vectors, mean, std)

    np.testing.assert_allclose(normalized, [[-1.0, -1.0], [1.0, 1.0]])
    assert normalized.dtype == np.float32


def test_int8_quantize_and_dequantize():
    details = {"quantization": (0.1, -3)}
    values = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    quantized = Int8Autoencoder._quantize(values, details)
    restored = Int8Autoencoder._dequantize(quantized, details)

    assert quantized.dtype == np.int8
    np.testing.assert_allclose(restored, values, atol=0.05)

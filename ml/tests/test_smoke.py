"""Dataset-free smoke tests so CI stays green without downloading MIMII.

These exercise the model + export path on random data: build the autoencoder, train
one step, int8-quantize, and emit the C header. They catch shape/topology/export
regressions in seconds.
"""
import numpy as np
import tensorflow as tf

import config
from model import build_autoencoder
import export_tflite


def test_autoencoder_io_shape():
    model = build_autoencoder()
    x = np.random.rand(4, config.FEATURE_DIM).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (4, config.FEATURE_DIM)


def test_train_one_step():
    model = build_autoencoder()
    x = np.random.rand(64, config.FEATURE_DIM).astype(np.float32)
    hist = model.fit(x, x, epochs=1, batch_size=32, verbose=0)
    assert np.isfinite(hist.history["loss"][-1])


def test_int8_export_and_header(tmp_path):
    model = build_autoencoder()
    rep = np.random.rand(64, config.FEATURE_DIM).astype(np.float32)
    tflite_bytes = export_tflite.to_int8_tflite(model, rep)
    assert len(tflite_bytes) > 0

    # input/output tensors must be int8 for TFLite-Micro
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.int8
    assert interp.get_output_details()[0]["dtype"] == np.int8

    out = tmp_path / "model_data.cc"
    export_tflite.tflite_to_c_header(tflite_bytes, "g_model_data", out)
    text = out.read_text()
    assert "g_model_data[]" in text
    assert "g_model_data_len" in text
    assert out.with_suffix(".h").exists()

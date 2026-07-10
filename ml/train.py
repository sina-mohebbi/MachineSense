"""Phase 0 entry point: train the autoencoder on normal MIMII clips and report AUC.

Protocol
--------
* Pool `normal` clips across machine ids, hold out TEST_NORMAL_FRACTION for testing.
* Train the autoencoder to reconstruct log-mel vectors of the training normals.
* Per-file anomaly score = mean reconstruction MSE over the file's vectors.
* AUC over the test set (held-out normal = 0, all abnormal = 1), overall and per id.

Artifacts written to ml/artifacts/:
    model.keras, rep_vectors.npy, metrics.json
Run `python export_tflite.py` afterwards to produce the int8 model + C header.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import config
import data
from model import build_autoencoder


def normalize(vectors, mean, std):
    """Apply feature-wise statistics learned from normal training vectors."""
    return ((vectors - mean) / std).astype(np.float32)


def file_score(model, path, mean, std) -> float:
    v = data.cached_file_to_vectors(path)
    if len(v) == 0:
        return 0.0
    v = normalize(v, mean, std)
    recon = model.predict(v, verbose=0)
    return float(np.mean((v - recon) ** 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MachineSense autoencoder")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument(
        "--max-train-files", type=int, default=None,
        help="Limit normal training clips for a quick experiment",
    )
    parser.add_argument(
        "--max-test-per-class", type=int, default=None,
        help="Limit normal and abnormal test clips per machine ID",
    )
    return parser.parse_args()


def limit_test_files(test_files, per_class):
    if per_class is None:
        return test_files
    counts = {}
    selected = []
    for item in test_files:
        key = (item[0], item[2])
        if counts.get(key, 0) < per_class:
            selected.append(item)
            counts[key] = counts.get(key, 0) + 1
    return selected


def limit_train_files(train_files, maximum):
    """Choose a deterministic round-robin sample across machine IDs."""
    if maximum is None or maximum >= len(train_files):
        return train_files
    by_id = {}
    for item in train_files:
        by_id.setdefault(item[0], []).append(item)
    selected = []
    offset = 0
    machine_ids = sorted(by_id)
    while len(selected) < maximum:
        added = False
        for machine_id in machine_ids:
            group = by_id[machine_id]
            if offset < len(group) and len(selected) < maximum:
                selected.append(group[offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def main() -> None:
    args = parse_args()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_files, test_files = data.build_dataset()
    train_files = limit_train_files(train_files, args.max_train_files)
    test_files = limit_test_files(test_files, args.max_test_per_class)
    print(f"[data] machine={config.MACHINE}  train_clips={len(train_files)}  "
          f"test_clips={len(test_files)}")

    x_train = data.stack_train_vectors(train_files)
    print(f"[data] training vectors: {x_train.shape}")
    feature_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x_train = normalize(x_train, feature_mean, feature_std)

    model = build_autoencoder()
    model.summary()
    model.fit(
        x_train, x_train,
        epochs=args.epochs, batch_size=config.BATCH,
        validation_split=config.VAL_SPLIT, shuffle=True, verbose=2,
    )

    # --- evaluate AUC --------------------------------------------------------
    scores = np.array([
        file_score(model, f, feature_mean, feature_std) for _, f, _ in test_files
    ])
    labels = np.array([lbl for _, _, lbl in test_files])
    overall_auc = roc_auc_score(labels, scores)

    per_id = {}
    ids = np.array([mid for mid, _, _ in test_files])
    for mid in sorted(set(ids)):
        m = ids == mid
        if len(set(labels[m])) == 2:
            per_id[mid] = round(float(roc_auc_score(labels[m], scores[m])), 4)

    print(f"\n[result] overall AUC = {overall_auc:.4f}")
    for mid, auc in per_id.items():
        print(f"[result]   {mid}: AUC = {auc:.4f}")

    # --- save artifacts ------------------------------------------------------
    model.save(config.MODEL_KERAS)
    np.savez(config.NORMALIZATION, mean=feature_mean, std=feature_std)
    rep_idx = np.random.default_rng(0).choice(
        len(x_train), size=min(1000, len(x_train)), replace=False)
    np.save(config.REP_VECTORS, x_train[rep_idx])
    metrics = {
        "machine": config.MACHINE,
        "overall_auc": round(float(overall_auc), 4),
        "per_id_auc": per_id,
        "train_vectors": int(x_train.shape[0]),
        "train_clips": len(train_files),
        "test_clips": len(test_files),
        "epochs": args.epochs,
        "feature_dim": config.FEATURE_DIM,
    }
    (config.ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[saved] {config.MODEL_KERAS}")
    print(f"[saved] {config.ARTIFACTS / 'metrics.json'}")
    print("[next] python export_tflite.py")


if __name__ == "__main__":
    main()

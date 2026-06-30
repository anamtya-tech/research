# 07 — YAMNet Training (export base → fine-tune → TFLite)

**Component:** `yamnet/` repo (`integration/export_yamnet_core.py`, `training/*.py`).
**Input:** a curator dataset (`labels.csv` + `audio/*.wav`, see [`06`](06_yamnet_dataset_curation.md)).
**Output:** a versioned `.tflite` + `custom_class_map.csv` in `model_store/releases/<version>/`.
**Goal:** adapt the 521-class AudioSet YAMNet into a small single-label classifier over the
Chatak target classes, exported for the on-device TFLite runtime.

> This is also where [`plan.md`](../plan.md)'s ML experiments land (baseline → embeddings →
> UMAP → linear probe → MLP head): the same 96×64 log-mel front-end and `embeddings` layer feed
> all of them.

## Fixed signal front-end (must match ODAS exactly)

`data_loader.py:40` constants — keep in lockstep with the C wrapper:
SR 16000 · STFT window 400 (25 ms) · hop 160 (10 ms) · FFT 512 · **257** spectrum bins ·
64 mel · 125–7500 Hz · `log_offset 0.001` · patch **96** frames · patch hop **48**.
The 257-bin magnitude spectrum is the ODAS handoff; mel-projection to 64 bins happens in both
training and firmware.

## 1. Base model export — `integration/export_yamnet_core.py`

Pulls upstream YAMNet from TF-Hub (`hub.load('https://tfhub.dev/google/yamnet/1')`, `:194`) and
rebuilds it as a **mel-input** Keras model (TF-Hub YAMNet takes a waveform; this variant takes a
precomputed 96×64 log-mel patch so ODAS can feed spectra directly):

- `yamnet_core_model` (`:26`): input `(B,96,64,1)` → depthwise-separable CNN (from
  `_YAMNET_LAYER_DEFS`) → `GlobalAveragePooling2D` named **`embeddings`** → `Dense(521)` logits
  → `sigmoid` `predictions`.
- `transfer_all_weights` (`:49`) copies *all* variables (conv + batch-norm running stats) by
  index, asserting count/shape.
- Exports `yamnet_core/` SavedModel (`:109`), `yamnet_core.tflite` (float32) and
  `yamnet_core_quantized.tflite` (dynamic-range) (`:126`). Sanity gate: sigmoid output sum < 10.
- I/O: in `(1,96,64,1)` float32; out `(1,521)` sigmoid.

```bash
python integration/export_yamnet_core.py            # → yamnet_core/, yamnet_core.tflite
cp -r integration/yamnet_core model_store/base/yamnet_core_savedmodel
```

## 2. Data loading — `training/data_loader.py`

- `waveform_to_mel_patches` (`:62`): `tf.signal.stft(400,160,512)` → magnitude (n,257) →
  mel filterbank `(257,64)` → `log(mel+0.001)` → `tf.signal.frame(96, 48)` → `(p,96,64,1)`.
  Returns empty if < 96 frames.
- `build_label_map` (`:113`): unique labels **sorted alphabetically** → `class→index`. (This
  ordering defines the class-map; keep it stable.)
- `load_dataset` (`:266`): honors `val`/`test` folds in `labels.csv` if present, else
  auto-splits **75/15/10** (seeded). One label per file (single-label), one-hot, batched.
- **Augmentation** (train only, log-mel domain): Gaussian noise σ=0.02, random gain
  [0.85,1.15], clip [−6.5,4.0].
- `compute_class_weights` (`:203`): sklearn balanced weights with optional floor.

## 3. Two-phase fine-tune — `training/train_yamnet.py`

- **Head** (`build_finetuned_model:110`): backbone → `embeddings` → `Dense(256,relu)` →
  `Dropout(0.3)` → `Dense(N, softmax)` `custom_predictions`. **Single-label softmax** (differs
  from the base sigmoid head).
- **Weight source priority:** local SavedModel → TF-Hub → random; optional warm-start from a
  prior `.keras` (auto-skips the head when class count changes).
- **Phase 1** (`:434`): freeze backbone, train the 3 head layers. `Adam(1e-3)`, ~20 epochs.
- **Phase 2** (`:459`): `unfreeze_top_layers(4)` — unfreeze top N layers **excluding batch-norm**
  (stable running stats). `Adam(1e-5)`, ~30 epochs. Skipped if `phase2-epochs=0`.
- **Loss:** `categorical_crossentropy` + class weights; `--focal-gamma>0` switches to
  alpha-weighted multi-class focal loss. Output is **softmax**.
- **Callbacks:** EarlyStopping(patience 7, restore best), ReduceLROnPlateau(0.5/4), TensorBoard,
  per-class recall metrics.
- **Outputs** → `model_store/checkpoints/chatak_yamnet_<ts>/`: `model.keras`, `class_map.csv`
  (`index,class_name`), `training_log.json` (classes, hyperparams, per-class val recall, test
  acc/loss).

```bash
python training/train_yamnet.py \
    --dataset ~/simulator/outputs/yamnet_datasets/yamnet_train_001 \
    --savedmodel model_store/base/yamnet_core_savedmodel \
    --phase1-epochs 20 --phase2-epochs 30
```

## 4. Export + registry — `training/export_finetuned.py`

- `export_tflite` (`:75`): `from_keras_model` → float32 always + INT8 (dynamic-range) when
  `--quantize`. Verifies softmax sum ≈ 1.0.
- `write_class_map` (`:128`): **3-column** `index,mid,display_name` with dummy mid
  `/m/custom_<i>` — required because the ODAS C++ `LoadClassNames()` always reads `fields[2]`.
- Outputs → `model_store/releases/<version>/`: `chatak_yamnet_<version>.tflite`,
  `..._int8.tflite`, `custom_class_map.csv`, `export_info.json` (provenance + suggested cfg keys).

```bash
python training/export_finetuned.py \
    --checkpoint model_store/checkpoints/chatak_yamnet_<ts> --version v1.0.0
```

## `registry.json` (the only tracked model state)

`model_store/registry.json` — binaries are gitignored; this index is the source of truth:
```json
{
  "schema_version": "1",
  "models": [{
    "run_name": "...", "nickname": "...", "timestamp": "...",
    "classes": ["Bear", ...], "num_classes": 17,
    "val_accuracy": 0.6355,            // NOTE: actually stores TEST accuracy
    "model_path": ".../model.keras",
    "tflite_path": ".../releases/v1.0.0/chatak_yamnet_v1.0.0.tflite",
    "tflite_int8_path": ".../v1.0.0_int8.tflite",
    "dataset": "...", "deployed": true, "version": "v1.0.0", "exported_at": "..."
  }],
  "active_model": "fixed-ambient-ele-fg-dr"
}
```
Current active model: `fixed-ambient-ele-fg-dr` — 4-class (Elephant/Frog/background/drone_bebop),
val_acc 0.858.

## Python reference classifier (used by the analyzer/curator)

`simulator/yamnet_helper/yamnet_spectrum_classifier.py` mirrors the ODAS C++ classifier in
Python: loads a TFLite interpreter, infers `NUM_CLASSES` from the CSV, converts each 257-bin
spectrum to mel and `classify_patch([96,257])` → `(class_id, name, confidence)`; `add_frame`
buffers for real-time (first at 96 frames, then every 48). The analyzer uses it to re-classify
`.bin` sidecars (base model) or the registry's active fine-tuned model.

## Known issues / gotchas (worth fixing)

- **`registry.json.val_accuracy` actually holds test accuracy** (`train_yamnet.py:533/584`) —
  misleading field name.
- **Versioning overwrites:** multiple registry entries point at the same `v1.0.0` tflite path;
  re-exporting another run to `v1.0.0` silently replaces the deployed binary. Use unique
  versions.
- **Mel-filterbank mismatch risk:** the Python helper hand-rolls an HTK triangular filterbank,
  while training uses `tf.signal.linear_to_mel_weight_matrix` — not guaranteed bit-identical.
- `SETUP.md` (TFLite C build for the standalone test) is an informal cheat-sheet with typos.

## Next

[`08_deployment_and_loop.md`](08_deployment_and_loop.md) — push a release into ODAS and close
the retraining loop.

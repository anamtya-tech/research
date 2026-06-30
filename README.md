# Chatak — Research

Research, experiments, and documentation for the **Chatak** wildlife acoustic-monitoring system:
a ReSpeaker 4-mic array on a Raspberry Pi running **ODAS** (localise + track) with an embedded
**YAMNet** classifier (what sound, from which direction). This repo is the **analysis + experiment
+ docs** layer; the runnable components live in their own repos.

## 📄 Start here
- **[`docs/REPORT.md`](docs/REPORT.md)** — full plain-language report: what we built, every
  experiment, findings, figures, conclusions.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — what's done and what remains, prioritized.
- **[`docs/README.md`](docs/README.md)** — index of all step/experiment docs.

## Headline outcomes
- **YAMNet is the right brain** — its embeddings cleanly separate our classes; only the head needs retraining.
- **False positives** come from mic-array geometry + ambient noise (not the tracker). **Hard
  negatives from *real* ambient cut them ~81%** on a realistic testbed.
- **Measure on real ambient, not silence** — our biggest methodological catch (silent renders were 2–3× too optimistic).
- **Accuracy ceiling (~0.50) broken to ~0.62** via backbone fine-tuning **+ SpecAugment** (the
  binding constraint was cross-render generalization, not data volume or model capacity alone).
- The fine-tuned **7-class model is shipped** as the default in the `yamnet` (registry) and
  `chatak-odas` (device binary) repos.

## Repo layout
```
docs/            all documentation — REPORT, ROADMAP, step docs (00–08), experiment docs, figures
experiments/
  scripts/       all experiment + pipeline-driver code (data prep, ODAS runs, training, export)
  outputs/       metrics (json/csv) + figures   (large binaries are gitignored)
  corpus/        labeled training-corpus record (meta.csv)
  odas/          ODAS Docker build + runtime configs
data_prep/       annotated clip inventory (scripts, inventory.csv, figures)
plan.md          ML warm-up plan        experiments.pdf  the production experiment program
```
**Gitignored** (large / separate / regenerable): `backup/`, `.venv/`, model+audio binaries, and the
**`chatak-odas` / `yamnet` / `simulator`** sub-repos (each has its own remote).

## Related repos
| Repo | Role |
|---|---|
| [anamtya-tech/chatak-odas](https://github.com/anamtya-tech/chatak-odas) | C firmware — ODAS + embedded YAMNet (what runs on the Pi; carries the deployed model) |
| [anamtya-tech/yamnet](https://github.com/anamtya-tech/yamnet) | YAMNet training / export / model registry |
| [anamtya-tech/simulator](https://github.com/anamtya-tech/simulator) | Python pipeline — scene render → ODAS → analyze → curate |

## Reproducing
Needs a Python venv with `tensorflow, tf-keras, soundfile, librosa, scikit-learn, umap-learn,
pyroomacoustics, matplotlib`, and Docker (arm64) for the ODAS runs. Each doc lists exact commands;
the ODAS build is in [`docs/ODAS_BRINGUP.md`](docs/ODAS_BRINGUP.md). Headline pipeline:
```
data_prep/scripts/build_dataset.py        # clip inventory
experiments/scripts/extract_yamnet.py     # YAMNet embeddings
experiments/scripts/corpus.py --build --train          # collective training corpus + head
experiments/scripts/finetune_backbone.py --sweep 6 --seeds 0,1 --augment   # backbone FT + SpecAugment
experiments/scripts/export_finetuned_model.py          # → TFLite + registry
```

## Datasets (Google Cloud Storage)
The generated datasets are kept in **`gs://chatak-data/research-2026/`** (the large binaries are
gitignored here). See `gs://chatak-data/research-2026/README.txt` for the full manifest.

| Path | What |
|---|---|
| `corpus/embeddings.npy` + `meta.csv` | the **labeled training corpus** (7,251 post-ODAS samples; 4 ambient envs, 3 SNRs) |
| `corpus/postodas_bins.tar.gz` | raw 96×257 `.bin` spectra — needed to redo **backbone** fine-tuning |
| `models/yamnet_finetuned.keras`, `chatak_yamnet_v2.0.0-specaug.tflite`, `custom_class_map.csv` | the shipped 7-class model + deployable TFLite |
| `gt_datasets.tar.gz` | clean GT clips + `labels.csv` |
| `data_prep_clips.tar.gz`, `data_prep/inventory.csv` | the 450-clip annotated inventory |
| `scenes.tar.gz` | scene JSONs (render recipe; raw renders are regenerable, not stored) |

```bash
# train collectively from the corpus (after pulling embeddings.npy + meta.csv):
gsutil cp -r gs://chatak-data/research-2026/corpus ./experiments/corpus
python experiments/scripts/corpus.py --train
```

## Data needs (for new client classes)
Per target class: **~100–200 clean example clips** (we synthesize ~400–600 training samples via the
simulator). Background: **a few hours → 24 h of on-site ambient on the mic array** covers the noise
side abundantly. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for detail.

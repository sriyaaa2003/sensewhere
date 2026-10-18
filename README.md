# sensewhere

**Radio-sensing classifier that keeps working at a new site.** A reproducible pipeline on real public mmWave-radar and Wi-Fi sensing data: cross-site evaluation, source-free adaptation, track-level prediction and adding new classes after deployment.

## The problem

A radio network can sense moving people and objects (integrated sensing and communication). A model trained at one site is deployed at the next, usually without labels there, and new classes keep arriving. This project builds and evaluates the full pipeline for that setting: preprocessing, training, adaptation, class extension and evaluation on unseen sites.

## Data

* [DopplerWild](https://www.kaggle.com/datasets/dopplerwild/dopplerwild) (CC BY 4.0): TI IWR6843 mmWave radar micro-Doppler spectrograms of pedestrians at four sites. 410 labeled tracks from 302 subjects, three motion classes (walking, dual-hand load, on wheels) plus a held-out single-hand-load class, and 2,237 unlabeled tracks.
* [Widar 3.0](https://ieee-dataport.org/open-access/widar30-dataset-cross-domain-gesture-recognition-wi-fi) Wi-Fi velocity profiles (session 20181130: 9 users, 9 gestures, 5 torso locations) for replication.

## What it does

* **Preprocessing.** Spectrograms are cut into 1-second windows (90 bins/s, 256 Doppler bins) aligned to tracker tracks, with quality checks that abort the build if windows fall outside their tracks. Unlabeled recordings are used for pretraining and adaptation.
* **Cross-site evaluation.** Leave-one-location-out splits, balanced accuracy, subject-clustered bootstrap intervals and pre-registered confirmatory runs on fresh seeds.
* **Models.** Small CNN backbone (plus ResNet-18 and MobileNetV2 variants) with flat, hierarchical-tree and mixture-of-experts heads.
* **Domain generalisation and adaptation.** CORAL, DANN, noise-floor transplant, AdaBN, Tent, SHOT, a custom track-consistency self-training, and source-free boundary recalibration.
* **Track-level prediction.** Window predictions are pooled over each tracked person; attention and GRU track models and track-contrastive pretraining are included.
* **Class extension.** Head expansion with replay, fine-tuning, and a prototype (nearest-class-mean) extension that adds a class from a few labeled windows without retraining.
* **Uncertainty.** Conformal prediction sets with coverage measured at the unseen site.

## Results

* Track pooling raises balanced accuracy at an unseen site by 0.044 [0.029, 0.058] on fresh seeds 3 to 7 (0.785 [0.742, 0.815] against 0.741 [0.707, 0.767]).
* A gradient-boosting model recognises the four sites from the signal with one-vs-rest AUC 0.720; viewing geometry (approach angle, speed, range) alone identifies site B with AUC 0.944.
* The prototype extension reaches joint balanced accuracy 0.541 [0.525, 0.558] with 5 labeled windows and nothing trained.
* On Widar 3.0, CORAL beats plain training by +0.014 [+0.006, +0.023].

Full tables and figures: [docs/RESULTS.md](docs/RESULTS.md). Design: [docs/BLUEPRINT.md](docs/BLUEPRINT.md). Literature review: [docs/LITERATURE.md](docs/LITERATURE.md).

## Tech stack

Python 3.11, PyTorch, torchvision, scikit-learn, NumPy, pandas, SciPy, Matplotlib, Pydantic, PyYAML, pytest (132 tests). Experiments run as Kaggle GPU script kernels; analysis scripts regenerate every table and figure from saved results.

## Reproduce

```
pip install -e .[dev]
pytest
python scripts/analyze.py          # tables, figures and docs/RESULTS.md from results/
python scripts/make_readme.py      # regenerates this file from README.template.md
```

The experiments live in `kaggle/*`: `build` and `build2` create the cached windows, `stage*` run the experiments, and `scripts/push_src.sh` publishes the package as a private Kaggle dataset.

## License and data

Code: MIT. DopplerWild is CC BY 4.0; Widar 3.0 is subject to its own terms. Neither dataset is redistributed here.

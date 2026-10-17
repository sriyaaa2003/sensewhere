# Confirmatory test (written before the confirmatory run)

The v2 arm tables compare many arms. The "whole pipeline" arm (noise-floor transplant, BatchNorm-only SHOT, track pooling) was picked **after** inspecting those tables, so its advantage over plain training is an optimistic estimate (best of many combinations). Contrasts 1 to 3 in `docs/RESULTS.md` section v2-4 were also written after seeing the tables.

To get an unbiased estimate, the following hypotheses are fixed here, before the confirmatory run, and tested on **fresh seeds 3 to 7** (new splits, new initialisations; the per-site learning rates reuse the nested selection already made in stage A, so nothing is re-tuned):

* **H1.** Track pooling raises balanced accuracy of plain training at an unseen site (arm `erm`, readout vote minus window).
* **H2.** The pipeline (`erm_floor` with BatchNorm-only SHOT, track-vote readout) beats plain training with track pooling (`erm`, no adaptation, vote).
* **H3.** The pipeline beats plain training with the window readout (`erm`, none, window).
* **H4 (control).** Augmentation with track pooling (`erm_aug`, vote) is not worse than plain training with track pooling; it is the strongest simple arm in stage A.

Decision rule: a hypothesis is "confirmed" only if the 95% subject-clustered interval of the difference on the fresh seeds excludes zero in the stated direction. Four tests, so Bonferroni-adjusted intervals (99.38%) are also reported. Whatever the outcome is goes into the README.

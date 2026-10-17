# Code and protocol audit (written after the v1 results)

v1 produced results that looked unconvincing (nothing beat plain training; adaptation partly hurt). Before changing the method I read every module and checked the saved results for evidence. Findings, with the status of each.

## Confirmed defects (each has a regression test)

| # | Where | Problem | Effect on v1 | Fix |
|---|---|---|---|---|
| 1 | `train.fit` | Augmentation was switched on for every method except plain ERM | CORAL, DANN and Group-DRO were really "ERM + augmentation + X" and were compared with un-augmented ERM | explicit `augment` flag; v1 behaviour kept behind `legacy_aug` so v1 can be rerun |
| 2 | `train.fit` with frozen backbone | `model.train()` put BatchNorm back into training mode, so running statistics kept changing while the backbone was "frozen" | old-class features drifted during class extension; the measured forgetting was partly an artefact | `freeze_bn` option (used for extension) |
| 3 | `train.fit` | if every validation score was NaN, `best_state` stayed `None` and loading it crashed | latent crash | keep last weights |
| 4 | `preprocess.process_bvp` | a Widar file with zero frames crashed the stage | stage 4 failed on first run | flagged as `empty` in QC |
| 5 | build script | time column `t_start` is measured from the recording, the track file needs `t_start_rel` | 30% of windows were outside their track on the first build | switched column; build now aborts if more than 1% of windows fall outside a track |
| 6 | build script | windows shorter than 1 s (track ends) | silently mis-shaped | dropped as `short_window` |

## Weaknesses shown by the v1 numbers

* **Model selection was a poor guide.** Source-validation balanced accuracy averaged 0.80 against 0.72 at the unseen site; across 120 runs the two correlate at 0.30. Choosing epochs and settings on it is close to noise, and DomainBed makes the same point about domain generalisation. v2 selects by nested leave-one-source-site-out, so validation itself simulates a site change and the target never influences any choice.
* **SHOT-style adaptation was run aggressively.** Full-backbone updates at lr 1e-3 with no pseudo-labels drove prediction entropy from 0.40 to 0.03 within ten epochs while accuracy fell. That is runaway self-training, so v1's "SHOT hurts" is as much a tuning result as a method result. v2 uses the full SHOT recipe (pseudo-labels, lr 1e-4, BatchNorm-only by default).
* **The no-replay extension arm is degenerate by construction.** Fine-tuning on new-class labels alone cannot learn the boundary to the old classes, and best-epoch selection hid the collapse (new-class recall 0.03). v2 adds a prototype (nearest-class-mean) extension that needs neither replay nor a boundary.
* **Small data.** 284 subjects, 17 windows per subject, 2.6k to 3.7k training windows per split; seed-to-seed standard deviation 0.02 to 0.04. Differences below that are not interpretable, which is why every interval resamples subjects.
* **Preprocessing was only examined numerically.** I had not looked at the spectrograms. Looking showed: wheeled movement is smooth sloping lines without gait ripple; the per-site background differs in level and shape (14 to 25 dB depending on Doppler bin); subtracting a per-track background leaves dark artefacts and was therefore not adopted.

## Findings that changed the design

* Sites are collection **days** (20250523 and 20250619 are site A, 20250528 C, 20250611 B, 20250612 D). So "location" is confounded with date and conditions. Roughly half of the 2,237 unlabeled tracks come from three dates (20250531, 20250613, 20250620) that have no labeled data and therefore no known site.
* The activity label is constant within every one of the 410 labeled tracks (14 windows per track on average). A deployed tracker supplies track ids, so predictions can be pooled per track, and a source-free method can exploit track consistency.
* Class mix differs by site (site B has far more wheeled movers and less walking), so part of any cross-site gap is label shift, not signal shift.

## Checked and found correct

Subject-disjointness of every split (`assert_disjoint`, rejection of a subject at two sites), per-site bootstrap resampling subjects, balanced accuracy skipping absent classes, hierarchy roll-up (`check_hierarchy_consistency`), head extension (old logits preserved, only the parent node of a tree grows), tamper-free adaptation (the source model is deep-copied), domain index of the held-out site never reaching training, selection flag bookkeeping in the v1 stage scripts.

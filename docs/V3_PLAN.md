# v3 plan (written before any v3 experiment runs)

Goal: raise balanced accuracy at an unseen site above the v2 baseline (plain CNN + track pooling, 0.785 on the confirmatory seeds) by attacking the one confusion that limits it (walking vs dual-hand load), with ideas grounded in the radar physics and in the dataset's structure.

## Protocol (what keeps this honest)

* **Exploratory seeds 0 to 2** (and nested selection of per-method settings as in v2) for every idea below. Nothing from this phase is quoted as a result except as exploratory.
* **Selection rule, fixed now:** the candidate for confirmation is the single configuration with the highest mean track-vote balanced accuracy on the exploratory seeds among configurations whose training pipeline is a *single* model (no ensembles of different configurations), with ties within 0.005 broken toward the simpler configuration (fewer components). Up to two further candidates are named in advance by category winner (best track-level model, best pretraining) and confirmed separately.
* **Confirmation:** fresh seeds **10 to 14** (v2 used 3 to 7), same four held-out sites, no re-tuning (per-site learning rates from the exploratory nested selection). Hypotheses: **H5** the selected candidate beats plain training with track pooling (v2 baseline re-run on seeds 10 to 14) in track-vote balanced accuracy; **H6** each category winner beats the same baseline. A hypothesis is confirmed only if the 95% subject-clustered interval of the difference excludes zero in the stated direction; Bonferroni-adjusted intervals are reported alongside.
* Everything is reported, including failures. No result from synthetic fixtures.

## Ideas (all exploratory until confirmed)

| Id | Idea | Why it might help |
|---|---|---|
| G | Viewing-geometry diagnostic (approach angle by site) | a radar sees only radial velocity, so approach angle changes Doppler scale; sites may differ in it |
| R1 | Torso-aligned input (centre each frame on its Doppler centroid) | removes walking speed and angle, leaves limb ripple, the cue for arm swing |
| R2 | Doppler-stretch augmentation | simulates changes of approach angle |
| R3 | Longer context (concatenate consecutive windows to 3 s) | gait cycles are longer than 1 s |
| H | Head variants: flat, tree with two different topologies, mixture of experts with 2/4/8 experts and top-1/top-2 | different ways of encoding the class structure |
| M | Multitask: 3 classes + 12 fine activities + ordinal arm-swing target | fine labels carry the posture cue the coarse label hides |
| B | Backbone variants: small CNN, ResNet-18, MobileNetV2 | capacity |
| P1 | Track-contrastive pretraining (windows of one track are positives) | the activity is constant within a track, so this pretext is task-aligned |
| T | Track-level models: attention multiple-instance pooling, GRU over window embeddings | the track is the deployment unit and the label unit |
| C | Conformal set prediction with coverage measured at the unseen site | honest handling of the intrinsic ambiguity |
| A | Source-free adaptation on the best model | as v2 |

Combinations: after the single-factor sweeps, a restricted grid over the winners of R, H, M, B, P1, T.

## Addendum 1 (written after the stage D and E sweeps, before any probability-level analysis)

What the sweeps showed (exploratory seeds 0 to 2): the plain CNN with track pooling (0.788) was not beaten by any of the 23 variants; the pre-fixed selection rule therefore selects `base`. Seed ensembles gave +0.013 with an interval spanning zero, diverse ensembles nothing. About half of the foot tracks are classified correctly by some models or seeds and wrongly by others, and each site is wrong in its own direction (site C calls dual-hand tracks walking, site D the reverse). That points at a per-site shift of the walking vs dual-hand decision boundary.

New idea, stage G (exploratory, seeds 0 to 2) and stage H (confirmatory, seeds 10 to 14, same code):

* **Source-free boundary recalibration.** Use only the unlabeled target tracks' model outputs to move the walking vs dual-hand boundary: (M1) EM prior estimation (Saerens et al.), (M2) re-centre the score so half of the foot tracks fall on each side, (M3) place the boundary at the valley of a two-component mixture fitted to the foot-track scores, (M4) re-centre on the mean score. On seeds 0 to 2 the method with the best mean track-vote balanced accuracy is chosen; if none beats no recalibration on those seeds, none is confirmed.
* **Seed ensemble.** Average the log-probabilities of the models of the 5 confirmatory seeds before track pooling.

Confirmatory hypotheses on seeds 10 to 14 (95% subject-clustered intervals; Bonferroni intervals reported): **H7** the chosen recalibration beats plain base with track pooling (mean over the 5 seeds' single models); **H8** the 5-seed ensemble beats the mean single model; **H9** the ensemble plus the chosen recalibration beats the mean single model. Conformal prediction sets are reported descriptively (coverage at the unseen site against the nominal level), with no hypothesis.

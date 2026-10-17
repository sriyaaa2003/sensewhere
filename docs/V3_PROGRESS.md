# v3 progress log (kept so work can resume after an interruption)

State at the time of writing:

* Plan and confirmation rules: docs/V3_PLAN.md (committed before any v3 experiment ran).
* Geometry diagnostic done: docs/GEOMETRY.md (site AUC from geometry alone up to 0.94; walking recall lowest at the site where walkers are slower than carriers).
* Running on Kaggle: stageD (representations, heads, multitask, backbones) and stageE (track models, track-contrastive pretraining), seeds 0 to 2.
* Next: `kaggle kernels output` for saf2003/sensewhere-staged and saf2003/sensewhere-stagee into results/stageD and results/stageE (set PYTHONUTF8=1), run `python scripts/analyze.py`, read the v3 section of docs/RESULTS.md, apply the selection rule, then build stageF (combinations of the category winners, conformal sets, adaptation) and the confirmatory stageG on seeds 10 to 14.

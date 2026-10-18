# v3 progress log

Status: complete. Plan and hypotheses were committed before each experiment (docs/V3_PLAN.md and its two addenda).

* Geometry diagnostic: docs/GEOMETRY.md.
* Exploratory sweeps (stages D, E, F, seeds 0 to 2): 23 single-factor arms; none beat the baseline.
* Design of boundary recalibration (stage G, seeds 0 to 2) and confirmation (stage H, seeds 10 to 14): H7, H8, H9 not confirmed.
* Confirmation of the stage F candidates (stage I, seeds 10 to 14): H5 not confirmed, H6 clearly worse than the baseline.
* The only confirmed effect in the whole project remains track pooling (docs/CONFIRMATION.md, H1).
* Reproduce: kaggle/* scripts, then `python scripts/analyze.py`, `python scripts/analyze_geometry.py`, `python scripts/analyze_recal.py`, `python scripts/make_readme.py`.

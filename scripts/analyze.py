"""Turn the saved Kaggle outputs in results/ into tables (results/tables/*.csv), figures and docs/RESULTS.md.

Every number in RESULTS.md and the README comes from this script. Run: python scripts/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensewhere import analysis  # noqa: E402

R = ROOT / "results"
T = R / "tables"
F = R / "figures"
T.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000
md: list[str] = []


def fmt(m: dict, key="mean", lo="lo", hi="hi", d=3) -> str:
    return f"{m[key]:.{d}f} [{m[lo]:.{d}f}, {m[hi]:.{d}f}]"


def table_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def cell_bootstrap(values: pd.Series, n_boot=N_BOOT, seed=0):
    """Mean and 95% interval over independent cells (site x seed) for quantities without saved predictions."""
    v = values.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_preds(path: Path, prefix: str, y_key: str, subj_key: str, methods_fn):
    z = np.load(path, allow_pickle=True)
    keys = [k for k in z.files if k.startswith(prefix + "|")]
    sites = sorted({k.split("|")[1] for k in keys})
    y = {s: z[f"{y_key}|{s}"] for s in sites}
    subj = {s: z[f"{subj_key}|{s}"] for s in sites}
    preds: dict = {s: {} for s in sites}
    for k in keys:
        parts = k.split("|")
        site, seed = parts[1], parts[-1]
        name = methods_fn(parts)
        preds[site].setdefault(name, {})[int(seed)] = z[k]
    for s in sites:
        for m in list(preds[s]):
            preds[s][m] = np.stack([preds[s][m][sd] for sd in sorted(preds[s][m])])
    return y, subj, preds


# ---------------------------------------------------------------- stage 1: shift
def stage1():
    p = R / "stage1"
    if not (p / "shift_priors.csv").exists():
        return
    pri = pd.read_csv(p / "shift_priors.csv")
    sub = pd.read_csv(p / "shift_subjects.csv")
    auc = pd.read_csv(p / "shift_site_auc.csv")
    md.append("## 1. How much do the sites differ?\n")
    md.append("Windows per site and motion class (after quality control):\n")
    md.append(table_md(pri))
    md.append("\nSubjects per site and class:\n")
    md.append(table_md(sub))
    a = auc[["variant", "scope", "n", "auc_macro", "auc_A", "auc_B", "auc_C", "auc_D"]].round(3)
    a.to_csv(T / "site_recoverability.csv", index=False)
    md.append("\nCan a gradient-boosting model tell the site from the window? (subject-grouped 5-fold, one-vs-rest AUC; 0.5 = cannot)\n")
    md.append(table_md(a))
    prof = R / "stage0" / "dw_loc_profile.json"
    if prof.exists():
        d = json.loads(prof.read_text())
        plt.figure(figsize=(7, 3.2))
        for s, v in d.items():
            plt.plot(np.arange(256) - 128, v, label=f"site {s}")
        plt.xlabel("Doppler bin (0 = static)")
        plt.ylabel("mean log magnitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(F / "site_profiles.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------- stage 2: DG, SFDA, preprocessing
def stage2():
    p = R / "stage2"
    if not (p / "stage2_rows.csv").exists():
        return
    rows = pd.read_csv(p / "stage2_rows.csv")
    sel = rows[rows.selected.astype(str) == "True"]
    # DG
    y, subj, preds = load_preds(p / "stage2_preds.npz", "dg", "y", "subject", lambda parts: parts[2])
    b1 = R / "stage1" / "baseline_preds.npz"
    if b1.exists():
        z = np.load(b1, allow_pickle=True)
        for s in preds:
            ks = sorted(k for k in z.files if k.startswith(f"load|{s}|gbm_features|"))
            if ks:
                preds[s]["gbm_features"] = np.stack([z[k] for k in ks])
    res = analysis.multi_site_bootstrap(y, subj, preds, 3, reference="erm", n_boot=N_BOOT)
    res_f1 = analysis.multi_site_bootstrap(y, subj, preds, 3, reference="erm", metric=analysis.macro_f1, n_boot=N_BOOT)
    per = analysis.per_site_scores(y, preds, 3)
    order = ["gbm_features", "erm", "erm_aug", "coral", "dann", "groupdro"]
    out = []
    for m in order:
        if m not in res:
            continue
        r = res[m]
        out.append({"method": m, "balanced accuracy [95% CI]": fmt(r), "difference vs ERM [95% CI]": fmt(r, "diff", "diff_lo", "diff_hi") if m != "erm" else "reference",
                    "macro-F1": f"{res_f1[m]['mean']:.3f}", **{f"site {s}": f"{per[m][s]:.3f}" for s in sorted(per[m])}})
    df = pd.DataFrame(out)
    df.to_csv(T / "dg.csv", index=False)
    md.append("\n## 2. Training-time methods for an unseen site (3 motion classes, leave-one-location-out)\n")
    md.append("Mean over sites of mean over seeds; intervals resample subjects within each site. Same tuning budget per method (2 settings, chosen on source-site validation only).\n")
    md.append(table_md(df))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ms = [m for m in order if m in res]
    ax.bar(range(len(ms)), [res[m]["mean"] for m in ms], yerr=[[res[m]["mean"] - res[m]["lo"] for m in ms], [res[m]["hi"] - res[m]["mean"] for m in ms]], capsize=3, color="#4c78a8")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, rotation=20)
    ax.set_ylabel("balanced accuracy at the unseen site")
    ax.axhline(1 / 3, color="grey", ls=":")
    plt.tight_layout()
    plt.savefig(F / "dg.png", dpi=150)
    plt.close()
    # source-free adaptation
    y2, subj2, p2 = load_preds(p / "stage2_preds.npz", "sfda", "y", "subject", lambda parts: f"{parts[2]}+{parts[3]}")
    for s in p2:  # add the unadapted models as references
        for base in ["erm", "erm_aug"]:
            p2[s][f"{base}+none"] = preds[s][base]
    out = []
    for base in ["erm", "erm_aug"]:
        ref = f"{base}+none"
        sub_p = {s: {m: v for m, v in p2[s].items() if m.startswith(base + "+")} for s in p2}
        rr = analysis.multi_site_bootstrap(y2, subj2, sub_p, 3, reference=ref, n_boot=N_BOOT)
        for m in sorted(rr, key=lambda k: (k != ref, k)):
            name = m.split("+")[1]
            sel_rows = sel[(sel.stage == "sfda") & (sel.method == base) & (sel.adapt == name)] if name != "none" else sel[(sel.stage == "dg") & (sel.method == base)]
            out.append({"source model": base, "adaptation": name, "balanced accuracy [95% CI]": fmt(rr[m]),
                        "difference vs no adaptation [95% CI]": fmt(rr[m], "diff", "diff_lo", "diff_hi") if name != "none" else "reference",
                        "mean share of target given to one class": f"{sel_rows.collapse.mean():.2f}"})
    df = pd.DataFrame(out)
    df.to_csv(T / "sfda.csv", index=False)
    md.append("\n## 3. Adapting at the new site with unlabeled windows only\n")
    md.append("The source model and the target site's unlabeled windows are the only inputs. The last column is the average share of target windows assigned to the single most-predicted class (1.0 would be total collapse).\n")
    md.append(table_md(df))
    # preprocessing ablation (rows; ERM lr=1e-3 on each variant)
    base_rows = rows[(rows.stage == "dg") & (rows.method == "erm") & (rows.cfg == json.dumps({"lr": 0.001}))].assign(variant="base")
    prep = rows[rows.stage == "prep"].rename(columns={"cfg": "variant"})
    allp = pd.concat([base_rows, prep])
    names = {"base": "per-window standardised, static band removed (default)", "nosup": "no static-band removal", "robust": "median/MAD instead of mean/std", "raw": "no normalisation (raw dB)"}
    out = []
    for v, g in allp.groupby("variant"):
        per_cell = g.groupby(["site", "seed"]).bal_acc.mean()
        mn, lo, hi = cell_bootstrap(per_cell)
        out.append({"preprocessing": names[v], "balanced accuracy [95% CI over site x seed cells]": f"{mn:.3f} [{lo:.3f}, {hi:.3f}]"})
    df = pd.DataFrame(out)
    df.to_csv(T / "preprocessing.csv", index=False)
    md.append("\n## 4. Does preprocessing matter?\n")
    md.append(table_md(df))


# ---------------------------------------------------------------- stage 3 / 5: heads and extension
def extension(stage_dir: str, title: str, has_freeze: bool):
    p = R / stage_dir
    f = next(iter(p.glob("stage*_rows.csv")), None) if p.exists() else None
    if f is None:
        return
    rows = pd.read_csv(f)
    ext = rows[rows.stage == "extend"].copy()
    if ext.empty:
        return
    ext["drop"] = ext.base_old_bal_acc - ext.bal_acc
    keys = ["head", "replay"] + (["freeze"] if has_freeze else [])
    out = []
    for k, g in ext.groupby(keys):
        cells = g.groupby(["site", "seed"])
        d, dlo, dhi = cell_bootstrap(cells["drop"].mean())
        nr, nlo, nhi = cell_bootstrap(cells["new_recall"].mean())
        jb = g.get("joint_bal_acc", g.get("load_bal_acc"))
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update({"old-class accuracy before": f"{g.base_old_bal_acc.mean():.3f}", "after": f"{g.bal_acc.mean():.3f}",
                    "drop [95% CI]": f"{d:.3f} [{dlo:.3f}, {dhi:.3f}]", "new-class recall [95% CI]": f"{nr:.3f} [{nlo:.3f}, {nhi:.3f}]",
                    "trainable parameters": int(g.trainable.mean())})
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(T / f"{stage_dir}_extension.csv", index=False)
    md.append(f"\n{title}\n")
    md.append(table_md(df))
    plt.figure(figsize=(6, 3.2))
    for h, g in ext[ext.freeze.astype(str) == "True"].groupby("head") if has_freeze else ext.groupby("head"):
        s = g.groupby("replay")["drop"].mean()
        plt.plot(s.index * 100, s.values, marker="o", label=h)
    plt.xlabel("replay budget (% of old training windows)")
    plt.ylabel("drop in old-class accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(F / f"{stage_dir}_extension.png", dpi=150)
    plt.close()


def heads():
    p = R / "stage3" / "stage3_rows.csv"
    if not p.exists():
        return
    rows = pd.read_csv(p)
    h = rows[rows.stage == "heads"]
    out = []
    for head, g in h.groupby("head"):
        cells = g.groupby(["site", "seed"])
        a = cell_bootstrap(cells.bal_acc.mean())
        l = cell_bootstrap(cells.load_bal_acc.mean())
        out.append({"head": head, "12-activity balanced accuracy [95% CI]": f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]",
                    "rolled up to 3 classes [95% CI]": f"{l[0]:.3f} [{l[1]:.3f}, {l[2]:.3f}]", "parameters": int(g.params.mean())})
    df = pd.DataFrame(out)
    df.to_csv(T / "heads.csv", index=False)
    md.append("\n## 5. Flat softmax, hierarchical tree or mixture of experts on the 12 fine activities\n")
    md.append(table_md(df))


def shot_control():
    p = R / "stage6" / "stage6_rows.csv"
    if not p.exists():
        return
    rows = pd.read_csv(p)
    ref = rows[rows.method == "none"].set_index(["site", "seed"]).bal_acc
    out = []
    for (aset, name), g in rows[rows.method != "none"].groupby(["adapt_set", "method"]):
        v = g.set_index(["site", "seed"]).bal_acc
        diff = v - ref.reindex(v.index)
        a = cell_bootstrap(diff)
        out.append({"adapted on": aset, "method": name, "balanced accuracy": f"{v.mean():.3f}", "change vs no adaptation [95% CI]": f"{a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]"})
    df = pd.DataFrame(out)
    df.to_csv(T / "shot_control.csv", index=False)
    md.append(chr(10) + "## 3b. Why does information maximisation hurt? Controlling the class mix of the adaptation data" + chr(10))
    md.append("SHOT-style adaptation adds a diversity term that pushes the average prediction toward a uniform class mix. Each row adapts on either all unlabeled target windows (natural class mix) or a class-balanced subsample, with diversity weight 0, 0.3 or 1, and is always evaluated on all target windows." + chr(10))
    md.append(table_md(df))


def widar():
    p = R / "stage4" / "stage4_rows.csv"
    if not p.exists():
        return
    rows = pd.read_csv(p)
    sel = rows[rows.selected.astype(str) == "True"]
    md.append("\n## 7. Replication on Wi-Fi sensing (Widar 3.0, session 20181130, 5 torso locations, 9 users)\n")
    out = []
    ref_dg = sel[(sel.stage == "dg") & (sel.method == "erm")].groupby(["site", "seed"]).bal_acc.mean()
    for m, g in sel[sel.stage == "dg"].groupby("method"):
        v = g.groupby(["site", "seed"]).bal_acc.mean()
        a = cell_bootstrap(v)
        d = cell_bootstrap(v - ref_dg.reindex(v.index)) if m != "erm" else None
        out.append({"training method": m, "balanced accuracy [95% CI over location x seed cells]": f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]",
                    "paired difference vs ERM [95% CI]": f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "reference"})
    df = pd.DataFrame(out)
    df.to_csv(T / "widar_dg.csv", index=False)
    md.append(table_md(df))
    out = []
    ref = sel[(sel.stage == "dg") & (sel.method == "erm")].groupby(["site", "seed"]).bal_acc.mean()
    out.append({"adaptation": "none", "balanced accuracy": f"{ref.mean():.3f}", "change vs none [95% CI]": "reference", "collapse share": "-"})
    for name, g in sel[sel.stage == "sfda"].groupby("adapt"):
        v = g.groupby(["site", "seed"]).bal_acc.mean()
        diff = (v - ref.reindex(v.index))
        a = cell_bootstrap(diff)
        out.append({"adaptation": name, "balanced accuracy": f"{v.mean():.3f}", "change vs none [95% CI]": f"{a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]", "collapse share": f"{g.collapse.mean():.2f}"})
    df = pd.DataFrame(out)
    df.to_csv(T / "widar_sfda.csv", index=False)
    md.append("\nSource-free adaptation on Widar:\n")
    md.append(table_md(df))
    ext = sel[sel.stage == "extend"].copy()
    if not ext.empty:
        out = []
        for (head, rf), g in ext.groupby(["head", "replay"]):
            c = g.groupby(["site", "seed"])
            d = cell_bootstrap(c.base_old_bal_acc.mean() - c.bal_acc.mean())
            nr = cell_bootstrap(c.new_recall.mean())
            out.append({"head": head, "replay": rf, "old-class drop [95% CI]": f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]", "new-class recall [95% CI]": f"{nr[0]:.3f} [{nr[1]:.3f}, {nr[2]:.3f}]"})
        df = pd.DataFrame(out)
        df.to_csv(T / "widar_extension.csv", index=False)
        md.append("\nClass extension on Widar (gestures 1-6, then add 7-9):\n")
        md.append(table_md(df))


# ---------------------------------------------------------------- v2 (fixed protocol)
def stageA():
    p = R / "stageA"
    if not (p / "stageA_rows.csv").exists():
        return
    rows = pd.read_csv(p / "stageA_rows.csv")
    y, subj, preds = load_preds(p / "stageA_preds.npz", "v2", "y", "subject", lambda parts: "|".join(parts[2:5]))
    md.append("\n# v2: fixed protocol, new components\n")
    md.append("v2 selects every setting by nested leave-one-source-site-out (the target site never influences a choice), augmentation is an explicit switch, and windows are standardised inside the network. 'vote' pools predictions over each track (labels are constant within a track).\n")
    train_arms = ["erm", "erm_aug", "erm_floor", "erm_aug_floor", "coral", "groupdro", "sitecal_q50", "sitecal_q10", "ssl_src", "ssl_all"]
    names = {"erm": "plain training", "erm_aug": "+ augmentation", "erm_floor": "+ noise-floor transplant", "erm_aug_floor": "+ augmentation + transplant", "coral": "CORAL", "groupdro": "Group-DRO",
             "sitecal_q50": "site calibration (median profile)", "sitecal_q10": "site calibration (10th percentile)", "ssl_src": "masked-reconstruction pretraining, source-site recordings only", "ssl_all": "masked-reconstruction pretraining, all unlabeled recordings"}
    sub = {s: {f"{a}|{r}": preds[s][f"{a}|none|{r}"] for a in train_arms for r in ("window", "vote") if f"{a}|none|{r}" in preds[s]} for s in preds}
    res = analysis.multi_site_bootstrap(y, subj, sub, 3, reference="erm|window", n_boot=N_BOOT)
    out = []
    for a in train_arms:
        if f"{a}|window" not in res:
            continue
        w, v = res[f"{a}|window"], res[f"{a}|vote"]
        out.append({"training arm": names[a], "window balanced accuracy [95% CI]": fmt(w), "diff vs plain [95% CI]": fmt(w, "diff", "diff_lo", "diff_hi") if a != "erm" else "reference",
                    "track-vote balanced accuracy [95% CI]": fmt(v), "vote diff vs plain window [95% CI]": fmt(v, "diff", "diff_lo", "diff_hi")})
    df = pd.DataFrame(out)
    df.to_csv(T / "v2_training_arms.csv", index=False)
    md.append("\n## v2-1. Training arms (3 motion classes, unseen site)\n")
    md.append(table_md(df))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ar = [a for a in train_arms if f"{a}|window" in res]
    xs = np.arange(len(ar))
    for k, (r, c) in enumerate((("window", "#4c78a8"), ("vote", "#f58518"))):
        ax.bar(xs + (k - 0.5) * 0.38, [res[f"{a}|{r}"]["mean"] for a in ar], 0.38, yerr=[[res[f"{a}|{r}"]["mean"] - res[f"{a}|{r}"]["lo"] for a in ar], [res[f"{a}|{r}"]["hi"] - res[f"{a}|{r}"]["mean"] for a in ar]], capsize=2, color=c, label=r)
    ax.set_xticks(xs)
    ax.set_xticklabels(ar, rotation=30, ha="right")
    ax.set_ylabel("balanced accuracy, unseen site")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(F / "v2_training_arms.png", dpi=150)
    plt.close()
    # adaptation
    out = []
    for arm in ("erm", "erm_floor", "ssl_src"):
        keys = [k for k in {k for s in preds for k in preds[s]} if k.startswith(arm + "|")]
        sub = {s: {k: preds[s][k] for k in keys if k in preds[s]} for s in preds}
        ref = f"{arm}|none|window"
        rr = analysis.multi_site_bootstrap(y, subj, sub, 3, reference=ref, n_boot=N_BOOT)
        for adapt_name in ("none", "adabn", "tent_e1", "shot_bn", "shot_all", "tcst"):
            if f"{arm}|{adapt_name}|window" not in rr:
                continue
            w, v = rr[f"{arm}|{adapt_name}|window"], rr[f"{arm}|{adapt_name}|vote"]
            out.append({"source model": names[arm], "adaptation": adapt_name, "window [95% CI]": fmt(w), "change vs none [95% CI]": fmt(w, "diff", "diff_lo", "diff_hi") if adapt_name != "none" else "reference",
                        "track-vote [95% CI]": fmt(v), "vote change vs none window [95% CI]": fmt(v, "diff", "diff_lo", "diff_hi")})
    df = pd.DataFrame(out)
    df.to_csv(T / "v2_adaptation.csv", index=False)
    md.append("\n## v2-2. Source-free adaptation (unlabeled target windows and track ids only)\n")
    md.append("TCST = track-consistency self-training: pool predictions per track, keep the most confident tracks per class, train BatchNorm scale and shift so every window of a track follows its track consensus.\n")
    md.append(table_md(df))
    w = rows[(rows.adapt == "none") & (rows.readout == "window") & rows.val_bal_acc.notna()]
    md.append(f"\nSource-validation balanced accuracy versus accuracy at the unseen site over {len(w)} v2 runs: correlation {w[['val_bal_acc', 'bal_acc']].corr().iloc[0, 1]:.2f} (v1: 0.30); mean {w.val_bal_acc.mean():.3f} vs {w.bal_acc.mean():.3f}.\n")
    sel = p / "stageA_selection.csv"
    if sel.exists():
        sl = pd.read_csv(sel)
        pre = sl[sl.kind == "pretrain"][["site", "ssl", "n", "first", "last"]].round(3)
        pre.to_csv(T / "v2_pretraining.csv", index=False)
        md.append("\nMasked-reconstruction pretraining (windows used, loss first to last epoch):\n")
        md.append(table_md(pre))


def stageB():
    p = R / "stageB" / "stageB_rows.csv"
    if not p.exists():
        return
    rows = pd.read_csv(p)
    ext = rows[rows.stage == "extend"].copy()
    ext["drop"] = ext.base_old_bal_acc - ext.bal_acc
    out = []
    for (head, mode, replay, shots), g in ext.groupby(["head", "mode", "replay", "shots"], dropna=False):
        c = g.groupby(["site", "seed"])
        d = cell_bootstrap(c["drop"].mean())
        nr = cell_bootstrap(c.new_recall.mean())
        jb = cell_bootstrap(c.joint_bal_acc.mean())
        out.append({"method": head, "backbone": mode, "replay": replay, "shots": "-" if pd.isna(shots) else ("all" if shots == -1 else int(shots)),
                    "old-class drop [95% CI]": f"{d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]", "new-class recall [95% CI]": f"{nr[0]:.3f} [{nr[1]:.3f}, {nr[2]:.3f}]",
                    "joint balanced accuracy [95% CI]": f"{jb[0]:.3f} [{jb[1]:.3f}, {jb[2]:.3f}]", "trainable parameters": int(g.trainable.mean())})
    df = pd.DataFrame(out)
    df.to_csv(T / "v2_extension.csv", index=False)
    md.append("\n## v2-3. Adding the single-hand-load class (BatchNorm statistics now frozen with the backbone; prototype extension added)\n")
    md.append("ncm = nearest class mean on the base model embedding: old classes keep one stored mean each, the new class is the mean of N labeled windows; nothing is retrained.\n")
    md.append(table_md(df))
    plt.figure(figsize=(6.4, 3.4))
    for (head, mode), g in ext[ext["mode"].isin(["frozen", "prototype"])].groupby(["head", "mode"]):
        if head == "ncm":
            s = g.groupby("shots").new_recall.mean()
            plt.plot(range(len(s)), s.values, marker="s", label="ncm (shots 5/20/all)")
        else:
            s = g.groupby("replay").new_recall.mean()
            plt.plot(range(len(s)), s.values, marker="o", label=f"{head} head")
    plt.xlabel("budget step (replay 0/5/20% or shots 5/20/all)")
    plt.ylabel("new-class recall")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(F / "v2_extension.png", dpi=150)
    plt.close()


def stageA_errors():
    p = R / "stageA"
    if not (p / "stageA_preds.npz").exists():
        return
    y, subj, preds = load_preds(p / "stageA_preds.npz", "v2", "y", "subject", lambda parts: "|".join(parts[2:5]))
    classes = ["normal_walking", "dual-hand load", "on wheels"]
    # pre-specified contrasts (written before looking at the table; no search over arms)
    contrasts = [
        ("track pooling alone", "erm|none|vote", "erm|none|window"),
        ("augmentation, window readout", "erm_aug|none|window", "erm|none|window"),
        ("BatchNorm-only SHOT on the transplant model, window readout", "erm_floor|shot_bn|window", "erm_floor|none|window"),
        ("whole pipeline: transplant + BatchNorm-only SHOT + track pooling vs plain window", "erm_floor|shot_bn|vote", "erm|none|window"),
        ("whole pipeline vs plain model with track pooling only", "erm_floor|shot_bn|vote", "erm|none|vote"),
    ]
    out = []
    for name, a, b in contrasts:
        sub = {s: {a: preds[s][a], b: preds[s][b]} for s in preds if a in preds[s] and b in preds[s]}
        r = analysis.multi_site_bootstrap({s: y[s] for s in sub}, {s: subj[s] for s in sub}, sub, 3, reference=b, n_boot=N_BOOT)
        out.append({"contrast": name, "A": a, "B": b, "A [95% CI]": fmt(r[a]), "B [95% CI]": fmt(r[b]), "A - B [95% CI]": fmt(r[a], "diff", "diff_lo", "diff_hi")})
    df = pd.DataFrame(out)
    df.to_csv(T / "v2_contrasts.csv", index=False)
    md.append(chr(10) + "## v2-4. Five exploratory contrasts (chosen after inspecting the arm tables)" + chr(10))
    md.append("The tables above compare many arms, so some differences look large by chance. These five were chosen after inspecting those tables, so the two pipeline contrasts are optimistic (the pipeline is the best of many combinations). The unbiased test is the confirmatory run on fresh seeds in v2-5, whose hypotheses were fixed in docs/CONFIRMATION.md before it ran." + chr(10))
    md.append(table_md(df))
    # per-class recall and confusion for the plain model
    from sensewhere.metrics import confusion

    for key, title in (("erm|none|window", "plain model, window readout"), ("erm|none|vote", "plain model, track-vote readout")):
        cms = []
        for s in sorted(preds):
            for seed_pred in preds[s][key]:
                cm = confusion(y[s], seed_pred, 3).astype(float)
                cms.append(cm / cm.sum(1, keepdims=True).clip(1))
        cm = np.mean(cms, axis=0)
        t = pd.DataFrame(np.round(cm, 3), index=[f"true {c}" for c in classes], columns=[f"predicted {c}" for c in classes]).reset_index().rename(columns={"index": ""})
        md.append(f"\nRow-normalised confusion, {title} (mean over sites and seeds):\n")
        md.append(table_md(t))
        t.to_csv(T / f"v2_confusion_{key.split('|')[2]}.csv", index=False)
    per_site = []
    for s in sorted(preds):
        rec = {}
        for c in range(3):
            m = y[s] == c
            rec[classes[c]] = float(np.mean([np.mean(p[m] == c) for p in preds[s]["erm|none|window"]]))
        per_site.append({"site": s, **{k: f"{v:.3f}" for k, v in rec.items()}})
    md.append("\nPer-class recall of the plain model at each unseen site (window readout):\n")
    md.append(table_md(pd.DataFrame(per_site)))


def stageC():
    p = R / "stageC" / "stageC_preds.npz"
    if not p.exists():
        return
    y, subj, preds = load_preds(p, "v2", "y", "subject", lambda parts: "|".join(parts[2:5]))
    tests = [
        ("H1 track pooling raises plain training", "erm|none|vote", "erm|none|window"),
        ("H2 pipeline beats plain + track pooling", "erm_floor|shot_bn|vote", "erm|none|vote"),
        ("H3 pipeline beats plain, window readout", "erm_floor|shot_bn|vote", "erm|none|window"),
        ("H4 augmentation + pooling not worse than plain + pooling", "erm_aug|none|vote", "erm|none|vote"),
    ]
    out = []
    for name, a, b in tests:
        sub = {s: {a: preds[s][a], b: preds[s][b]} for s in preds}
        r95 = analysis.multi_site_bootstrap(y, subj, sub, 3, reference=b, n_boot=N_BOOT, ci=95.0)
        rbf = analysis.multi_site_bootstrap(y, subj, sub, 3, reference=b, n_boot=N_BOOT, ci=100 * (1 - 0.05 / 4))
        confirmed = r95[a]["diff_lo"] > 0
        out.append({"hypothesis": name, "A [95% CI]": fmt(r95[a]), "B [95% CI]": fmt(r95[b]), "A - B [95% CI]": fmt(r95[a], "diff", "diff_lo", "diff_hi"),
                    "A - B [Bonferroni 99.4% CI]": fmt(rbf[a], "diff", "diff_lo", "diff_hi"), "95% interval excludes zero": "yes" if confirmed else "no"})
    df = pd.DataFrame(out)
    df.to_csv(T / "v2_confirmation.csv", index=False)
    md.append(chr(10) + "## v2-5. Confirmatory run on fresh seeds 3 to 7 (hypotheses fixed in docs/CONFIRMATION.md before the run)" + chr(10))
    md.append(table_md(df))


# ---------------------------------------------------------------- v3 exploratory sweeps (stage D, E, F)
V3_CATEGORY = {
    "base": "-", "aligned": "R", "stretch": "R", "aligned_stretch": "R", "ctx3": "R", "ctx3_aligned": "R",
    "hier_foot": "H", "hier_walk": "H", "moe_e2k1": "H", "moe_e4k2": "H", "moe_e8k2": "H", "moe_e8k1": "H",
    "multitask_w05": "M", "multitask_w1": "M", "resnet18": "B", "mobilenetv2": "B",
    "mil_attn": "T", "mil_gru": "T", "mil_attn_aligned": "T", "mil_attn_stretch": "T",
    "con_src_ft": "P", "con_all_ft": "P", "con_src_mil_attn": "P",
}
V3_COMPLEXITY = {"base": 0, "aligned": 1, "stretch": 1, "aligned_stretch": 2, "ctx3": 1, "ctx3_aligned": 2, "hier_foot": 1, "hier_walk": 1, "moe_e2k1": 1, "moe_e4k2": 1,
                 "moe_e8k2": 1, "moe_e8k1": 1, "multitask_w05": 1, "multitask_w1": 1, "resnet18": 1, "mobilenetv2": 1, "mil_attn": 2, "mil_gru": 2, "mil_attn_aligned": 3,
                 "mil_attn_stretch": 3, "con_src_ft": 2, "con_all_ft": 2, "con_src_mil_attn": 4}


def v3_load():
    ys, ss, ps = {}, {}, {}
    for stage in ("stageD", "stageE", "stageF"):
        f = R / stage / f"{stage}_preds.npz"
        if not f.exists():
            continue
        y, subj, preds = load_preds(f, "v3", "y", "subject", lambda parts: "|".join(parts[2:5]))
        for s in preds:
            ys.setdefault(s, y[s])
            ss.setdefault(s, subj[s])
            ps.setdefault(s, {}).update(preds[s])
    return ys, ss, ps


def stage_v3():
    y, subj, preds = v3_load()
    if not preds:
        return
    arms = sorted({k.split("|")[0] for s in preds for k in preds[s]}, key=lambda a: (V3_CATEGORY.get(a, "Z"), a))
    ref = "base|none|vote"
    sub = {s: {k: v for k, v in preds[s].items() if k.endswith("|vote") or k.endswith("|window")} for s in preds}
    res = analysis.multi_site_bootstrap(y, subj, sub, 3, reference=ref, n_boot=N_BOOT)
    out = []
    for a in arms:
        v, w = res.get(f"{a}|none|vote"), res.get(f"{a}|none|window")
        if v is None:
            continue
        out.append({"arm": a, "idea": V3_CATEGORY.get(a, "?"), "track-vote balanced accuracy [95% CI]": fmt(v),
                    "vote diff vs base [95% CI]": fmt(v, "diff", "diff_lo", "diff_hi") if a != "base" else "reference",
                    "window balanced accuracy [95% CI]": fmt(w) if w else "n/a (track model)"})
    df = pd.DataFrame(out)
    df.to_csv(T / "v3_arms.csv", index=False)
    md.append("\n# v3: exploratory sweeps (seeds 0 to 2; see docs/V3_PLAN.md; nothing here is a confirmed result)\n")
    md.append("Letters: R representation, H head, M multitask, B backbone, T track-level model, P pretraining. All arms use plain training with the settings in the stage scripts; each differs from `base` in one thing.\n")
    md.append(table_md(df))
    # candidate selection by the rule fixed in docs/V3_PLAN.md
    ranked = sorted([(res[f"{a}|none|vote"]["mean"], a) for a in arms if f"{a}|none|vote" in res], reverse=True)
    top = ranked[0][0]
    near = [a for m_, a in ranked if top - m_ <= 0.005]
    cand = min(near, key=lambda a: (V3_COMPLEXITY.get(a, 9), -res[f"{a}|none|vote"]["mean"]))
    winners = {}
    for cat in sorted({c for c in V3_CATEGORY.values() if c != "-"}):
        c_arms = [(res[f"{a}|none|vote"]["mean"], a) for a in arms if V3_CATEGORY.get(a) == cat and f"{a}|none|vote" in res]
        if c_arms:
            winners[cat] = max(c_arms)[1]
    (T / "v3_selection.json").write_text(json.dumps({"candidate": cand, "top_arm": ranked[0][1], "category_winners": winners}, indent=1), encoding="utf-8")
    md.append(f"\nCandidate selected by the pre-fixed rule: **{cand}** (highest track-vote balanced accuracy {top:.3f} by `{ranked[0][1]}`; ties within 0.005 go to the simpler arm). Category winners: {winners}.\n")
    # where does the confusion move? per-class recall of key arms at each unseen site
    classes = ["walking", "dual-hand load", "on wheels"]
    key_arms = [a for a in ["base", "aligned", "stretch", "ctx3", cand] if f"{a}|none|vote" in sub[next(iter(sub))]]
    tab = []
    for a in dict.fromkeys(key_arms):
        for s in sorted(y):
            recs = [np.mean(p[y[s] == c] == c) for p in sub[s][f"{a}|none|vote"] for c in range(3)]
            r3 = np.array(recs).reshape(-1, 3).mean(0)
            tab.append({"arm": a, "site": s, **{f"{classes[c]} recall": f"{r3[c]:.3f}" for c in range(3)}})
    md.append("\nPer-class recall (track vote) by arm and held-out site, to see where the walking vs dual-hand confusion moves (site B is where walkers are slower than carriers):\n")
    md.append(table_md(pd.DataFrame(tab)))


if __name__ == "__main__":
    md.append("# Results\n\nGenerated by `scripts/analyze.py` from the files in `results/`. Do not edit by hand.\n")
    stage1()
    stage2()
    shot_control()
    heads()
    extension("stage3", "## 6a. Adding 7 new fine activities to the 12-activity model (frozen backbone)", False)
    extension("stage5", "## 6b. Adding the single-hand-load class to the 3-class model", True)
    widar()
    stageA()
    stageA_errors()
    stageC()
    stage_v3()
    stageB()
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote docs/RESULTS.md")

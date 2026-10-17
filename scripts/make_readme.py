"""Fill README.template.md from results/tables/*.csv so no number in the README is typed by hand.

Placeholders:
  {{csv|key=value|key=value|column}}   one cell; filters are exact matches on column values; a filter written
                                       "=text|column" matches the first column on its text (row label)
  {{TABLE:csv}}                        the whole table as markdown
Fails loudly if a placeholder matches zero or several rows.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


def table_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def cell(spec: str) -> str:
    parts = spec.split("|")
    df, column, filters = load(parts[0]), parts[-1], parts[1:-1]
    for f in filters:
        if f.startswith("="):  # row label in the first column
            df = df[df.iloc[:, 0].astype(str) == f[1:]]
            continue
        k, v = f.split("=", 1)
        col = df[k]
        df = df[col.astype(str) == v] if not pd.api.types.is_numeric_dtype(col) else df[col.astype(float) == float(v)]
    if len(df) != 1:
        raise SystemExit(f"placeholder {{{{{spec}}}}} matched {len(df)} rows")
    v = df.iloc[0][column]
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    text = (ROOT / "README.template.md").read_text(encoding="utf-8")
    text = re.sub(r"\{\{TABLE:([^}]+)\}\}", lambda m: table_md(load(m.group(1))), text)
    text = re.sub(r"\{\{([^}]+)\}\}", lambda m: cell(m.group(1)), text)
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print("wrote README.md")


if __name__ == "__main__":
    sys.exit(main())

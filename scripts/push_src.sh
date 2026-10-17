#!/usr/bin/env bash
# Package src/sensewhere as the private Kaggle dataset saf2003/sensewhere-src (create on first run, new version afterwards).
set -euo pipefail
K="/c/Users/sriya/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/Scripts/kaggle.exe"
export PYTHONUTF8=1
STAGE=".stage_src"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp src/sensewhere/*.py "$STAGE/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "sensewhere src", "id": "saf2003/sensewhere-src", "licenses": [{"name": "MIT"}]}
JSON
if "$K" datasets status saf2003/sensewhere-src >/dev/null 2>&1; then
  "$K" datasets version -p "$STAGE" -m "${1:-update}" 
else
  "$K" datasets create -p "$STAGE" 
fi

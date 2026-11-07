#!/usr/bin/env bash
# Reproduces the datasets used by LUMEN-X from public GitHub mirrors (the original LOL hosts,
# Google Drive / Baidu, are not scriptable). Provenance: third-party mirrors; LOL © Wei et al. 2018
# (research use). Verify integrity afterwards: python scripts/verify_data.py --data data
set -euo pipefail
DATA="${1:-data}"; TMP="$(mktemp -d)"; mkdir -p "$DATA/unseen"
echo "[1/3] LOL (our485 + eval15)"
git clone --depth 1 -q https://github.com/misbahulamin/Low-light-image-enhancement-GAN "$TMP/lol"
rm -rf "$DATA/LOL"; cp -r "$TMP/lol/lol_dataset" "$DATA/LOL"
echo "[2/3] DICM + LIME"
git clone --depth 1 -q https://github.com/23225114YKW/Low-light-image-enhancement-with-LoL-dataset "$TMP/dl"
rm -rf "$DATA/unseen/DICM" "$DATA/unseen/LIME"; cp -r "$TMP/dl/test_data/DICM" "$TMP/dl/test_data/LIME" "$DATA/unseen/"
echo "[3/3] ExDark (deterministic 100-image subset of the test split, every 7th file)"
git -c gc.auto=0 clone --depth 1 --filter=blob:none --no-checkout -q https://github.com/Chinese-sausage/ExDark "$TMP/exd"
( cd "$TMP/exd"
  git ls-tree -r --name-only HEAD | grep -E '^ExDark/test/images/.*\.(jpg|jpeg|png|JPG|JPEG|PNG)$' | sort | awk 'NR%7==1' | head -100 > ../exd_pick.txt
  xargs -a ../exd_pick.txt -n 50 git checkout HEAD -- )
rm -rf "$DATA/unseen/ExDark100"; mkdir -p "$DATA/unseen/ExDark100"
while read -r p; do cp "$TMP/exd/$p" "$DATA/unseen/ExDark100/"; done < "$TMP/exd_pick.txt"
rm -rf "$TMP" || true   # git auto-gc may still hold files; harmless
python3 scripts/prepare_data.py --lol "$DATA/LOL" --out "$DATA/splits.json"
echo "done. counts:"; for d in "$DATA/LOL/our485/low" "$DATA/LOL/eval15/low" "$DATA/unseen/DICM" "$DATA/unseen/LIME" "$DATA/unseen/ExDark100"; do echo "  $d: $(ls "$d" | wc -l)"; done

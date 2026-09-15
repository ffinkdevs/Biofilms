#!/usr/bin/env bash
# Build the Odin biofilm port. Headless binary needs only Odin in PATH;
# the viewer additionally needs raylib in the link path
# (nix-shell -p raylib, or add pkgs.raylib to the flake devShell).
set -euo pipefail
cd "$(dirname "$0")"

OPT="${OPT:--o:speed}"
OUTDIR="${OUTDIR:-build}"
mkdir -p "$OUTDIR"

echo "== tests =="
odin test tests/ -o:none

echo "== biofilm (headless) =="
odin build . -out:"$OUTDIR/biofilm" "$OPT"
echo "built $OUTDIR/biofilm"

if [[ "${1:-}" == "viewer" || "${1:-}" == "all" ]]; then
  echo "== biofilm-viewer (raylib) =="
  odin build viewer -out:"$OUTDIR/biofilm-viewer" "$OPT"
  echo "built $OUTDIR/biofilm-viewer"
fi

if [[ "${1:-}" == "all" ]]; then
  echo "== compute shader =="
  if command -v glslangValidator >/dev/null; then
    glslangValidator -V shaders/field_diffusion.comp -o "$OUTDIR/field_diffusion.spv"
    echo "built $OUTDIR/field_diffusion.spv"
  else
    echo "glslangValidator not found; skipping SPIR-V (shader ships as source)"
  fi
fi

echo "== smoke =="
"$OUTDIR/biofilm" --n 20 --mcs 2 --seed 42 --layout aos --no-stats | tail -n 3
"$OUTDIR/biofilm" --n 20 --mcs 2 --seed 42 --layout aosoa --no-stats | grep -c '^CSV' | xargs -I{} echo "aosoa CSV lines: {}"

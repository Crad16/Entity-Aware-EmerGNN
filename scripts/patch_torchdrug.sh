#!/usr/bin/env bash
# torchdrug 0.2.1 does not compile its sparse kernels against torch 2.x: two headers moved.
# Patch the installed copy in place (originals are kept as *.orig).
set -euo pipefail
# locate the package without importing it (importing triggers the failing JIT build)
PKG=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('torchdrug').origin))")
EXT=$PKG/layers/functional/extension
for f in $(grep -l -E "ATen/SparseTensorUtils.h|THC/THCAtomics.cuh" "$EXT"/*.h "$EXT"/*.cu "$EXT"/*.cpp); do
    cp --update=none "$f" "$f.orig"
    sed -i 's#<ATen/SparseTensorUtils.h>#<ATen/native/SparseTensorUtils.h>#; s#<THC/THCAtomics.cuh>#<ATen/cuda/Atomic.cuh>#' "$f"
    echo "patched $f"
done
rm -rf ~/.cache/torch_extensions/*/spmm* 2>/dev/null || true

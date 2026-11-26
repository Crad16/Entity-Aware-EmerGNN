# source scripts/env.sh  -- Python venv + CUDA toolkit for torchdrug's JIT kernel
DS535_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DS535_ROOT/.venv/bin/activate"
# Machine-specific overrides (CUDA_HOME, TORCH_CUDA_ARCH_LIST, cache dirs) go in scripts/env.local.sh
[ -f "$DS535_ROOT/scripts/env.local.sh" ] && source "$DS535_ROOT/scripts/env.local.sh"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # must contain bin/nvcc
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
# Shared server: cap CPU threads (by default torch grabs ~15 cores per process)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

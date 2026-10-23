# source scripts/env.sh  -- Python venv + CUDA toolkit for torchdrug's JIT kernel
DS535_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DS535_ROOT/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
# Shared server: cap CPU threads (by default torch grabs ~15 cores per process)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

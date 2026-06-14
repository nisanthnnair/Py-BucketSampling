#!/usr/bin/env bash
# Launcher for the unified bucket-sampling reweighting driver bucket_reweight.py
#
# bucket_reweight.py is now SELF-CONTAINED (the ct/vbias/probability/stitching
# logic is all inlined -- no helper .py files are imported).  The driver itself is
# SERIAL (one bucket folder at a time); it only parallelises the vbias step, which
# it runs by re-launching ITSELF under mpirun in a hidden worker mode:
#     mpirun -np NRANKS python bucket_reweight.py <inp> --vbias-worker ...
# Therefore this script does NOT launch python under mpirun -- it just sets the
# thread/MPI environment and runs the driver once.
#
# Usage: ./run_bucket_reweight.sh [NRANKS] [THREADS] [BACKEND] [INPUT] [ROOT] [FORCE]
#   NRANKS   : MPI ranks for the vbias step          (default 2)
#   THREADS  : OpenMP threads per rank for vbias      (default 1)
#   BACKEND  : numpy | numba                          (default numpy)
#   INPUT    : control file                           (default bucketsampling.inp)
#   ROOT     : base dir holding the bucket folders    (default .)
#   FORCE    : force|--force|1|yes -> recompute ct.dat/vbias.dat even if present
#                                                     (default: reuse existing files)
#
# Total cores used by vbias = NRANKS * THREADS. Pick so it fits your node.
set -euo pipefail

NRANKS=${1:-2}
THREADS=${2:-1}
BACKEND=${3:-numpy}
INPUT=${4:-bucketsampling.inp}
ROOT=${5:-.}
FORCE=${6:-}

# Turn the 6th arg into the driver's --force flag when set to a truthy value.
FORCE_FLAG=""
case "$(printf '%s' "$FORCE" | tr '[:upper:]' '[:lower:]')" in
    force|--force|1|yes|y|true) FORCE_FLAG="--force" ;;
esac

# Thread counts for the per-rank shared-memory (OpenMP / BLAS / numba) layer.
export OMP_NUM_THREADS=$THREADS
export NUMBA_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS

# macOS: the default per-user $TMPDIR is often not writable by OpenMPI/PMIx,
# which then fails to mkdir its session dir. Use a clean /tmp (override with
# MPI_TMPDIR if desired).
export TMPDIR="${MPI_TMPDIR:-/tmp}"

# Use anaconda's python (numpy/scipy live there) unless PYTHON is overridden.
PY=${PYTHON:-/opt/anaconda3/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3

"$PY" bucket_reweight.py "$INPUT" \
      --root "$ROOT" \
      --nranks "$NRANKS" \
      --threads "$THREADS" \
      --backend "$BACKEND" \
      $FORCE_FLAG

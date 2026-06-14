# bucket_reweight.py - User Manual

A single, self-contained Python program that reconstructs free-energy surfaces
from **bucket (window) sampling with well-tempered metadynamics (WT-MetaD)**.

It reads **one** control file (`bucketsampling.inp`), runs the complete
per-bucket reweighting pipeline inside each bucket folder, and stitches the
buckets together into a 1D (and optionally 2D) free-energy surface. The program
works for any dimensions, and for selected set of CVs along which WT-MetaD is
applied. Note, as per the original protocol of the bucket sampling, WT-MetaD
is expected along the bucket-CV.

See Ref https://doi.org/10.1021/acs.jctc.4c00776 for details. 

Note that this code is an experimental one, differs from the ones used in the above publication for making the entire postprocessing pipeline simple to execute. The code has only been tested for a few examples. Thus you may use the code with caution. If you find any bugs, report it to nnair@iitk.ac.in (Nisanth N. Nair, IIT Kanpur). 

---

## 1. What it does

For each bucket folder it produces:

| File            | Meaning                                                        |
|-----------------|---------------------------------------------------------------|
| `ct.dat`        | WT-MetaD time-dependent constant *c(t)* (one row per hill)     |
| `vbias.dat`     | Metadynamics bias *V(t)* (one row per MD/COLVAR frame)         |
| `Pu.dat`        | Reweighted 1D probability and `-kT ln P` along the bucket CV   |
| `gradients.dat` | Mean force `dF/ds` on the bucket's inner grid (spline)         |

Then, in the directory where you run it, it writes the stitched results:

| File                   | Meaning                                                    |
|------------------------|------------------------------------------------------------|
| `fes_along_bucket.dat` | 1D free energy `F(s)` along the bucket CV (mean-force int.) |
| `free_energy_2D.dat`   | 2D surface `F(s1, s2)` (only when a `PRINT2D` block exists) |

The pipeline replaces what used to be four separate programs
(`ct_factor.py`, `vbias_mpi.py`, `probability_spline.py`, `reweight_MF_new.py`);
all of their logic is now inlined in `bucket_reweight.py`. Those standalone
files are still present and usable on their own, but are **not** imported.

---

## 2. Requirements

- **Python 3.8+**
- **NumPy** and **SciPy** (required)
- **mpi4py** + an MPI runtime (OpenMPI / MPICH providing `mpirun`) - *optional*,
  only needed to run the `vbias` step across multiple cores/nodes with true MPI.
- **numba** - *optional*, enables a real-OpenMP (`prange`) kernel for `vbias`
  (`--backend numba`). Without it the default NumPy backend is used.

Quick environment check:

```bash
python -c "import numpy, scipy; print('numpy/scipy OK')"
command -v mpirun && python -c "import mpi4py; print('mpi4py OK')"   # for MPI
python -c "import numba; print(numba.__version__)"                   # optional
```

If a package is missing, install it into your environment, e.g.:

```bash
pip install numpy scipy mpi4py numba
```

---

## 3. Directory layout

Run the program from the directory that contains `bucketsampling.inp`. Each
bucket folder named in that file must already contain a `COLVAR` and a `HILLS`
file from the simulation:

```
my_run/
├── bucketsampling.inp
├── W1/
│   ├── COLVAR
│   └── HILLS
├── W2/
│   ├── COLVAR
│   └── HILLS
└── ...
```

Use `--root DIR` if the bucket folders live somewhere other than the current
directory.

---

## 4. Input file: `bucketsampling.inp`

Free-format, one record per line; trailing `# ...` comments and blank lines are
ignored. Fortran-style reals (`1.0d0`, `-3.2D0`) are accepted.

### 4.1 Header (in this exact order)

```
nbucket            # number of buckets (windows)
ncv                # number of CVs printed in COLVAR (columns after time)
iwrap(1:ncv)       # 0/1 per CV: 1 = periodic/torsion (wrap CVs into [-pi,pi]) separated by space
icv_metad(1:ncv)   # 0/1 per CV: 1 = a metadynamics bias acts on this CV separated by space
icv_bucket         # which CV (1-based) the buckets run along
T_phys  T_ext      # physical temp (K), extended/CV temp (K).  
deltaT             # WT-MetaD bias factor
```

### 4.2 One block per bucket (repeated `nbucket` times)

```
folder                         # bucket directory name (relative to --root)
gmin gmax dgrid                # grid for CV 1   ┐
gmin gmax dgrid                # grid for CV 2   │  ncv lines, one per CV
   ...                         #   ...           ┘
t_min  t_max                   # MD-step window for the probability histogram
w_cv   w_hill                  # COLVAR print stride / HILLS deposition stride
```

### 4.3 Optional 2D block (at the very end)

```
PRINT2D
icv2  gridmin2  gridmax2  dgrid2     # 2nd CV (1-based) and its grid for F(s1,s2)
```

Omit the `PRINT2D` block to compute only the 1D profile.

In the above icv2 can be any CV other than the bucket-CV.


### 4.4 Key conventions

- **Number of bins** along any CV: `nbin = round((gmax - gmin)/dgrid) + 1`.
- **COLVAR file** It is iexpected to 1+ncv columns of CV values (for non-temperature accelerated version)
  and extended CV values (for temperature accelerated version)
- **`T_ext` is the reweighting temperature** is the auxiliary variable temperature. 
  It used as `kT` everywhere; `T_phys` is informational only and not used by the code.
- **`t_max < 0`** means "use all MD steps of that bucket" (= number of COLVAR
  rows). `vbias.dat` is always computed for *every* COLVAR frame; the
  `t_min..t_max` window only restricts the computing of probability histogram.
- **Metad subset:** the bias is summed only over CVs with `icv_metad == 1`
  (call it `ncv_mtd`). Therefore:
  - `COLVAR` has `1 + ncv` columns: `time, cv_1 … cv_ncv`.
  - `HILLS` has `1 + 2*ncv_mtd + 1` columns:
    `time, center_1…center_ncv_mtd, sigma_1…sigma_ncv_mtd, height`.
- **Bucket grids must tile the bucket CV:** equal `dgrid`, increasing
  `gmin`/`gmax`, and `gmax` of bucket *i* equals `gmin` of bucket *i+1*.
- **F(s1,s2) computation:** The code is assuming that the following is accurate:
   F(s1,s2) = constant * Integral ds1 ds2 delta(s1-s1^prime) delta(s2-s2^prime)
   even for a case where the number of CVs more than 2. Ideally, it should be computed from
   projecting from P(s1,...sn) to P(s1,s2).
   

### 4.5 Example (2 buckets, 2 CVs, both biased)

```
2                  # nbucket
2                  # ncv
1 1                # iwrap   (both periodic)
1 1                # icv_metad (bias on both CVs)
1                  # icv_bucket (buckets run along CV 1)
300.0 300.0        # T_phys  T_ext
20                 # bias factor
W1
 -1.0 0.0 0.1      # CV1 grid for bucket W1
 -1.0 1.0 0.1      # CV2 grid
 1 -1              # t_min t_max  (t_max<0 -> all MD steps)
 1 6               # w_cv w_hill
W2
 0.0 1.0 0.1       # CV1 grid for bucket W2
 -1.0 1.0 0.1      # CV2 grid
 1 -1
 1 6
PRINT2D
 2 -1.0 1.0 0.1    # 2D surface over CV2
```

> **Metad subset example:** if only CV 1 carries the bias, set `icv_metad` to
> `1 0`. Then `ncv_mtd = 1`, and `HILLS` must have `1 + 2*1 + 1 = 4` columns
> (`time, center, sigma, height`), while `COLVAR` still has all 3 columns
> (`time, cv1, cv2`).

---

## 5. Running it

### 5.1 With the launcher script (recommended)

```bash
cd my_run                       # directory containing bucketsampling.inp
./run_bucket_reweight.sh [NRANKS] [THREADS] [BACKEND] [INPUT] [ROOT] [FORCE]
```

| Arg       | Meaning                                            | Default              |
|-----------|----------------------------------------------------|----------------------|
| `NRANKS`  | MPI ranks for the `vbias` step                     | `2`                  |
| `THREADS` | threads per rank for `vbias`                        | `1`                  |
| `BACKEND` | `numpy` or `numba`                                 | `numpy`              |
| `INPUT`   | control file                                       | `bucketsampling.inp` |
| `ROOT`    | base dir holding the bucket folders                | `.`                  |
| `FORCE`   | `force`/`1`/`yes` → recompute `ct.dat`/`vbias.dat` | *(reuse existing)*   |

The script sets the thread/MPI environment, then runs the driver once. Total
cores used by `vbias` = **`NRANKS × THREADS`**; keep it ≤ your physical cores.

Examples:

```bash
./run_bucket_reweight.sh 4 1 numpy                 # 4 MPI ranks
./run_bucket_reweight.sh 8 1 numpy                 # 8 MPI ranks
./run_bucket_reweight.sh 4 2 numba                 # 4 ranks x 2 threads (numba)
./run_bucket_reweight.sh 1 8 numba                 # no MPI, 8 threads, 1 node
./run_bucket_reweight.sh 4 1 numpy bucketsampling.inp . force   # force recompute
```

Override the Python interpreter with `PYTHON=...`, and the MPI temp dir with
`MPI_TMPDIR=...` (defaults to `/tmp`, which avoids macOS PMIx tmpdir issues).

It is advisable to test the performance on your resources. In some cases that we have tested, 
using 1 thread but multiple MPI ranks seem to perform better than using multiple threads 
and 1 MPI rank.

### 5.2 Calling the driver directly

```bash
python bucket_reweight.py [INPUT] [options]
```

| Option              | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| `INPUT`             | control file (default `bucketsampling.inp`)                   |
| `--root DIR`        | base directory of the bucket folders (default `.`)            |
| `--nranks N`        | MPI ranks for the `vbias` step (default `1`)                  |
| `--threads T`       | threads per rank for `vbias` (default `1`)                    |
| `--backend B`       | `numpy` (default) or `numba`                                  |
| `--force`           | recompute `ct.dat`/`vbias.dat` even if they already exist     |
| `--no-reweight`     | stop after per-bucket files; skip the final stitching         |
| `-v`, `--verbose`   | extra per-grid / per-bucket diagnostics in the stitching step |

Example:

```bash
export OMP_NUM_THREADS=1 TMPDIR=/tmp
python bucket_reweight.py bucketsampling.inp --root . --nranks 4 --backend numpy --force
```

---

## 6. How the parallelism works

The driver is **serial** — it processes one bucket folder at a time. Only the
expensive `vbias` step is parallel. Because a serial process cannot turn itself
into MPI ranks, the driver **re-launches itself under `mpirun`** in a hidden
worker mode, once per bucket:

```
mpirun -np N python bucket_reweight.py <inp> --vbias-worker --bucket-index i ...
```

- You never type `mpirun` yourself, and you never run the driver under `mpirun`.
- Each rank computes its share of MD steps (round-robin for load balance); rank 0
  writes `vbias.dat`.
- If `mpirun`/`mpi4py` are unavailable, or `--nranks 1`, `vbias` is computed
  **in-process** (NumPy, optionally numba threads) automatically.

Two knobs:

- **`--nranks`** → MPI processes (scale across cores/nodes).
- **`--threads`** → shared-memory threads per rank. Real OpenMP threads with
  `--backend numba`; faster vectorized/BLAS math with `--backend numpy`.

You'll see, per bucket, either

```
vbias.dat: COMPUTING (mpirun -np 4 x 1 threads, self-worker) ...   # true MPI
vbias.dat: COMPUTING (single process x 1 threads) ...              # in-process
```

---

## 7. Reusing vs. recomputing

By default, if a bucket folder already contains `ct.dat` or `vbias.dat`, they are
**reused** (you'll see `FOUND existing file -> REUSED`). `Pu.dat` and
`gradients.dat` are always rebuilt from them. To recompute the expensive files,
pass `--force` (or the `FORCE` arg to the launcher); you'll then see `COMPUTED`.

---

## 8. Output file formats

- **`ct.dat`** — `hill_index   c(t)`  (one row per HILLS line) (See Eq. 12 of Ref.)
- **`vbias.dat`** — `md_step   V_bias`  (one row per COLVAR frame; `V` is shifted
  so `V(step 1) = 0`) (See Eq. 6 of Ref.)
- **`Pu.dat`** — `s   P(s)   -kT·ln P(s)`  on the bucket CV grid (See Eq.10 of Ref.)
- **`gradients.dat`** — `s   F(s)   dF/ds`  on the inner grid (See Eq. 9 of Ref.)
  (`nbin-1` points: `gmin … gmax-dgrid`)
- **`fes_along_bucket.dat`** — `s   F(s)`  (1D profile, kJ/mol) (See Eq. 9 of Ref.)
- **`free_energy_2D.dat`** — `s1   s2   F(s1,s2)   P(s2|s1)`  with a blank line
  after each `s1` block (gnuplot `pm3d`/`splot`-friendly) (See Eq. 15, 8 of Ref.)

Energies are in **kJ/mol** (`kB = 8.314472e-3 kJ/mol/K`); verify your simulation
outputs use the same units.

Quick plots (gnuplot):

```gnuplot
plot 'fes_along_bucket.dat' u 1:2 w lp           # 1D
set pm3d map; splot 'free_energy_2D.dat' u 1:2:3 # 2D
```

---

## 9. Typical workflow

```bash
cd my_run
# 1. First run: compute everything with 4 MPI ranks
./run_bucket_reweight.sh 4 1 numpy
# 2. Re-tune the 2D grid / window in bucketsampling.inp, then re-stitch fast
#    (ct.dat & vbias.dat are reused automatically)
./run_bucket_reweight.sh 4 1 numpy
# 3. Changed temperature / bias factor / HILLS? force a clean recompute
./run_bucket_reweight.sh 4 1 numpy bucketsampling.inp . force
```

---

## 10. Troubleshooting

| Message / symptom                                            | Likely cause & fix                                                                 |
|-------------------------------------------------------------|------------------------------------------------------------------------------------|
| `running vbias as a single threaded process` (with `--nranks>1`) | `mpi4py` or `mpirun` not found. Install MPI; ensure `mpirun` is on `PATH`.    |
| `missing COLVAR/HILLS in <folder>`                          | The bucket folder lacks input files, or `--root` points to the wrong place.        |
| `HILLS columns (...) < expected ...`                        | `icv_metad` count vs. HILLS columns mismatch (`HILLS = 1 + 2*ncv_mtd + 1`).        |
| `gradients.dat for bucket k has N rows, expected M`         | Bucket grid changed after `gradients.dat` was written — run with `--force`.         |
| `grid spacing / gridmin / gridmax / tiling` errors          | Bucket grids along the bucket CV must share `dgrid` and tile contiguously.          |
| `t_max (...) > md_steps (...)`                              | `t_max` exceeds the number of COLVAR frames; lower it or use `-1` for "all".        |
| `i_mtd computation` error                                   | `w_cv`/`w_hill` inconsistent with HILLS length; check the deposition stride.        |
| PMIx / tmpdir error from `mpirun` (macOS)                   | Set `MPI_TMPDIR=/tmp` (the launcher already does this).                             |

For more diagnostics during the stitching step, add `-v`.

---

## 11. Files 

| File                       | Role                                                       |
|----------------------------|------------------------------------------------------------|
| `bucket_reweight.py`       | **Main self-contained driver** (run this)                  |
| `run_bucket_reweight.sh`   | Launcher: sets thread/MPI env and runs the driver          |
| `bucketsampling.inp`       | Control input file (you provide/edit this)                 |


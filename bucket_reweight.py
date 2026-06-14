#!/usr/bin/env python3
"""
Self-contained driver for bucket-sampling reweighting and free energy surface computation.

This program reads ONE control file (`bucketsampling.inp`) and runs the whole
per-bucket reweighting pipeline.  

This will compute
1. c(t) factor                       ->  ct.dat                   (serial, NumPy)
2. V^bias(s,t) bias potential        ->  vbias.dat                (PARALLEL: MPI + threads)
3. 1D probability along bucket CV + 
4. gradient dF(s1)/ds1 along 
   the bucket CV                     ->  Pu.dat, gradients.dat (serial, NumPy + SciPy)
5. Compute F(s1,s2) by mean-force 
   based stitching                   ->  fes_along_bucket.dat, free_energy_2D.dat (serial)

The per-bucket calculations run *inside each bucket folder* (the folder names in
bucketsampling.inp).  Buckets are processed one at a time (serial driver); only
the computationally expensive vbias step is parallelised.

Implementation Note: How the parallel vbias works 
-------------------------------------------------
A plain serial Python process cannot turn *itself* into N MPI ranks.  To keep
true MPI inside one file, the driver re-launches THIS file under mpirun in a
hidden worker mode, once per bucket:

    mpirun -np N python bucket_reweight.py <inp> --vbias-worker --bucket-index i ...

Each of the N ranks imports mpi4py, computes its share of the MD steps for that
one bucket, and rank 0 writes vbias.dat.  When mpi4py / mpirun are unavailable
(or --nranks 1), vbias is computed in-process instead (NumPy, optionally numba
threads).  The driver process never imports mpi4py, which avoids nested-MPI
issues when it spawns mpirun.

Other convention used:
----------------------
* Uses kJ/mol as the energy unit.
* t_max < 0  ->  "use all MD steps of that bucket" (= COLVAR line count).
* vbias.dat is produced for ALL md_steps (the stitching needs one vbias value
  per COLVAR frame); the t_min/t_max window only restricts the probability.
* gradients.dat holds the bucket's INNER grid (ngrids-1 points,
  gmin .. gmax-dgrid) with the spline derivative dF/ds in column 3.
* Spline interpolation uses 4-th order. Thus, at least 3 grid points needed
  along the bucket coordinate.

Usage:
    python bucket_reweight.py [bucketsampling.inp] [--root DIR]
                              [--nranks N] [--threads T] [--backend numpy|numba]
                              [--force] [--no-reweight] [-v]

Each bucket folder must already contain COLVAR and HILLS.  Existing ct.dat /
vbias.dat are REUSED unless --force is given.

(Nisanth N. Nair/nnair@iitk.ac.in; Jun 12, 2026)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

KB = 8.314472e-3        # kJ/(mol*K)
SPLINE_K = 4            # fes spline order
MIN_PROB = 1.0e-32      # probability floor inside the log
GRID_ERR = 1.0e-4       # grid-consistency tolerance for checks within the program
TWOPI = 2.0 * np.pi

SCRIPT_PATH = Path(__file__).resolve()   # used to re-exec self for the vbias worker


# =========================================================================== #
#  Generic helpers
# =========================================================================== #
def _to_float(token: str) -> float:
    """Parse a Fortran-style real such as '-3.2d0' or '1.0E-3'."""
    return float(token.replace("d", "e").replace("D", "e"))


def _nint(x):
    """Fortran NINT: round half away from zero (numpy rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def count_lines(path: Path) -> int:
    """Count data lines, skipping blank lines and '#' comments/headers."""
    n = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return n


def load_table(path: Path, label: str, min_cols: int) -> np.ndarray:
    """Load a whitespace-separated numeric file, skipping '#'/blank lines."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"!!ERROR: file not found while reading {label}: {path}")
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            fields = s.split()
            if len(fields) < min_cols:
                sys.exit(
                    f"!!ERROR reading {label} ({path}) line {lineno}: "
                    f"expected at least {min_cols} columns, got {len(fields)}: {s!r}"
                )
            try:
                rows.append([_to_float(x) for x in fields])
            except ValueError:
                sys.exit(
                    f"!!ERROR reading {label} ({path}) line {lineno}: "
                    f"cannot parse as numbers: {s!r}"
                )
    if not rows:
        sys.exit(f"!!ERROR: no data found in {label} ({path})")
    width = max(len(r) for r in rows)
    if any(len(r) != width for r in rows):
        rows = [r + [np.nan] * (width - len(r)) for r in rows]
    return np.asarray(rows, dtype=np.float64)


class InputReader:
    """Line-oriented reader.

    Each logical READ consumes one record (line); trailing '#...' comments and
    extra tokens are ignored, and blank lines are skipped.
    """

    def __init__(self, text: str):
        self._lines = text.splitlines()
        self._i = 0

    def _advance(self) -> str | None:
        while self._i < len(self._lines):
            raw = self._lines[self._i]
            self._i += 1
            if raw.strip() == "":
                continue
            return raw
        return None

    def tokens(self, label: str) -> list[str]:
        raw = self._advance()
        if raw is None:
            sys.exit(f"!!ERROR: unexpected end of input while reading {label}")
        return raw.split("#", 1)[0].split()

    def raw_line(self) -> str | None:
        return self._advance()

    def one_int(self, label: str) -> int:
        return int(self.tokens(label)[0])

    def ints(self, n: int, label: str) -> list[int]:
        tok = self.tokens(label)
        if len(tok) < n:
            sys.exit(f"!!ERROR: expected {n} integers for {label}, got {tok}")
        return [int(t) for t in tok[:n]]

    def floats(self, n: int, label: str) -> list[float]:
        tok = self.tokens(label)
        if len(tok) < n:
            sys.exit(f"!!ERROR: expected {n} reals for {label}, got {tok}")
        return [_to_float(t) for t in tok[:n]]


def parse_bucketsampling(text: str) -> dict:
    """Read bucketsampling.inp -> dict of parameters + per-bucket entries."""
    rdr = InputReader(text)
    nbucket = rdr.one_int("nbucket")
    ncv = rdr.one_int("ncv")
    iwrap = rdr.ints(ncv, "iwrap")
    icv_metad = rdr.ints(ncv, "icv_metad")
    icv_bucket = rdr.one_int("icv_bucket")
    t_phys, t_ext = rdr.floats(2, "T_phys T_ext")
    bias_fact = rdr.floats(1, "bias factor")[0]

    buckets = []
    for ib in range(nbucket):
        folder = rdr.tokens(f"bucket folder #{ib + 1}")[0]
        grids = []
        for icv in range(ncv):
            gmin, gmax, dg = rdr.floats(3, f"grid (bucket {ib + 1}, cv {icv + 1})")
            grids.append((gmin, gmax, dg))
        t_min, t_max = rdr.ints(2, f"t_min t_max (bucket {ib + 1})")
        w_cv, w_hill = rdr.ints(2, f"w_cv w_hill (bucket {ib + 1})")
        buckets.append(dict(folder=folder, grids=grids, t_min=t_min, t_max=t_max,
                            w_cv=w_cv, w_hill=w_hill))

    return dict(nbucket=nbucket, ncv=ncv, iwrap=iwrap, icv_metad=icv_metad,
                icv_bucket=icv_bucket, t_phys=t_phys, t_ext=t_ext,
                bias_fact=bias_fact, buckets=buckets)


def nbins_from_grid(gmin: float, gmax: float, dgrid: float) -> int:
    """Grid spacing -> number of bins (matches the stitching's ngrids)."""
    return int(round((gmax - gmin) / dgrid)) + 1


def metad_indices(cfg: dict) -> list[int]:
    """0-based indices of the CVs that carry a metadynamics bias (icv_metad==1)."""
    idx = [i for i, m in enumerate(cfg["icv_metad"]) if m == 1]
    if not idx:
        sys.exit("!!ERROR: at least one CV must have icv_metad==1")
    return idx


# =========================================================================== #
#  Step 1: c(t) factor  ->  ct.dat   (inlined ct_factor kernels, serial NumPy)
# =========================================================================== #
def ct_read_hills(path: Path, mtd_steps: int, ndim: int):
    """HILLS -> (hill, width, ht) over `ndim` metad CVs (1+2*ndim+1 columns)."""
    nd = int(ndim)
    cols = 1 + 2 * nd + 1
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] != mtd_steps:
        raise ValueError(f"HILLS rows ({data.shape[0]}) != mtd_steps ({mtd_steps})")
    if data.shape[1] < cols:
        raise ValueError(f"HILLS columns ({data.shape[1]}) < expected {cols}")
    hill = data[:, 1:1 + nd].T.copy()
    width = data[:, 1 + nd:1 + 2 * nd].T.copy()
    ht = data[:, 1 + 2 * nd].copy()
    return hill, width, ht


def build_grid_coords(gridmin, gridmax, nbin):
    """Coordinates of all grid bins, shape (n_total, ndim)."""
    axes = [np.linspace(gridmin[d], gridmax[d], int(nbin[d]), dtype=np.float64)
            for d in range(gridmin.size)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def wrap_diff(diff, mask, twopi=TWOPI):
    """diff - twopi*nint(diff/twopi) applied only to periodic dims."""
    wrapped = diff - twopi * np.rint(diff / twopi)
    return np.where(mask, wrapped, diff)


def compute_ct(hill, width, ht, coords, kt_energy, gamma_, iwrap, twopi=TWOPI):
    """c(t) factor for WT-MetaD (arbitrary metad-CV dimension)."""
    _, mtd_steps = hill.shape
    n_total = coords.shape[0]
    mask = np.asarray(iwrap).reshape(-1).astype(bool)
    any_periodic = bool(mask.any())

    fes = np.zeros(n_total, dtype=np.float64)
    ct = np.empty(mtd_steps, dtype=np.float64)
    for i in range(mtd_steps):
        diff = coords - hill[:, i]
        if any_periodic:
            diff = wrap_diff(diff, mask, twopi)
        w2 = width[:, i] * width[:, i]
        diff2_sum = 0.5 * np.sum((diff * diff) / w2, axis=1)
        fes -= float(ht[i]) * np.exp(-diff2_sum)
        num = float(np.sum(np.exp(-fes / kt_energy)))
        den = float(np.sum(np.exp(-fes / kt_energy + fes * gamma_ / kt_energy)))
        if den <= 0.0 or num <= 0.0:
            raise RuntimeError(f"non-positive accumulator at MTD step {i+1}")
        ct[i] = kt_energy * np.log(num / den)
    return ct


def run_ct_factor(folder: Path, cfg: dict, bucket: dict,
                  md_steps: int, mtd_steps: int) -> None:
    """Compute ct.dat in `folder` (grid restricted to the metad CVs)."""
    kt_energy = KB * cfg["t_ext"]
    gamma = (cfg["bias_fact"] - 1.0) / cfg["bias_fact"]

#get the array with CV indices on which the metadynamics bias is acting.  midx=[0,2] means, 1st and 3rd CVs are metadynamics active. 
    midx = metad_indices(cfg) 
    ncv_mtd = len(midx)
    gmin = np.array([bucket["grids"][i][0] for i in midx], dtype=np.float64)
    gmax = np.array([bucket["grids"][i][1] for i in midx], dtype=np.float64)
    nbin = np.array([nbins_from_grid(*bucket["grids"][i]) for i in midx], dtype=np.int64)
    iwrap = np.array([cfg["iwrap"][i] for i in midx], dtype=np.int64)

    hill, width, ht = ct_read_hills(folder / "HILLS", mtd_steps, ncv_mtd)
    coords = build_grid_coords(gmin, gmax, nbin)
    ct = compute_ct(hill, width, ht, coords, kt_energy, gamma, iwrap)

    np.savetxt(folder / "ct.dat",
               np.column_stack([np.arange(1, mtd_steps + 1, dtype=np.int64), ct]),
               fmt="%10d %16.8f")
    print(f"    ct.dat   : COMPUTED  (md_steps={md_steps}, mtd_steps={mtd_steps})")


# =========================================================================== #
#  Step 2: bias potential  ->  vbias.dat   (inlined vbias kernels; MPI worker)
# =========================================================================== #
def vbias_load_colvar(path: Path, metad_idx):
    """COLVAR -> cv (md_steps, ncv_mtd): only the metad-CV columns (col d+1)."""
    cols = tuple(int(d) + 1 for d in metad_idx)
    cv = np.loadtxt(path, comments="#", usecols=cols, ndmin=2)
    return np.ascontiguousarray(cv, dtype=np.float64)


def vbias_load_hills(path: Path, ncv_mtd: int):
    """HILLS -> (centers, widths, heights) over ncv_mtd metad CVs."""
    c_cols = tuple(range(1, ncv_mtd + 1))
    s_cols = tuple(range(ncv_mtd + 1, 2 * ncv_mtd + 1))
    h_col = (2 * ncv_mtd + 1,)
    centers = np.loadtxt(path, comments="#", usecols=c_cols, ndmin=2)
    widths = np.loadtxt(path, comments="#", usecols=s_cols, ndmin=2)
    heights = np.loadtxt(path, comments="#", usecols=h_col, ndmin=2).ravel()
    return (np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(widths, dtype=np.float64),
            np.ascontiguousarray(heights, dtype=np.float64))


def vbias_steps_numpy(steps, cv, centers, inv_sig2, ht_gamma, iwrap, w_cv, w_hill):
    """Vectorised NumPy kernel: V_bias for each MD step in `steps` (1-based)."""
    out = np.empty(steps.size, dtype=np.float64)
    wrap_mask = iwrap.astype(bool)
    for k in range(steps.size):
        i_md = steps[k]
        mtd_max = (i_md * w_cv) // w_hill
        if mtd_max <= 0:
            out[k] = 0.0
            continue
        diff = cv[i_md - 1] - centers[:mtd_max]
        if wrap_mask.any():
            d = diff[:, wrap_mask]
            diff[:, wrap_mask] = d - TWOPI * np.round(d / TWOPI)
        arg = 0.5 * np.einsum("md,md->m", diff * diff, inv_sig2[:mtd_max])
        out[k] = np.dot(ht_gamma[:mtd_max], np.exp(-arg))
    return out


def _build_numba_kernel():
    """Compile the threaded (OpenMP) numba kernel lazily; None if unavailable."""
    try:
        from numba import njit, prange
    except Exception:
        return None

    @njit(parallel=True, fastmath=True, cache=True)
    def _kernel(steps, cv, centers, inv_sig2, ht_gamma, iwrap, w_cv, w_hill):
        n = steps.shape[0]
        ncv = cv.shape[1]
        out = np.zeros(n, dtype=np.float64)
        for k in prange(n):
            i_md = steps[k]
            mtd_max = (i_md * w_cv) // w_hill
            acc = 0.0
            for m in range(mtd_max):
                arg = 0.0
                for d in range(ncv):
                    diff = cv[i_md - 1, d] - centers[m, d]
                    if iwrap[d] == 1:
                        diff -= TWOPI * np.round(diff / TWOPI)
                    arg += 0.5 * diff * diff * inv_sig2[m, d]
                acc += ht_gamma[m] * np.exp(-arg)
            out[k] = acc
        return out

    return _kernel


def compute_vbias_bucket(folder: Path, cfg: dict, bucket: dict, md_steps: int,
                         threads: int, backend: str, comm=None) -> None:
    """Compute vbias.dat for one bucket; serial (comm=None) or MPI (comm given).

    vbias is evaluated for ALL md_steps; rank 0 writes vbias.dat.
    """
    rank, nranks = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    parent = (rank == 0)

    midx = np.array(metad_indices(cfg), dtype=np.int64)
    ncv_mtd = midx.size
    gamma = (cfg["bias_fact"] - 1.0) / cfg["bias_fact"]
    iwrap = np.ascontiguousarray(np.array([cfg["iwrap"][i] for i in midx], dtype=np.int64))
    w_cv, w_hill = bucket["w_cv"], bucket["w_hill"]
    t_max = md_steps                              # vbias for every COLVAR frame

    # Load on rank 0, then broadcast (no-op when serial).
    if parent:
        cv = vbias_load_colvar(folder / "COLVAR", midx)
        centers, widths, heights = vbias_load_hills(folder / "HILLS", ncv_mtd)
    else:
        cv = centers = widths = heights = None
    if comm is not None:
        cv = comm.bcast(cv, root=0)
        centers = comm.bcast(centers, root=0)
        widths = comm.bcast(widths, root=0)
        heights = comm.bcast(heights, root=0)

    inv_sig2 = np.ascontiguousarray(1.0 / (widths * widths))
    ht_gamma = np.ascontiguousarray(heights * gamma)

    all_steps = np.arange(1, t_max + 1, dtype=np.int64)
    my_steps = all_steps[rank::nranks]            # round-robin load balance

    kernel = _build_numba_kernel() if backend == "numba" else None
    if kernel is not None:
        if threads > 0:
            try:
                import numba
                numba.set_num_threads(threads)
            except Exception:
                pass
        my_vbias = kernel(my_steps, cv, centers, inv_sig2, ht_gamma, iwrap, w_cv, w_hill)
    else:
        my_vbias = vbias_steps_numpy(my_steps, cv, centers, inv_sig2, ht_gamma,
                                     iwrap, w_cv, w_hill)

    if comm is not None and nranks > 1:
        gathered_steps = comm.gather(my_steps, root=0)
        gathered_vbias = comm.gather(my_vbias, root=0)
    else:
        gathered_steps, gathered_vbias = [my_steps], [my_vbias]

    if parent:
        vbias = np.empty(t_max, dtype=np.float64)
        for s, v in zip(gathered_steps, gathered_vbias):
            vbias[s - 1] = v
        ref = vbias[0]                            # subtract vbias(1), as in Fortran
        with (folder / "vbias.dat").open("w") as fh:
            for i_md in range(1, t_max + 1):
                fh.write(f"{i_md:10d}{vbias[i_md - 1] - ref:15.6f}\n")


def run_vbias(folder: Path, cfg: dict, bucket: dict, inp_path: Path, root: Path,
              ib_index: int, md_steps: int, nranks: int, threads: int,
              backend: str, use_mpi: bool, env: dict) -> None:
    """Compute vbias.dat: true MPI via a self re-exec, else in-process."""
    if use_mpi and nranks > 1:
        cmd = ["mpirun",
               "--mca", "orte_tmpdir_base", env.get("TMPDIR", "/tmp"),
               "--mca", "pmix_server_tmpdir", env.get("TMPDIR", "/tmp"),
               "-np", str(nranks),
               sys.executable, str(SCRIPT_PATH), str(inp_path),
               "--vbias-worker", "--bucket-index", str(ib_index),
               "--root", str(root), "--threads", str(threads),
               "--backend", backend]
        print(f"    vbias.dat: COMPUTING (mpirun -np {nranks} x {threads} threads, "
              f"self-worker) ...")
        subprocess.run(cmd, env=env, check=True)
    else:
        print(f"    vbias.dat: COMPUTING (single process x {threads} threads) ...")
        compute_vbias_bucket(folder, cfg, bucket, md_steps, threads, backend, comm=None)
    print("    vbias.dat: COMPUTED")


def vbias_worker(args) -> int:
    """Hidden mode: one MPI rank computing vbias.dat for a single bucket."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except Exception:
        comm = None                               # launched without MPI -> single rank

    cfg = parse_bucketsampling(Path(args.input).read_text())
    bucket = cfg["buckets"][args.bucket_index]
    folder = Path(args.root) / bucket["folder"]
    md_steps = count_lines(folder / "COLVAR")
    compute_vbias_bucket(folder, cfg, bucket, md_steps, args.threads, args.backend,
                         comm=comm)
    return 0


# =========================================================================== #
#  Step 3: probability / free energy  ->  Pu.dat + gradients.dat   (serial)
# =========================================================================== #
def run_probability(folder: Path, cfg: dict, bucket: dict,
                    md_steps: int, mtd_steps: int) -> None:
    """Reweighted 1D P(s) along the bucket CV + spline gradients.dat."""
    kt = KB * cfg["t_ext"]
    icvb = cfg["icv_bucket"]                       # 1-based COLVAR column (time=col0)
    gmin_b, gmax_b, dgrid_b = bucket["grids"][icvb - 1]
    nbin_b = nbins_from_grid(gmin_b, gmax_b, dgrid_b)
    griddiff = (gmax_b - gmin_b) / float(nbin_b - 1)

    t_min = bucket["t_min"]
    t_max = md_steps if bucket["t_max"] < 0 else bucket["t_max"]
    if t_max > md_steps:
        sys.exit(f"!!ERROR: t_max ({t_max}) > md_steps ({md_steps}) in {folder}")

    colvar = np.loadtxt(folder / "COLVAR", comments="#", ndmin=2)[:md_steps]
    cv1 = colvar[:, icvb]
    vbias = np.loadtxt(folder / "vbias.dat", ndmin=2)[:, 1]
    ct = np.loadtxt(folder / "ct.dat", ndmin=2)[:, 1]

    i_md = np.arange(t_min, t_max + 1)
    index1 = np.rint((cv1[i_md - 1] - gmin_b) / griddiff).astype(int) + 1
    valid = (index1 >= 1) & (index1 <= nbin_b)

    i_mtd = (i_md * bucket["w_cv"]) // bucket["w_hill"]
    i_mtd = np.clip(i_mtd, 1, mtd_steps)

    weight = np.exp((vbias[i_md - 1] - ct[i_mtd - 1]) / kt)

    prob = np.zeros(nbin_b, dtype=np.float64)
    np.add.at(prob, index1[valid] - 1, weight[valid])
    den = float(weight[valid].sum())
    prob *= 1.0 / (den * griddiff)

    s1 = gmin_b + np.arange(nbin_b) * griddiff
    fes = -kt * np.log(np.maximum(MIN_PROB, prob))

    with (folder / "Pu.dat").open("w") as fh:
        for si, pi, fi in zip(s1, prob, fes):
            fh.write(f" {si:.10g} {pi:.10g} {fi:.10g}\n")

    if nbin_b <= SPLINE_K:
        sys.exit(f"!!ERROR: need > {SPLINE_K} grid points for the spline in {folder}")
    spl = UnivariateSpline(s1, fes, s=0, k=SPLINE_K)
    der = spl.derivative(1)
    x_inner = gmin_b + dgrid_b * np.arange(nbin_b - 1)   # gmin .. gmax-dgrid
    with (folder / "gradients.dat").open("w") as fh:     # s, fes(s), dF/ds
        for xi in x_inner:
            fh.write("%.6f %.6f %.6f \n" % (xi, float(spl(xi)), float(der(xi))))

    print(f"    Pu.dat + gradients.dat written  (window {t_min}..{t_max}, "
          f"{nbin_b - 1} inner grid points)")


# =========================================================================== #
#  Final stitching: mean-force integration (inlined reweight_MF_new logic)
# =========================================================================== #
def run_reweight(inp_path: Path, root: Path, verbose: bool) -> int:
    """1D mean-force integration + optional 2D free-energy surface."""
    rdr = InputReader(Path(inp_path).read_text())

    nbucket = rdr.one_int("nbucket")
    ncv = rdr.one_int("ncv")
    rdr.ints(ncv, "iwrap")                         # read, unused here
    rdr.ints(ncv, "icv_metad")                     # read, unused here
    icv_bucket = rdr.one_int("icv_bucket")
    _kt0, kt = rdr.floats(2, "kt0 kt")
    rdr.floats(1, "deltaT")                        # read, unused here

    print(f" (Info) nbucket=       {nbucket}")
    print(f" (Info) ncv=           {ncv}")
    print(f" (Info) icv_bucket=    {icv_bucket}")
    print(f" (Info) kt=            {kt}")

    if not (1 <= icv_bucket <= ncv):
        sys.exit("!!ERROR: icv_bucket out of range 1..ncv")

    bucket_folder: list[str] = []
    gridmin = np.zeros((nbucket, ncv))
    gridmax = np.zeros((nbucket, ncv))
    dgrids = np.zeros((nbucket, ncv))
    ngrids = np.zeros((nbucket, ncv), dtype=int)
    t_min = np.zeros(nbucket, dtype=int)
    t_max = np.zeros(nbucket, dtype=int)
    w_cv = np.zeros(nbucket, dtype=int)
    w_hill = np.zeros(nbucket, dtype=int)

    for ib in range(nbucket):
        bucket_folder.append(rdr.tokens(f"bucket folder #{ib + 1}")[0])
        for icv in range(ncv):
            gmin, gmax, dg = rdr.floats(3, f"grid (bucket {ib + 1}, cv {icv + 1})")
            gridmin[ib, icv], gridmax[ib, icv], dgrids[ib, icv] = gmin, gmax, dg
            ngrids[ib, icv] = int(round((gmax - gmin) / dg)) + 1
            print(f" grids info {ib + 1:8d}{icv + 1:8d}"
                  f"{gmin:16.4f}{gmax:16.4f}{dg:16.4f}{ngrids[ib, icv]:8d}")
        t_min[ib], t_max[ib] = rdr.ints(2, f"t_min t_max (bucket {ib + 1})")
        w_cv[ib], w_hill[ib] = rdr.ints(2, f"w_cv w_hill (bucket {ib + 1})")

    icvb = icv_bucket - 1

    for ib in range(nbucket - 1):
        if abs(dgrids[ib, icvb] - dgrids[ib + 1, icvb]) > GRID_ERR:
            sys.exit("!!ERROR: grid spacing along the bucket CV must be the same")
        if gridmin[ib, icvb] > gridmin[ib + 1, icvb]:
            sys.exit("!!ERROR: gridmin values must increase along the buckets")
        if gridmax[ib, icvb] > gridmax[ib + 1, icvb]:
            sys.exit("!!ERROR: gridmax values must increase along the buckets")
        if abs(gridmax[ib, icvb] - gridmin[ib + 1, icvb]) > GRID_ERR:
            sys.exit("!!ERROR: gridmax of bucket i must equal gridmin of bucket i+1")

    print_2d = False
    icv2 = gridmin2 = gridmax2 = dgrid2 = None
    marker = rdr.raw_line()
    if marker is not None and "PRINT2D" in marker:
        print_2d = True
        tok = rdr.tokens("PRINT2D grid (icv2 gridmin2 gridmax2 dgrid2)")
        if len(tok) < 4:
            sys.exit("!!ERROR reading the 2nd-dimension grid for the 2D free energy")
        icv2 = int(tok[0])
        gridmin2, gridmax2, dgrid2 = (_to_float(t) for t in tok[1:4])
        print(f" {icv2} {gridmin2} {gridmax2} {dgrid2}")
        if gridmax2 <= gridmin2:
            sys.exit("!!ERROR: gridmax2 must be greater than gridmin2")
        if icv2 == icv_bucket:
            sys.exit("!!ERROR: the 2nd CV cannot be the bucket CV")
        if not (1 <= icv2 <= ncv):
            sys.exit("!!ERROR: icv2 out of range 1..ncv")
    else:
        print(" Warning! Free Energy 2D will not be printed")

    kt = KB * kt

    ngrid1 = int(sum(ngrids[ib, icvb] - 1 for ib in range(nbucket)))
    print(f" (Info) Number of grids along bucket CV = {ngrid1}")
    print(" (Info) Taking units of energy as kJ/mol (check simulation outputs!)")

    gridmin1 = gridmin[0, icvb]
    gridmax1 = gridmax[nbucket - 1, icvb]          
    dgrid1 = dgrids[0, icvb]

    ngrid2 = None
    if print_2d:
        ngrid2 = int(round((gridmax2 - gridmin2) / dgrid2)) + 1
        print(f" ngrid2= {ngrid2}")

    if verbose:
        print(" Grid (info) along the bucket CV")
        idx = 0
        for ib in range(nbucket):
            for ibin in range(1, ngrids[ib, icvb]):
                idx += 1
                s = gridmin[ib, icvb] + (ibin - 1) * dgrids[ib, icvb]
                print(f" bin index= {idx}  ibucket= {ib + 1}"
                      f"  bucket bin index= {ibin}  grid= {s}")

    # ----- 1D: F(bucket CV) by trapezoidal integration of dF/ds -------------- #
    dfds = np.empty(ngrid1)
    idx = 0
    for ib in range(nbucket):
        n_inner = ngrids[ib, icvb] - 1
        grad = load_table(root / bucket_folder[ib] / "gradients.dat",
                          f"gradients.dat (bucket {ib + 1})", min_cols=3)
        if grad.shape[0] != n_inner:
            sys.exit(f"!!ERROR: gradients.dat for bucket {ib + 1} has {grad.shape[0]} "
                     f"rows, expected {n_inner}")
        for ibin in range(1, n_inner + 1):
            s_expected = gridmin[ib, icvb] + (ibin - 1) * dgrids[ib, icvb]
            if verbose:
                print(f" (Dbg-Info) Reading MF of bucket {ib + 1} for grid = {s_expected}")
            if abs(grad[ibin - 1, 0] - s_expected) > GRID_ERR:
                sys.exit(f"!!ERROR: grid mismatch in gradients.dat (bucket {ib + 1}, "
                         f"row {ibin}): file s={grad[ibin - 1, 0]}, expected {s_expected}")
            dfds[idx] = grad[ibin - 1, 2]
            idx += 1

    fes1 = np.zeros(ngrid1)
    increments = 0.5 * dgrid1 * (dfds[:-1] + dfds[1:])
    fes1[:-1] = np.cumsum(increments)[: ngrid1 - 1]

    s1_grid = gridmin1 + np.arange(ngrid1) * dgrid1

    print(" (Info) Grid points along the bucket CV (trapezoidal mean-force integration):")
    for ibin in range(ngrid1 - 1):
        print(f"   bin {ibin + 1:6d}   s = {s1_grid[ibin]:16.8f}   F = {fes1[ibin]:16.8f}")

    with (Path.cwd() / "fes_along_bucket.dat").open("w") as fh:
        for ibin in range(ngrid1 - 1):
            fh.write(f" {s1_grid[ibin]:.10g}   {fes1[ibin]:.10g}\n")
    print(" (Info) Free energy profile along bucket CV -> fes_along_bucket.dat")

    if not print_2d:
        print(" Exiting without computing 2D free energy surface....")
        return 0

    print(" Computing 2D free energy surface....")

    # ----- 2D: F(bucket CV, CV2) by WT-MetaD reweighting --------------------- #
    icol_b = icv_bucket
    icol_2 = icv2
    prob = np.zeros((ngrid1, ngrid2))
    norm = np.zeros(ngrid1)

    for ib in range(nbucket):
        folder = root / bucket_folder[ib]
        colvar = load_table(folder / "COLVAR", f"COLVAR (bucket {ib + 1})",
                            min_cols=ncv + 1)
        ct = load_table(folder / "ct.dat", f"ct.dat (bucket {ib + 1})", min_cols=2)[:, 1]
        vbias = load_table(folder / "vbias.dat", f"vbias.dat (bucket {ib + 1})",
                           min_cols=2)[:, 1]

        md_steps = colvar.shape[0]
        mtd_steps = ct.shape[0]
        if vbias.shape[0] != md_steps:
            sys.exit(f"!!ERROR: md_steps mismatch in bucket {ib + 1}: "
                     f"vbias.dat has {vbias.shape[0]}, COLVAR has {md_steps}")

        tmax_eff = md_steps if t_max[ib] < 0 else t_max[ib]
        i_md = np.arange(1, md_steps + 1)
        cv1 = colvar[:, icol_b]
        cv2 = colvar[:, icol_2]

        ibin = (_nint((cv1 - gridmin1) / dgrid1) + 1).astype(int)
        jbin = (_nint((cv2 - gridmin2) / dgrid2) + 1).astype(int)

        ok_ibin = (ibin >= 1) & (ibin <= ngrid1 - 1)
        ok_jbin = (jbin >= 1) & (jbin <= ngrid2 - 1)
        ok_bucket = (cv1 > gridmin[ib, icvb]) & (cv1 <= gridmax[ib, icvb])
        ok_time = (i_md >= t_min[ib]) & (i_md <= tmax_eff)
        mask = ok_ibin & ok_jbin & ok_bucket & ok_time

        i_mtd = (i_md * w_cv[ib]) // w_hill[ib]
#bug
#        bad = mask & ~((i_mtd >= 1) & (i_mtd <= mtd_steps - 1))
#        if np.any(bad):
#            sys.exit("!!ERROR in i_mtd computation - something is wrong! check inputs "
#                     f"(bucket {ib + 1})")

        if np.any(mask):
            sel = np.where(mask)[0]
            weight = np.exp((vbias[sel] - ct[i_mtd[sel] - 1]) / kt)
            np.add.at(prob, (ibin[sel] - 1, jbin[sel] - 1), weight)
            np.add.at(norm, ibin[sel] - 1, weight)

    print(" (Info) probability is computed.")

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_norm = 1.0 / (norm * dgrid1 * dgrid2)
    inv_norm[~np.isfinite(inv_norm)] = 0.0

    s2_grid = gridmin2 + np.arange(ngrid2) * dgrid2
    with (Path.cwd() / "free_energy_2D.dat").open("w") as fh:
        for ibin in range(ngrid1):
            for jbin in range(ngrid2):
                p_cond = prob[ibin, jbin] * inv_norm[ibin]
                fe = -kt * np.log(max(p_cond, MIN_PROB)) + fes1[ibin]
                fh.write(f"{s1_grid[ibin]:16.8E}{s2_grid[jbin]:16.8E}"
                         f"{fe:16.8E}{p_cond:16.8E}\n")
            # one blank line after each inner (jbin) block (gnuplot-friendly).
            fh.write("\n")

    print(" (Info) Free-energy and unbiased distribution -> free_energy_2D.dat (kJ/mol)")
    return 0


# =========================================================================== #
#  Main driver
# =========================================================================== #
def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="bucketsampling.inp",
                        help="control input file (default: bucketsampling.inp)")
    parser.add_argument("--root", default=".",
                        help="base directory the bucket folders are under (default: cwd)")
    parser.add_argument("--nranks", type=int, default=1,
                        help="MPI ranks for the vbias step (default: 1)")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads per rank for vbias (default: 1)")
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy",
                        help="vbias shared-memory kernel (default: numpy)")
    parser.add_argument("--force", action="store_true",
                        help="recompute ct.dat / vbias.dat even if they already exist")
    parser.add_argument("--no-reweight", action="store_true",
                        help="stop after per-bucket ct/vbias/gradients (skip stitching)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output from the final reweight step")
    # Hidden self re-exec mode for the parallel vbias step (launched via mpirun).
    parser.add_argument("--vbias-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bucket-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Hidden worker mode: one MPI rank computing a single bucket's vbias.dat.
    if args.vbias_worker:
        return vbias_worker(args)

    inp_path = Path(args.input)
    if not inp_path.is_file():
        sys.exit(f"!!ERROR: control file not found: {inp_path}")
    root = Path(args.root)

    cfg = parse_bucketsampling(inp_path.read_text())
    print("=" * 70)
    print(f"bucket_reweight: {cfg['nbucket']} buckets, ncv={cfg['ncv']}, "
          f"bucket CV={cfg['icv_bucket']}")
    print(f"  T_phys={cfg['t_phys']}  T_ext={cfg['t_ext']}  bias_fact={cfg['bias_fact']}")
    print(f"  iwrap={cfg['iwrap']}  icv_metad={cfg['icv_metad']}")

    # Is true MPI available for the parallel vbias step?  (Driver never imports
    # mpi4py itself; it only checks availability before spawning mpirun.)
    have_mpi4py = importlib.util.find_spec("mpi4py") is not None
    have_mpirun = shutil.which("mpirun") is not None
    use_mpi = args.nranks > 1 and have_mpi4py and have_mpirun
    if args.nranks > 1 and not use_mpi:
        print(f"  [warn] --nranks {args.nranks} requested but "
              f"{'mpi4py' if not have_mpi4py else 'mpirun'} unavailable -> "
              f"running vbias as a single threaded process.")

    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(args.threads)
    env.setdefault("TMPDIR", env.get("MPI_TMPDIR", "/tmp"))

    t0 = time.time()
    for ib, bucket in enumerate(cfg["buckets"], start=1):
        folder = root / bucket["folder"]
        print("-" * 70)
        print(f"[bucket {ib}/{cfg['nbucket']}] {folder}")
        if not folder.is_dir():
            sys.exit(f"!!ERROR: bucket folder not found: {folder}")
        for req in ("COLVAR", "HILLS"):
            if not (folder / req).is_file():
                sys.exit(f"!!ERROR: missing {req} in {folder}")

        md_steps = count_lines(folder / "COLVAR")
        mtd_steps = count_lines(folder / "HILLS")

        # Step 1: c(t)  (reuse existing ct.dat unless --force).
        if (folder / "ct.dat").is_file() and not args.force:
            print("    ct.dat   : FOUND existing file -> REUSED (not recomputed)")
        else:
            run_ct_factor(folder, cfg, bucket, md_steps, mtd_steps)

        # Step 2: vbias  (reuse existing vbias.dat unless --force).
        if (folder / "vbias.dat").is_file() and not args.force:
            print("    vbias.dat: FOUND existing file -> REUSED (not recomputed)")
        else:
            run_vbias(folder, cfg, bucket, inp_path, root, ib - 1, md_steps,
                      args.nranks, args.threads, args.backend, use_mpi, env)

        # Step 3: probability + gradients (always rebuilt from ct/vbias).
        run_probability(folder, cfg, bucket, md_steps, mtd_steps)

    print("-" * 70)
    print(f"All per-bucket calculations done in {time.time() - t0:.1f}s")

    if args.no_reweight:
        print("Skipping final reweight (per --no-reweight).")
        return 0

    print("=" * 70)
    print("Running final mean-force integration ...")
    rc = run_reweight(inp_path, root, args.verbose)
    print("=" * 70)
    print("Pipeline complete.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

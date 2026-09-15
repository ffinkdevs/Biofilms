# Biofilm CPM in Odin — AoS / SoA / AoSoA + SIMD + raylib voxels

Odin port of `biofilms_potts.jl` (3D Cellular Potts + melanin/nutrient
reaction-diffusion) with a voxel renderer in the style of the reference
screenshot: shaded species-coloured cubes with outlines, MCS counter
top-left, species legend right, axes gizmo bottom-left.

![headless preview](preview_mcs300.png)
*Headless output (`render/ppm.odin`, no GPU): N=40, MCS 300, seed 42.
`biofilm --n 40 --mcs 300 --seed 42 --ppm preview.ppm` reproduces it.*

## Binaries

| Binary | Needs | What |
|---|---|---|
| `biofilm` | nothing (stdlib) | headless sim + Julia-compatible CSV + PPM voxel render |
| `biofilm-viewer` | raylib, display | interactive orbit viewer (`SPACE` +10 MCS, `R` reset, `1/2/3` layout) |
| `shaders/field_diffusion.comp` | Vulkan, opt-in | compute port of the field sweeps (design in `shaders/design.md`) |

```sh
./build.sh            # biofilm (headless) + tests
./build.sh viewer     # + biofilm-viewer (needs raylib in link path)
./build.sh all        # + compiles the compute shader if glslangValidator exists

./biofilm --n 40 --mcs 100 --seed 42 --layout aos --ppm out.ppm
./biofilm --no-coupled --n 20 --mcs 2 --seed 42   # CPM-only, no PDE
./biofilm --n 20 --mcs 400 --seed 42 --cells 2 --frames anim/f --every 4 \
  --width 960 --height 600 --no-stats             # 100 PPMs -> mp4 via ffmpeg
./biofilm-viewer --n 30 --mcs 4 --seed 42
odin test tests/
```

`odin/biofilm_seed42_mcs400.mp4` is a ready-made example: N=20, seed 42,
one frame per 4 MCS to 400 at 12 fps, encoded with
`ffmpeg -framerate 12 -i f_%d.ppm -c:v libx264 -pix_fmt yuv420p`.

## Per-seed numerical parity with Julia

`--rng julia` replays Julia 1.12's `MersenneTwister` stream and takes the
scalar exact-order field path (slower; for validation, not production).
`--trace` prints a machine-checkable per-MCS trace (lattice/field SHA-256,
per-cell volumes + COM bit-patterns, per-species stats); `compare/` holds
the Julia side emitting byte-identical lines:

```
TRACE_N=16 TRACE_CELLS=2 TRACE_MCS=8 TRACE_SEED=42 julia compare/jl_trace.jl
biofilm --n 16 --mcs 8 --seed 42 --cells 2 --rng julia --no-coupled --trace
diff <(...) <(...)   # identical: lattice, fields, volumes, COMs, snapshots
```

Verified identical for seeds 42 (8 MCS), 7 and 123 (30 MCS) at N=16,
plain and coupled, plus N=20 at 100 MCS (2222 trace lines) — every
Metropolis decision agrees. Note Julia 1.12 does NOT call libm
for `exp` — it evaluates `base/special/exp.jl` inline (table reduction +
minimax kernel with fused multiply-adds), which differs from libm by
1 ulp on ~10% of inputs; `cpm/julia_exp.odin` ports exactly that
(including its 256-entry table), verified 4015/4015 bit-exact at both
`-o:none` and `-o:speed`. The coupled
PDE path is verified too (RD state bit-identical). Requirements for
parity that are easy to break — do not "clean up" without re-running the
trace diff: 1-based site coordinates, `NEIGHBOURS_26` order (dz fastest),
scalar Laplacian order (x+,x-,y+,y-,z+,z-), banker's rounding in
radiolysis binning, stale-if-zero COMs, two RNG instances (placement vs
stepping), float draw only on the `dH > 0` path, and FP association:
`r*(dr*dr)`, never `(r*dr)*dr` — Julia's `dr^2` lowers to `dr*dr`, and
the two associations differ by 1 ulp (caught live at MCS 7, lane 33).

## Layouts (all three ship, selectable at runtime)

```
--layout aos    Array of Structures:  []Cell  (one struct per cell)
--layout soa    Structure of Arrays:  struct of []u8/[]i32/[]f64 columns
--layout aosoa  Array of Structures of Arrays: []Cell_Block, BLOCK=8
```

`cpm/layouts.odin` holds the canonical forms. Cell IDs are 1-based on the
lattice (`0` medium, `-1` wall) and index slots as `id-1` in every layout,
so all three consume the RNG stream in identical order: **same seed gives
bit-identical integer state in every layout** (volumes, counts), pinned by
`tests/layouts_test.odin`. Float accumulations (COM, melanin means) agree
to 1e-4.

`AOSOA_BLOCK = 8` is one choice everywhere: two AVX2 `#simd[4]f64`
vectors, the Vulkan subgroup size in `field_diffusion.comp`, one cache
line per `com_x` lane-group. Measured N=40/100 MCS, splitmix, CPM-only:

| layout | wall | CSV vs AoS |
|---|---|---|
| aos | 0.19 s | — |
| soa | 0.19 s | identical |
| aosoa | 0.19 s | identical |

Coupled mode (PDE + projection + pairwise diagnostic, i.e. the full Julia
workload) adds the fixed radial sweeps (~0.1 s per 100 MCS at N=40):
splitmix plain 0.19 s → coupled 0.33 s; in `--rng julia` mode the slower
RNG dominates and coupling hides in noise (plain 0.33 s → coupled
0.34 s). The 1D PDE itself is negligible either way.

No layout wins at N=40 because the scalar Metropolis core dominates and
costs the same everywhere. SoA/AoSoA pay off in the streaming field
sweeps at larger N (and AoSoA is the layout the compute shader wants —
see below). The honest summary: AoS for readability, AoSoA for the
GPU upgrade path, all three tested equivalent.

## SIMD (shipped, on by default)

`cpm/fields_simd.odin`: the two reaction-diffusion sweeps over 4-wide
`#simd[4]f64` x-runs with unaligned loads/stores
(`intrinsics.unaligned_load/store` — plain `^Vec` dereference emits
aligned MOVAPS and faults on unaligned rows). Per-lane production
terms (`α_M`, uptake) are gathered scalar then lifted with
`simd.from_array`; negatives clamped lane-wise. Scalar reference stays in
`cpm/sim.odin`; `test_simd_matches_scalar` asserts max diff < 1e-4
(SIMD reassociates the Laplacian sum, so tolerance — not bits — is the
contract).

Deliberately NOT vectorised: the Metropolis copy attempts (random
gather/scatter, 26-neighbour adhesion gather, data-dependent branch,
per-cell atomics). Same conclusion as the JACC port: parallelising that
core changes the dynamics (checkerboard ≠ random-sequential).

## Compute shaders (provided, opt-in)

`shaders/field_diffusion.comp` + `shaders/design.md`. One dispatch for
the field sweeps only; the Metropolis core stays on CPU (§1 of the design
doc explains why it is slower on GPU at these sizes). Memory model is
**one allocation**: `cpm/arena.odin` packs lattice + all float fields
into a single 64-byte-aligned backing buffer with recorded offsets, which
maps 1:1 onto one `VkBuffer` (bindless index 0, offsets as push
constants). Rule of thumb: SIMD fields + raylib voxels win at N≤60;
compute pays for the sweeps at N≥60; the Metropolis core never moves.

## Validation vs Julia

Default splitmix stream is statistically equivalent only — same seed is
NOT the same trajectory as Julia (splitmix64 here vs MersenneTwister
there, per the `validate_serial.jl` contract). `--rng julia` is
bit-identical per seed (see parity section). What is checked in each
mode:

- CSV columns match `validate_serial.jl` format (`CSV,seed,species,vol,ncells,mel,survived`).
- N=40, 6 cells/species, 100 MCS, seed 42, coupled, `--rng julia`:
  `tests/fixtures/serial_seed42.csv` reproduces 7/7 plus membrane m
  (0.7786 vs 0.77856) — stream and model settled in one run.
- Lattice trajectory is provably unaffected by coupling (one-way;
  `test_coupling_moves_no_sites`), so CPM-only runs share the same
  lattice path.
- `odin test tests/`: layout equivalence, SIMD-vs-scalar, determinism,
  voxel collection, coupling lattice-identity, membrane closed forms,
  coupled determinism, Julia-RNG bit-parity (2 seeds), Julia-exp
  bit-parity (20 points) — 10/10 pass.

## Ensemble: melanin ordering across seeds

`--ensemble` prints machine-readable rows per snapshot with both the
published observable (`mel_site`, volume-weighted mean over occupied
sites) and the per-parcel mean (`mel_parcel`). 256 seeds (42–297),
N=40/p6/400 MCS, plain, `--rng julia` —
`odin/compare/ensemble_n40_p6_42-297_odin.csv` (35,840 rows, ~40 s wall
at 10-way parallel):

| MCS | CS>CN>AN | CS top | CN−AN mean/min/sd |
|---|---|---|---|
| 100 | 237/256 (92.6%) | 249/256 | 0.355 / −0.290 / 0.206 |
| 200 | 211/256 (82.4%) | 234/256 | 0.476 / −0.780 / 0.375 |
| 300 | 201/256 (78.5%) | 225/256 | 0.555 / −1.253 / 0.509 |
| 400 | 190/256 (74.2%) | 217/256 | 0.615 / −1.354 / 0.611 |

P(16/16 | p=0.926) ≈ 29% — a 15/16 and a 16/16 run are the same result.
The mean gap grows with MCS while the spread grows faster.

Seeds 42–57 subset, site vs parcel side by side (sample sd, matching the
Julia table's digits):

| MCS | site CS>CN>AN | site CS top | parcel CS>CN>AN | parcel CS top | CN−AN mean/min/sd |
|---|---|---|---|---|---|
| 100 | 16/16 | 16/16 | 15/16 | 16/16 | 0.326 / 0.00065 / 0.248 |
| 200 | 11/16 | 15/16 | 11/16 | 15/16 | 0.400 / −0.18121 / 0.387 |
| 300 | 11/16 | 15/16 | 11/16 | 15/16 | 0.423 / −0.31304 / 0.490 |
| 400 | 9/16 | 12/16 | 9/16 | 12/16 | 0.453 / −0.40656 / 0.570 |

Site reproduces the Julia table cell-for-cell except MCS-200 CS-top
(15/16 here vs 16/16 printed — seed 54, CN 2.470696 > CS 2.345212
either way). Averages disagree on 3/1024 (seed,mcs) pairs; seed 43
MCS 100 flips sign between them (+0.00065 vs −0.00055). The observable
choice doesn't move the finding.

## Performance (exactness-preserving only)

N=40, 100 MCS (interleaved medians, `-o:speed`; "after" adds
`-no-bounds-check`):

| workload | before | after | speedup |
|---|---|---|---|
| julia, coupled (default) | 519 ms | 338 ms | **1.54×** |
| julia, CPM-only | 460 ms | 327 ms | **1.41×** |
| splitmix, coupled | 443 ms | 312 ms | **1.42×** |

Everything below was verified by the `--trace` diff to change no bit:

| kept | what | measured |
|---|---|---|
| lazy rejection threshold | NDL's `t` needs a 64-bit div but is only consulted with probability ~s/2⁶⁴; compute it inside the taken branch | small positive, kept (provably neutral) |
| `#force_inline` hot accessors | `cell_alive/species/volume`, `lidx`, samplers — kills call overhead in the 26-neighbour adhesion loop | ~11% |
| hoisted radial loops | radius/interior/band depend on (x,y) only; was recomputed (with sqrt+div) per site per MCS in biomass + projection | ~17% |
| `-no-bounds-check` (production binary only) | every indexed access is guarded or clamped; traces prove none fire | ~13% |
| `#force_inline dsfmt_rec` | call per state word per refill | minor, kept |

Multiplicative ≈ 0.64 (~1.5×). Profile after: `mcs_step` ~75%,
fields ~8%, radiolysis sweeps ~5%, COM/snapshots ~2%.

Deliberately NOT done:

- **Threads (measured, rejected).** Fields + projection are
  disjoint-write parallel and were threaded (8 workers) with
  bit-exact results proven by trace diff — but net was ±10% noise
  around zero at N=40: the win in the sweeps (~3× on fields) is eaten
  by migration/cache effects on the dominant sequential Markov chain
  (verified: the `mcs_step` section itself got *slower* with workers
  present). 150 lines of concurrency for ~0% is a bad trade in
  verified code.
- **Anything touching `mcs_step` arithmetic.** It is ~75% of runtime
  and ~60% of that is the fixed RNG stream (24M NDL draws + dSFMT
  refills per N=40 run — same values required, only codegen can move).
  Adhesion sums, volume terms, and acceptance tests cannot be skipped,
  reordered, or fused without changing rounding. This is the floor for
  exact reproduction; beating it means a faster (not identical) model.
- **`--rng splitmix` keeps its own stream.** The exactness work above
  applies to `--rng julia`; splitmix benefits from the same codegen
  (inline/bounds/SIMD) with its own unchanged values.

Caveats: medians on a shared box (load ~7), so treat second digits as
noise. `-o:speed` vs `-o:none` parity is covered by the same trace
diffs (run the compare scripts under both flags when touching FP
code — `-o:speed` is known to reassociate aggressively).

## Files

```
odin/
  main.odin            headless CLI (CSV + PPM)
  cpm/params.odin      species registry, beta/alpha/uptake, J matrix
  cpm/rng.odin         splitmix64 deterministic stream
  cpm/arena.odin       ONE huge allocation (bindless-ready offsets)
  cpm/layouts.odin     AoS / SoA / AoSoA canonical forms
  cpm/sim.odin         init, Metropolis step, delta_H, snapshots
  cpm/fields_simd.odin #simd[4]f64 field sweeps (splitmix mode)
  cpm/julia_exp.odin   Julia 1.12 inline exp port (bit-exact, both opt levels)
  cpm/radiolysis.odin  1D radiodialysis PDE + radial<->3D coupling (coupled mode)
  cpm/dsfmt.odin       pure-Odin dSFMT19937 (Julia-MT core, verified)
  cpm/julia_rng.odin   Julia 1.12 MersenneTwister front-end (bit-exact stream)
  render/palette.odin  FIG_COLORS parity (screenshot legend)
  render/voxels.odin   occupied-site gather + face-culling masks
  render/ppm.odin      headless isometric rasterizer (CI check)
  viewer/main.odin     raylib orbit viewer (publication view)
  shaders/             compute shader + placement rationale
  tests/               odin test suite (layouts, SIMD, determinism, coupling, RNG, exp)
  compare/jl_trace.jl  Julia side of the --trace diff (plain + coupled)
  compare/ensemble_n40_p6_42-297_odin.csv  256-seed ensemble rows (committed for citation)
```

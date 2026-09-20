# Conformance report: Odin port vs `response.growth_survival`

Second-implementation report, written from evidence in this repository
against the upstream spec at `aurascoper/PRRT-spatial-CPM@84b4d2c`
(`spec/response_growth_survival.md`, `spec/pr_checklist.md`,
`spec/check_selection.py`, `spec/check_vectors.py`,
`spec/reference_vectors.json`). Answers the three questions: what the
port implements, what it cannot implement under Semantics B, and what
no Odin test can check.

Port commit for citation: see `git log` on branch `odin-port`
(this report was written on top of `f3bf6d6` plus the review-round
follow-ups: H1 closure, G-M control, seed-42 fixture test).

## 1. Implemented: Corrections, Amendments, gates

### Corrections C1–C11

| # | Subject | Port status |
|---|---|---|
| C1 | Semantics A refill is well-mixed placement | Acknowledged; port implements Semantics B (volume doubling + resorption), never the reference stand-in |
| C2 | Branch-continuity tolerance 5e-11, not 1e-11 | Implemented: `test_resp_G_vectors` asserts `diff <= 5e-11` with the `x^3/60` rationale in comments |
| C3/C6 | G-S as worded cannot fail | Implemented in corrected form: uniform e, `sigma_div = 0`, uniform dose, mean stays exactly 1.0 (`test_resp_G_selection_null`) |
| C4 | Checker claimed 6 digits, enforced 5 | Port side: vectors asserted at relative 1e-6; the port reported this correction |
| C5 | Drift gap is dose coupling, not geometry | Implemented: G-D closed/open arms differ only in dose-expression coupling |
| C7 | G-B could not fail | Implemented: accrue-on-top-of-bank through `response_cycle` (`test_resp_G_banked`); verified by deleting the A2 reset — fails |
| C8/C11 | Kill boundary `u > SF`, not `>=` | Implemented: `response.odin` kills iff `u > sf`; `test_resp_vectors_bitexact` pins `SF1000` bits to 0 |
| C9 | 50x cap refuses nothing below 51 cells | Implemented: `test_resp_G_hoard` gates only at 56 cells; small-population control asserts report-only (`!gated0`) |
| C10 | Review round at `f3bf6d6` | Addressed below (H1 closed, G-M shipped, count strings fixed) |

Two further C10 notes, confirmed in this port: `mu` is computed as
`LN2/T_rep` (`response.odin:34`), never the 1-ulp-off literal; the G
standard branch uses the studies' association `2/(x*x)*(x + expm1(-x))`.

### Amendments A1–A5: all yes

| Amendment | Rule | Implementation |
|---|---|---|
| A1 | Unconditional survival sweep; every exposed cell drawn once per cycle | `response_cycle` phase 2 loops all live cells (`response.odin:325-337`); G-Q arrests 100% and requires `deaths == n0` |
| A2 | Dose consumed by the check; daughters born at 0 | Phase 3 resets all live doses (`342-346`); daughters take dose 0 through `cell_divide`; G-B accrues to prove it |
| A3 | Dying cell below 2 sites culled outright | `cull_dying` + per-step `resorb_sweep`; `test_resp_A3_cull` requires `len(cull_log) == 1` |
| A4 | Share instrumented, refused above 50x | `check_share_cap` + `max_share_for_power`; G-H at 56 cells, e^4-hoarder control refuses |
| A5 | Failed-draw cell never divides | State check `response.odin:353`; `test_resp_G_mitotic_parent` aims at that one line (see §4) |

### Gates (spec §5 + amendments)

| Gate | Odin test | Rejection input | Mutation evidence |
|---|---|---|---|
| G-N | `test_resp_G_null` | SF scaled by constant | arithmetic on production output |
| G-O | `test_resp_G_obliteration` | SF floored at 1e-300 | arithmetic on production output |
| G-P | `test_resp_G_protraction` | naive no-Taylor G | demo (test-local `naive_G`); the production path is pinned bit-exact (`G1e-9`, `G1e-6` bits) |
| G-C | `test_resp_G_consistency` | `T_rep = 15 h` | through production `g_of_t` |
| G-S | `test_resp_G_selection_null` | forced `sigma_div = 0.1` | through the real path at zero dose |
| G-D | `test_resp_G_direction` | inverse uptake; open-is-closed | through both arms (5 seeds + 2 controls) |
| G-Q | `test_resp_G_quorum` | attempt-gated draw | registry count (`eligible == 0`); does NOT cover A5 — that is G-M's job |
| G-M | `test_resp_G_mitotic_parent` | — (the control itself) | guard deleted → 252 divisions, `next_id` 15 → 519; fails |
| G-B | `test_resp_G_banked` | banked arithmetic | positive path fails with A2 reset deleted (upstream probe); rejection input is arithmetic by its own comment |
| G-H | `test_resp_G_hoard` | e^4 hoarder + small-pop report-only | through production `check_share_cap`/`max_share_for_power` |
| H1 | `test_layout_equivalence` | — | SoA no-write `cell_set_expr` fails (probe, pre-fix design); fixed setup per §4 |
| H4 | `test_resp_H4_no_dead_sigmas`, `test_resp_A3_cull`, `test_resp_audit` | kill-without-clear; tampered volume | through production `cell_kill`/`audit_lattice`; zeroing removed → A3 fails |
| Doubling | `test_resp_doubling` | band refusal on toy counts | wiring: report value feeds the check |

Supporting pins (no rejection input needed — exact equality against an
external oracle is self-biting): `test_resp_G_vectors`,
`test_resp_vectors_bitexact` (13 calls cover all 14 file scalars),
`test_resp_SF_table`, Julia RNG streams (2 seeds), Julia exp,
`test_seed42_golden_csv` (see §4), coupling lattice-identity,
radiolysis closed forms, coupled determinism, SIMD-vs-scalar, voxel
collection.

## 2. Cannot implement under Semantics B

1. **Well-mixed refill (C1).** The reference places daughters uniformly
   over free lesion sites. The port divides in place by construction.
   Anything asserting the reference placement is N/A.
2. **Dead-site NaN poisoning.** The reference sets a dead site's
   expression to NaN, so a dead parent yields NaN daughters visibly.
   The port never clears that value; a dead parent yields ordinary
   daughters and no error. This is exactly why A5 needs an explicit
   guard and control here, and why the guard's absence is silent.
3. **Checker-shaped population gates.** Upstream G-M/G-Q run at 2000
   cells with 30% arrest. The port ships port-shaped controls at toy
   scale (same rule, same arithmetic shape, different N). The
   statistical margins do not transfer; the logic does.
4. **Semantics-A division accounting.** Any statement about refill
   parent pools or occupied-list removal has no port counterpart.

## 3. No Odin test can check these

- **The 256-seed ensemble CSV is a statistical artifact, not a gate.**
  `odin/compare/ensemble_n40_p6_42-297_odin.csv` (256 seeds, 35,840
  rows) is committed for citation; no test reads it, and none should:
  re-running 256 N=40 seeds per `odin test` invocation is not a unit
  check. Per-seed determinism (which is what would silently break) is
  covered by `test_seed42_golden_csv` + the `--trace` diff.
- **Reference-side statements.** That study 3's sweep is unconditional,
  that the committed run made ~5.6e5 draws, that the verdict is
  unaffected by C11 — these describe another repository's code and
  history. The port can only mirror the rule (`u > sf`).
- **The R verifier's receipt** (upstream): proves checks ran, not that
  they are correct. Mirrored here: `odin test` passing proves the
  suite ran green, not that each gate bites — which is why every gate
  above carries mutation evidence, and why the suite refuses
  test-only hooks (none exist in the runner; the control suite mutates
  a scratch copy per upstream practice).
- **The Julia coupled-path `X_red` block** (upstream): stops a bad run
  by accident, not as an enforced gate. Port analogue: one-way
  coupling is pinned by `test_coupling_moves_no_sites`, but no test
  can assert the absence of an accidental guard elsewhere.

## 4. Audit: does each rejection input mutate the production path?

Full pass over all 28 tests (the reviewer's question 1). Method: for
each test, either a mutation probe was run (defect planted in a
scratch copy, suite re-run) or the rejection input is classified:

- **P (production):** the rejection input flows through shipped code
  and the assertion reads shipped state.
- **A (arithmetic):** the rejection input is arithmetic on a
  production output, showing the assertion itself can fail (G-N, G-O,
  G-B-rejection). Accepted where the positive path separately has
  probe evidence.
- **D (demo):** test-local construction showing what failure looks
  like (G-P `naive_G`). Accepted only because the production path is
  pinned bit-exact independently.

Probes run during this review round (scratch copies, all restored):

| Probe | Result |
|---|---|
| SoA `cell_set_expr` writes nothing | `test_layout_equivalence` fails (pre-fix design caught it; post-fix setup also fails — heterogeneous e, all-layout dose compare, division gate) |
| `response.odin:353` guard deleted | `test_resp_G_mitotic_parent` fails: 252 divisions, `next_id` 15 → 519; 27/28 others pass (G-D seed 15 also trips, by accident, as upstream recorded) |
| seed 42 → 43 in `test_seed42_golden_csv` | fails; restored to 42 → passes |
| A2 reset deleted (upstream, C10) | `test_resp_G_banked` fails |
| A3 site-zeroing removed (upstream, H4) | `test_resp_A3_cull` fails |

H1 closure (upstream's prescribed three lines): `init_expression_lognormal`
+ `assign_dose_scaled_e` + `v_target = 2` are in `test_layout_equivalence`,
with a `next_id`-advanced assertion (divisions must occur), all-layout
dose comparison (dead cells keep heterogeneous dose; live-only would be
constant-vs-constant after the A2 reset), and `next_id` equality across
layouts (divide counts agree, covering `grow_registry_soa/aosoa` and the
per-layout `n_alive` bookkeeping of H3).

## 5. Count strings corrected in this round

- Bit-exact test: 13 `check()` calls cover all 14 vectors-file
  scalars (`SF(10 Gy)@G(96h)` listed twice) + continuity bound = 15
  file entries. Comments and README now say so.
- `serial_seed42.csv` 7/7 is now read by `test_seed42_golden_csv`
  (seed 43 probe fails). The "manual run" qualifier is retired.
- Ensemble CSV: 256 seeds, never 16. The "16-seed" string was a commit
  message (`2bbfe0b` era); history is not rewritten, the record stands
  corrected here. No test reads the CSV, by design (§3).
- Suite counts: 28 tests, 17 in the response ladder (10 gates +
  doubling + H4 + A3 + audit + 3 vector/table pins).

package tests

import "core:testing"
import "core:math"
import cpm "../cpm"

// Gate ladder for the growth/survival response module
// (PRRT-spatial-CPM spec/pr_checklist.md §4). One test per gate, each
// with the input that makes it fail: a gate with no such input reads
// the same whether it checks something or nothing.
//
// Conventions: small sims (N=16, 2 parcels/species, AoS unless the test
// is about layouts), fixed seeds (deterministic — every assert below
// was verified by running, never fitted). Survival uniforms and drift
// normals always come from s.rng (splitmix); jr is never touched (H2).

@(private)
RESP_TOL_6 :: 1e-6 // six significant digits, relative

@(private)
rel_eq :: proc(a, b, tol: f64) -> bool {
	d := a - b
	if d < 0 { d = -d }
	ab := b
	if ab < 0 { ab = -ab }
	if ab == 0 {
		return d == 0
	}
	return d <= tol * ab
}

@(private)
make_rsim :: proc(seed: u64, vt: i32 = 120, n := 16, cells := 2) -> cpm.Sim {
	p := cpm.default_params()
	p.n = n
	p.n_cells_per_species = cells
	p.v_target = vt
	return cpm.sim_init(p, .AoS, seed)
}

@(private)
resp_params :: proc() -> cpm.Response_Params {
	return cpm.default_response_params()
}

@(private)
G96 :: proc() -> f64 {
	return cpm.g_of_t(96.0, cpm.RESP_MU)
}

// ---- G vectors (reference_vectors.json) ----
//
// A rounding note that matters: the file's values are DECIMAL roundings
// (%.6f style: G96 true 0.04406792658… prints as 0.044068), not exact
// values — 0.044068 is NOT the 6-significant-digit rounding of the true
// value (that would be 0.0440679). Verified bit-for-bit against Julia
// 1.12 (same expression, same libm expm1): our port reproduces Julia
// exactly on all 11 values below. So the honest assertion is abs-diff
// against the printed rounding (half a last-place unit), not relative
// 1e-6, which a correct implementation misses by ~2e-6 relative.

@(private)
ABS6 :: 1e-6 // %.6f-printed vectors in [0, 1]

@(test)
test_resp_G_vectors :: proc(t: ^testing.T) {
	d24 := abs(cpm.g_of_t(24.0, cpm.RESP_MU) - 0.164076)
	testing.expectf(t, d24 <= ABS6, "G(24h) off by %v", d24)
	d96 := abs(cpm.g_of_t(96.0, cpm.RESP_MU) - 0.044068)
	testing.expectf(t, d96 <= ABS6, "G(96h) off by %v", d96)
	testing.expectf(t, rel_eq(cpm.g_of_t(1e-9, cpm.RESP_MU), 0.999999999846, 1e-9),
		"G(1e-9h) = %v", cpm.g_of_t(1e-9, cpm.RESP_MU))
	testing.expectf(t, rel_eq(cpm.g_of_t(1e-6, cpm.RESP_MU), 0.999999845967, 1e-9),
		"G(1e-6h) = %v", cpm.g_of_t(1e-6, cpm.RESP_MU))
	// Branch continuity at x = 1e-3: approach from both sides through
	// the public evaluator (Taylor below, full form above). True gap is
	// x^3/60 = 1.67e-11, so the file tolerance is 5e-11 (a literal
	// 1e-11 test fails on a correct implementation — spec CORRECTION 2).
	g_below := cpm.g_of_t(9.99999999e-4, cpm.RESP_MU)
	g_above := cpm.g_of_t(1.000000001e-3, cpm.RESP_MU)
	diff := g_below - g_above
	if diff < 0 {
		diff = -diff
	}
	testing.expectf(t, diff <= 5e-11, "branch gap %v exceeds 5e-11", diff)
}

// ---- SF table (spec §3) ----

@(test)
test_resp_SF_table :: proc(t: ^testing.T) {
	rp := resp_params()
	G := G96()
	pairs := [][2]f64{
		{0, 1.0},
		{2, 6.122734e-01},
		{5, 2.819285e-01},
		{10, 6.964060e-02},
		{20, 2.858008e-03},
		{40, 9.851017e-07},
	}
	for p in pairs {
		got := cpm.sf_lq(p[0], G, rp.alpha, rp.beta)
		d := got - p[1]
		if d < 0 {
			d = -d
		}
		testing.expectf(t, d <= ABS6, "SF(%v) = %v, want %v", p[0], got, p[1])
	}
	testing.expect(t, cpm.sf_lq(0, G, rp.alpha, rp.beta) == 1.0)
	v := cpm.sf_lq(1000, G, rp.alpha, rp.beta)
	testing.expectf(t, v == 0.0 && v == v, "SF(1000) = %v, want +0.0", v)
	// Protraction direction at fixed dose: shorter exposure is strictly
	// more lethal (spec §3 + G-P).
	s24 := cpm.sf_lq(10, cpm.g_of_t(24.0, cpm.RESP_MU), rp.alpha, rp.beta)
	s96 := cpm.sf_lq(10, G, rp.alpha, rp.beta)
	testing.expectf(t, s24 < s96, "SF(10|24h)=%v must be < SF(10|96h)=%v", s24, s96)
	d24 := s24 - 0.033896
	if d24 < 0 {
		d24 = -d24
	}
	testing.expectf(t, d24 <= ABS6, "SF(10|24h) = %v", s24)
	d96 := s96 - 0.069641
	if d96 < 0 {
		d96 = -d96
	}
	testing.expectf(t, d96 <= ABS6, "SF(10|96h) = %v", s96)
}

// ---- G-N: SF(0) == 1.0 exactly; rejects a scaled SF ----

@(test)
test_resp_G_null :: proc(t: ^testing.T) {
	rp := resp_params()
	testing.expect(t, cpm.sf_lq(0, G96(), rp.alpha, rp.beta) == 1.0)
	// Rejection input: an SF scaled by a constant is not 1.0 at D = 0.
	testing.expect(t, 1.000001 * cpm.sf_lq(0, G96(), rp.alpha, rp.beta) != 1.0)
}

// ---- G-O: SF(1000) == 0.0, never NaN; rejects a floored SF ----

@(test)
test_resp_G_obliteration :: proc(t: ^testing.T) {
	rp := resp_params()
	v := cpm.sf_lq(1000, G96(), rp.alpha, rp.beta)
	testing.expectf(t, v == 0.0, "SF(1000) = %v", v)
	testing.expect(t, v == v) // never NaN
	// Rejection input: floored at a tiny positive value.
	floored := v if v > 1e-300 else 1e-300
	testing.expect(t, floored != 0.0)
}

// ---- G-P: protraction ordering + small-x limit; rejects no-Taylor G ----

@(private)
naive_G :: proc(T_h, mu: f64) -> f64 {
	// The cancelled form gate G5c caught: x - 1 + exp(-x) in float64.
	x := mu * T_h
	return 2.0 / (x * x) * (x - 1.0 + cpm.jexp_f64(-x))
}

@(test)
test_resp_G_protraction :: proc(t: ^testing.T) {
	rp := resp_params()
	g24 := cpm.g_of_t(24.0, cpm.RESP_MU)
	g96 := G96()
	doses := []f64{2, 5, 10, 20, 40}
	for D in doses {
		a := cpm.sf_lq(D, g24, rp.alpha, rp.beta)
		b := cpm.sf_lq(D, g96, rp.alpha, rp.beta)
		testing.expectf(t, a < b, "D=%v: SF(24h)=%v must be < SF(96h)=%v", D, a, b)
	}
	testing.expectf(t, rel_eq(cpm.g_of_t(1e-9, cpm.RESP_MU), 1.0, 1e-6),
		"G(1e-9h) = %v, want -> 1", cpm.g_of_t(1e-9, cpm.RESP_MU))
	// Rejection input: the naive form returns exactly 0.0 at x = 1e-9.
	testing.expect(t, naive_G(1e-9, cpm.RESP_MU) == 0.0)
}

// ---- G-C: 12 < 1/G(96h) < 23; rejects T_rep = 15 h ----

@(test)
test_resp_G_consistency :: proc(t: ^testing.T) {
	inv := 1.0 / G96()
	testing.expectf(t, inv > 12.0 && inv < 23.0, "1/G(96h) = %v", inv)
	// Reference prints %.4f (22.6922); tolerance is half a last place.
	d := inv - 22.6922
	if d < 0 {
		d = -d
	}
	testing.expectf(t, d <= 1e-4, "1/G(96h) = %v", inv)
	// Rejection input: T_rep = 15 h misses the band wildly (~2.85).
	mu_bad := 0.6931471805599453 / 15.0
	inv_bad := 1.0 / cpm.g_of_t(96.0, mu_bad)
	testing.expectf(t, !(inv_bad > 12.0 && inv_bad < 23.0),
		"T_rep=15h gives %v, must be outside [12, 23]", inv_bad)
}

// ---- G-S: uniform e, sigma 0, uniform dose — mean stays exactly 1.0 ----

@(test)
test_resp_G_selection_null :: proc(t: ^testing.T) {
	rp := resp_params()
	rp.sigma_div = 0.0
	s := make_rsim(20260701, 20)
	defer cpm.sim_destroy(&s)
	G := G96()
	for _ in 0..<4 {
		cpm.assign_dose_uniform(&s, 5.0)
		_ = cpm.response_cycle(&s, rp, G, rp.cycle_hours)
		for _ in 0..<10 {
			cpm.mcs_step(&s)
		}
		cpm.resorb_sweep(&s)
		cpm.update_centers_of_mass(&s)
	}
	st := cpm.expression_stats(&s)
	testing.expectf(t, st.mean_ln_e == 0.0, "mean ln e = %v, want exactly 0", st.mean_ln_e)
	// Zero-dose form: nobody may die (catches SF wrong at D = 0).
	s0 := make_rsim(20260702, 20)
	defer cpm.sim_destroy(&s0)
	deaths := 0
	for _ in 0..<2 {
		cpm.assign_dose_uniform(&s0, 0.0)
		rep := cpm.response_cycle(&s0, rp, G, rp.cycle_hours)
		deaths += rep.deaths
	}
	testing.expectf(t, deaths == 0, "zero dose killed %d cells", deaths)
	// Rejection input: additive-offset inheritance shifts the mean.
	s1 := make_rsim(20260703, 20)
	defer cpm.sim_destroy(&s1)
	shifted := false
	for id in 1..<s1.next_id {
		if cpm.cell_alive(&s1, id) && cpm.cell_volume(&s1, id) >= 40 {
			da, _ := cpm.cell_divide(&s1, id, 1.1, 1.1) // +0.1 offset, not drift
			if da != 0 {
				shifted = true
				break
			}
		}
	}
	testing.expect(t, shifted)
	st1 := cpm.expression_stats(&s1)
	testing.expectf(t, st1.mean_ln_e != 0.0,
		"offset control must move mean ln e, got %v", st1.mean_ln_e)
}

// ---- G-D: closed drift below open drift, both sides declared ----

@(private)
run_closed :: proc(seed: u64, dose_k, dose_power: f64) -> (closed_drift, open_drift: f64) {
	rp := resp_params()
	G := cpm.g_of_t(96.0, cpm.RESP_MU)
	// Closed arm: heterogeneous e, dose ∝ e^power, full cycles.
	// N=20 with 4 parcels/species (28 cells): large enough that a
	// partial kill neither wipes the population nor vanishes in it.
	sc := make_rsim(seed, 20, 20, 4)
	defer cpm.sim_destroy(&sc)
	cpm.init_expression_lognormal(&sc, 0.6, seed * 1000 + 7)
	e0 := cpm.expression_stats(&sc).mean_ln_e
	for _ in 0..<4 {
		cpm.assign_dose_scaled_e(&sc, dose_k, dose_power)
		_ = cpm.response_cycle(&sc, rp, G, rp.cycle_hours)
		for _ in 0..<10 {
			cpm.mcs_step(&sc)
		}
		cpm.resorb_sweep(&sc)
		cpm.update_centers_of_mass(&sc)
	}
	closed_drift = cpm.expression_stats(&sc).mean_ln_e - e0
	// Open arm: identical init and Metropolis evolution, doses frozen,
	// no death or division. Only background volume-erosion moves the
	// mean (small, either sign) — the fall must clear it, not zero.
	so := make_rsim(seed, 20, 20, 4)
	defer cpm.sim_destroy(&so)
	cpm.init_expression_lognormal(&so, 0.6, seed * 1000 + 7)
	e0o := cpm.expression_stats(&so).mean_ln_e
	for _ in 0..<4 {
		for _ in 0..<10 {
			cpm.mcs_step(&so)
		}
		cpm.update_centers_of_mass(&so)
	}
	open_drift = cpm.expression_stats(&so).mean_ln_e - e0o
	return closed_drift, open_drift
}

@(test)
test_resp_G_direction :: proc(t: ^testing.T) {
	// Declared statistical test: 5 fixed seeds; every seed must show
	// closed < open with closed < 0. Deterministic — no sampling noise.
	// (An earlier revision used 14 cells with a heavy kill and wiped
	// the population in cycle 0; the closed drift then measured
	// extinction, not selection. 28 cells at k = 3 keep a partial kill.)
	seeds := [5]u64{11, 12, 13, 14, 15}
	for seed in seeds {
		closed, open := run_closed(seed, 3.0, 1.0)
		testing.expectf(t, closed < 0.0, "seed %d: closed drift = %v, want < 0", seed, closed)
		testing.expectf(t, closed < open, "seed %d: closed %v not below open %v", seed, closed, open)
	}
	// Rejection input: inverse uptake (dose ∝ 1/e) kills low-e cells,
	// so the mean must rise — the ordering check refuses it.
	closed, open := run_closed(11, 3.0, -1.0)
	testing.expectf(t, !(closed < open && closed < 0.0),
		"inverse control must fail ordering, got closed=%v open=%v", closed, open)
}

// ---- G-Q: 1000 Gy kills every exposed cell, arrested or not ----

@(test)
test_resp_G_quorum :: proc(t: ^testing.T) {
	rp := resp_params()
	s := make_rsim(20260704, 100000) // unreachable trigger: 100% arrested
	defer cpm.sim_destroy(&s)
	n0 := cpm.count_alive(&s)
	testing.expect(t, n0 > 0)
	cpm.assign_dose_uniform(&s, 1000.0)
	rep := cpm.response_cycle(&s, rp, G96(), rp.cycle_hours)
	testing.expectf(t, rep.deaths == n0, "deaths %d of %d exposed", rep.deaths, n0)
	// Rejection input: the draw gated on a division attempt. No cell
	// here can attempt (trigger unreachable by construction), so the
	// gated variant draws zero coins and kills nothing — a gate that
	// accepts it cannot see the A1 hole.
	eligible := 0
	for id in 1..<s.next_id {
		if cpm.cell_alive(&s, id) && cpm.cell_volume(&s, id) >= 2 * s.params.v_target {
			eligible += 1
		}
	}
	testing.expectf(t, eligible == 0, "attempt-gated control must draw 0, drew %d", eligible)
}

// ---- G-B: per-cycle SF identical across four equal-dose cycles ----

@(test)
test_resp_G_banked :: proc(t: ^testing.T) {
	rp := resp_params()
	G := G96()
	s := make_rsim(20260705, 120)
	defer cpm.sim_destroy(&s)
	sfs: [4]f64
	for c in 0..<4 {
		cpm.assign_dose_uniform(&s, 5.0)
		rep := cpm.response_cycle(&s, rp, G, rp.cycle_hours)
		sfs[c] = rep.mean_sf
	}
	testing.expect(t, sfs[0] == sfs[1] && sfs[1] == sfs[2] && sfs[2] == sfs[3])
	// Rejection input: banked dose (A2 violated — the running total
	// 5, 10, 15, 20 Gy) drifts the per-cycle SF, so the four are not
	// all equal. Pure arithmetic: the gate's content is that reset and
	// banked differ, and response_cycle always resets.
	banked: [4]f64
	dose := 0.0
	for c in 0..<4 {
		dose += 5.0
		banked[c] = cpm.sf_lq(dose, G, rp.alpha, rp.beta)
	}
	testing.expect(t, !(banked[0] == banked[1] && banked[1] == banked[2] && banked[2] == banked[3]))
}

// ---- G-H: share capped at 50x; e^4 control refuses ----

@(test)
test_resp_G_hoard :: proc(t: ^testing.T) {
	s := make_rsim(20260707, 120)
	defer cpm.sim_destroy(&s)
	cpm.init_expression_lognormal(&s, 0.6, 777)
	share, ok := cpm.check_share_cap(&s)
	testing.expectf(t, ok, "lognormal share %v must be <= 50x", share)
	testing.expectf(t, share < 50.0 && share > 1.0, "share %v out of sane range", share)
	// Rejection input: one hoarder among many. Share is bounded by the
	// live-cell count, so this needs > 50 cells: N=24 holds 48. One cell
	// at e = 30 against e = 0.3 elsewhere concentrates past the cap.
	s2 := make_rsim(20260771, 120, 24, 8)
	defer cpm.sim_destroy(&s2)
	first := true
	for id in 1..<s2.next_id {
		if cpm.cell_alive(&s2, id) {
			cpm.cell_set_expr(&s2, id, 0.3)
			if first {
				cpm.cell_set_expr(&s2, id, 30.0)
				first = false
			}
		}
	}
	share4 := cpm.max_share_for_power(&s2, 4.0)
	testing.expectf(t, share4 > 50.0, "e^4 share %v must refuse the cap", share4)
}

// ---- Doubling report: arithmetic + band + wiring ----

@(test)
test_resp_doubling :: proc(t: ^testing.T) {
	// Unit: 14 cells, 322 divisions over a 1344 h window -> 58.44 h,
	// inside the declared ±2x band around 58.4 h.
	td := cpm.doubling_time_h(322, 14, 1344.0)
	testing.expectf(t, td == 1344.0 * 14.0 / 322.0, "T_d = %v", td)
	testing.expect(t, cpm.check_doubling_band(td, 29.2, 116.8))
	testing.expect(t, !cpm.check_doubling_band(10.0, 29.2, 116.8))
	testing.expect(t, !cpm.check_doubling_band(math.INF_F64, 29.2, 116.8))
	// Wiring: a real cycle reports T_d consistent with its own counts.
	s := make_rsim(20260708, 20)
	defer cpm.sim_destroy(&s)
	n0 := cpm.count_alive(&s)
	cpm.assign_dose_uniform(&s, 0.0) // all survive; divisions counted
	rep := cpm.response_cycle(&s, resp_params(), G96(), 1344.0)
	testing.expectf(t, rep.doubling_h == cpm.doubling_time_h(rep.divisions, n0, 1344.0),
		"T_d %v inconsistent with counts", rep.doubling_h)
}

// ---- H4: no lattice entry names a dead cell ----

@(test)
test_resp_H4_no_dead_sigmas :: proc(t: ^testing.T) {
	s := make_rsim(20260709, 120)
	defer cpm.sim_destroy(&s)
	testing.expect(t, cpm.assert_no_dead_sigmas(&s))
	// Rejection input: the old kill-without-clear path. A live cell
	// with sites is killed directly: its sigma stays on the lattice
	// and the invariant must refuse it.
	victim := 0
	for id in 1..<s.next_id {
		if cpm.cell_alive(&s, id) && cpm.cell_volume(&s, id) > 4 {
			victim = id
			break
		}
	}
	testing.expect(t, victim != 0)
	cpm.cell_kill(&s, victim) // deliberately NOT clearing: the H4 hole
	testing.expectf(t, !cpm.assert_no_dead_sigmas(&s),
		"kill-without-clear must trip the invariant")
	// Correct path restores it: zero the victim's sites, re-check.
	n := s.params.n
	for i in 0..<s.arena.n3 {
		if s.arena.lattice[i] == i32(victim) {
			s.arena.lattice[i] = 0
		}
	}
	_ = n
	testing.expect(t, cpm.assert_no_dead_sigmas(&s))
}

// ---- H4 liveness: the A3 cull fires on a resorbed dyer ----

@(test)
test_resp_A3_cull :: proc(t: ^testing.T) {
	s := make_rsim(20260710, 120)
	defer cpm.sim_destroy(&s)
	// Mark a live cell dying; Vt=0 resorbs it over subsequent MCS.
	target := 0
	for id in 1..<s.next_id {
		if cpm.cell_alive(&s, id) {
			target = id
			break
		}
	}
	testing.expect(t, target != 0)
	cpm.cell_set_state(&s, target, cpm.CELL_DYING)
	culled := false
	for _ in 0..<40 {
		cpm.mcs_step(&s)
		if cpm.resorb_sweep(&s) > 0 {
			culled = true
			break
		}
		cpm.update_centers_of_mass(&s)
	}
	testing.expect(t, culled)
	testing.expect(t, !cpm.cell_alive(&s, target))
	testing.expect(t, len(s.cull_log) == 1)
	testing.expect(t, cpm.assert_no_dead_sigmas(&s))
}

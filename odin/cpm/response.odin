package cpm

import "core:math"

foreign import libc "system:c"

foreign libc {
	@(link_name="expm1")
	c_expm1 :: proc(x: f64) -> f64 ---
}

// Growth/survival response module: protracted linear-quadratic survival
// with the Lea-Catcheside factor, evaluated as a stochastic transition
// at mitosis (PRRT-spatial-CPM spec/response_growth_survival.md +
// Amendments A1-A4). This is the minimal extension flipping
// response.growth_survival from unsupported to supported — nothing more.
//
// Provenance (spec §2, declared not tuned):
//   alpha = 0.24 Gy^-1, beta = 0.06 Gy^-2 — EBRT fit on NCI-H69
//     (Tamborino et al. 2025, JNM 66:1291). NOT the PRRT-fitted pair
//     (those embed protraction; G would double-count it).
//   T_rep = 1.5 h declared choice; mu = ln2/T_rep.
//   T_active = 96 h exposure window for the per-cycle G.
//   sigma_div = 0.10 (study3_run.py:47).
// Test vectors: PRRT-spatial-CPM spec/reference_vectors.json.

RESP_ALPHA   :: 0.24
RESP_BETA    :: 0.06
RESP_T_REP_H :: 1.5
// mu is COMPUTED as ln2/T_rep, exactly like the reference
// (MU = np.log(2.0)/T_REP) and Julia (log(2)/1.5) — not the decimal
// literal 0.4620981203732968, which differs by 1 ulp and moves G24 by
// 1 ulp with it. Comptime division rounds identically to runtime.
RESP_MU :: math.LN2 / RESP_T_REP_H
RESP_T_ACTIVE_H :: 96.0
RESP_SIGMA_DIV  :: 0.10
RESP_T_DOUBLING_H :: 58.4 // reference doubling (report band, not dynamics)
RESP_SHARE_CAP  :: 50.0   // A4: refuse above 50x uniform share

// Lea-Catcheside factor for exposure window T_h at repair rate mu.
// Exact evaluation order of the reference (study2_run.py G_of_T):
// Taylor 1 - x/3 + x^2/12 below x = 1e-3, else 2/(x*x)*(x + expm1(-x)).
// expm1 is C99 (correctly rounded): the naive x - 1 + exp(-x) cancels to
// 0.0 at small x (gate G5c's defect), and the Taylor branch is the only
// accepted fix. jexp-based exp would also work here (x >= 1e-3 on the
// full branch), but expm1 states the intent.
//
// Bit-parity note: mu is computed as ln2/T_rep like the reference, not
// taken from the file's decimal literal (which differs 1 ulp and moves
// G24 with it — found by bit-matching). expm1 is C99 correctly rounded;
// gates assert 1e-6, six orders above any libm dust.
g_of_t :: proc(T_h, mu: f64) -> f64 {
	x := mu * T_h
	if x < 1e-3 {
		return 1.0 - x / 3.0 + x * x / 12.0
	}
	return 2.0 / (x * x) * (x + c_expm1(-x))
}

// Protracted LQ survival (preprint Eq. sf): SF = exp(-alpha*D - beta*G*D^2).
// Exact association of the reference: -(alpha*D + beta*G*D^2), one exp.
// jexp_f64 = Julia's inline exp bit-for-bit (any correctly-rounded exp
// agrees to the 6-sig-digit gates; jexp keeps one exp everywhere).
sf_lq :: proc(D, G, alpha, beta: f64) -> f64 {
	return jexp_f64(-(alpha * D + beta * G * D * D))
}

// Standard normal draw (Marsaglia polar, no cached spare so consumption
// is a pure function of the stream). H2: drift normals come from s.rng
// (splitmix64 fast stream) — never jr, which the parity fixtures own.
rng_normal :: proc(r: ^Rng) -> f64 {
	for {
		u := 2.0 * rng_f64(r) - 1.0
		v := 2.0 * rng_f64(r) - 1.0
		s := u * u + v * v
		if s >= 1.0 || s == 0.0 {
			continue
		}
		return u * math.sqrt(-2.0 * math.ln(s) / s)
	}
}

// Assign a uniform dose (Gy) to every live cell.
assign_dose_uniform :: proc(s: ^Sim, D: f64) {
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			cell_set_dose(s, id, D)
		}
	}
}

// Accrue a uniform delivery on top of the current accumulated dose.
// Models transport delivering D Gy into whatever is banked: with the A2
// reset present every cycle reads exactly D; with it deleted the bank
// grows and the G-B gate fails. Assign overwrites; accrue adds — the
// distinction the whole gate turns on.
accrue_dose_uniform :: proc(s: ^Sim, D: f64) {
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			cell_set_dose(s, id, cell_dose(s, id) + D)
		}
	}
}

// Assign dose proportional to expression^power (power = 1 is the
// specified uptake rule; power = -1 is the G-D rejection control that
// must reverse selection).
assign_dose_scaled_e :: proc(s: ^Sim, k, power: f64) {
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		e := cell_expr(s, id)
		cell_set_dose(s, id, k * math.pow(e, power))
	}
}

// Assign each live cell the mean of a per-site dose map over its own
// sites. The G-D open arm builds the map once (cycle-1 doses) and
// freezes it: refilled sites keep their old dose and kill whoever moves
// in, so selection still acts, blind to who moved. Map layout is the
// arena's (lidx order); exterior entries are read as-is.
assign_dose_from_map :: proc(s: ^Sim, dose_map: []f64) {
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		sum := 0.0
		cnt := 0
		for i in 0..<s.arena.n3 {
			if s.arena.lattice[i] == i32(id) {
				sum += dose_map[i]
				cnt += 1
			}
		}
		if cnt > 0 {
			cell_set_dose(s, id, sum / f64(cnt))
		} else {
			cell_set_dose(s, id, 0.0)
		}
	}
}

// Snapshot the current per-site dose assignment into a frozen map
// (cell dose spread over that cell's sites; medium reads 0.0).
freeze_dose_map :: proc(s: ^Sim, dose_map: []f64) {
	for i in 0..<s.arena.n3 {
		sig := s.arena.lattice[i]
		if sig > 0 && cell_alive(s, int(sig)) {
			dose_map[i] = cell_dose(s, int(sig))
		} else {
			dose_map[i] = 0.0
		}
	}
}

// Heterogeneous initial expression: lognormal(sigma) per live cell in
// ascending id order from a FRESH stream (never a sim stream, so setup
// cannot disturb any trajectory), mean-normalized to exactly 1.0 over
// live cells (reference: mean over occupied voxels exactly 1.0).
init_expression_lognormal :: proc(s: ^Sim, sigma: f64, seed: u64) {
	r := rng_init(seed)
	sum := 0.0
	cnt := 0
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		e := math.exp(sigma * rng_normal(&r))
		cell_set_expr(s, id, e)
		sum += e
		cnt += 1
	}
	if cnt == 0 {
		return
	}
	mean := sum / f64(cnt)
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			cell_set_expr(s, id, cell_expr(s, id) / mean)
		}
	}
}

// Per-cycle expression distribution + activity share (checklist §5.5:
// report min/max/mean-ln-e/max-share, not a mean alone). Share assumes
// uptake proportional to e under K1 renormalization (reference
// act_map_prop_e): share_i = e_i / mean(e), max over live cells.
Expression_Stats :: struct {
	n:         int,
	min_e:     f64,
	max_e:     f64,
	mean_ln_e: f64,
	max_share: f64,
}

expression_stats :: proc(s: ^Sim) -> Expression_Stats {
	st: Expression_Stats
	sum_ln := 0.0
	sum_e := 0.0
	st.min_e = math.INF_F64
	st.max_e = -math.INF_F64
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		e := cell_expr(s, id)
		st.n += 1
		if e < st.min_e { st.min_e = e }
		if e > st.max_e { st.max_e = e }
		sum_ln += math.ln(e)
		sum_e += e
	}
	if st.n == 0 {
		return st
	}
	st.mean_ln_e = sum_ln / f64(st.n)
	st.max_share = st.max_e / (sum_e / f64(st.n))
	return st
}

// A4/G-H: refuse above 50x uniform share — but only where refusal is
// possible. Share is max/mean, so it can never exceed the live-cell
// count: at 50 cells or fewer the cap refuses nothing (Correction 9).
// Returns (share, gated, ok): gated is false at n <= 50 (report only),
// ok is false only when gated and over cap.
check_share_cap :: proc(s: ^Sim, cap := RESP_SHARE_CAP) -> (share: f64, gated, ok: bool) {
	st := expression_stats(s)
	if st.n == 0 {
		return 0, false, true
	}
	gated = st.n > 50
	ok = !gated || st.max_share <= cap
	return st.max_share, gated, ok
}

// Same share under an alternate uptake power (the G-H control is
// power = 4, which must refuse).
max_share_for_power :: proc(s: ^Sim, power: f64) -> f64 {
	sum := 0.0
	mx := 0.0
	n := 0
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		w := math.pow(cell_expr(s, id), power)
		sum += w
		if w > mx { mx = w }
		n += 1
	}
	if n == 0 {
		return 0
	}
	return mx / (sum / f64(n))
}

// Realized population doubling time over a cycle window from the
// growth factor g = 1 + f (f = divisions per starting cell):
// T_d = cycle_hours * ln2 / ln(g). (An earlier revision used
// cycle_hours * n_start / n_divisions, which errs by 17% at f = 0.5;
// the two agree only at f = 1.) +Inf when nothing divided. The band is
// declared by the caller; the check, not the number, is the gate.
doubling_time_h :: proc(n_divisions, n_start: int, cycle_hours: f64) -> f64 {
	if n_divisions <= 0 || n_start <= 0 {
		return math.INF_F64
	}
	f := f64(n_divisions) / f64(n_start)
	return cycle_hours * math.LN2 / math.ln(1.0 + f)
}

check_doubling_band :: proc(t_d, lo, hi: f64) -> bool {
	return t_d >= lo && t_d <= hi
}

Cycle_Report :: struct {
	deaths:     int,
	divisions:  int,
	n_start:    int,
	g_used:     f64, // G derived from rp (t_rep_h/t_active_h live here)
	mean_sf:    f64,
	min_ln_e:   f64,
	max_e:      f64,
	mean_ln_e:  f64,
	max_share:  f64,
	share_ok:   bool,
	doubling_h: f64,
}

// One exposure cycle (spec §4 + A1/A2/A3):
//  0. G derived from rp.t_rep_h/t_active_h (mu = ln2/t_rep), so the
//     declared T_rep drives the computation (G-C bites through it).
//  1. Share measured on the dose-assigned population BEFORE the sweep
//     (A4: the activity's share belongs to its receivers, not the
//     next cycle's survivors).
//  2. Death sweep, ascending cell id, EVERY live cell drawn exactly once
//     (A1 — unconditional, so arrested cells still die; G-Q). The draw
//     lives on RNG <= SF and dies on RNG > SF (spec §4; Correction 8:
//     at SF = 0 a draw of exactly 0.0 survives with probability 2^-53,
//     so G-Q is exact for its pinned seed, not a theorem). Uniforms
//     from s.rng (H2).
//  3. A2: every live dose resets to 0 (lived or died); SF sees only this
//     cycle's dose (G-B).
//  4. Division loop over viable cells at volume >= 2*V_target: divide
//     with lognormal drift; daughters born at dose 0 (A2).
//  5. Centers of mass reconciled; A3 resorb sweep for <2-site dyers.
//  6. Distribution + share refusal + doubling report.
response_cycle :: proc(s: ^Sim, rp: Response_Params, cycle_hours: f64) -> Cycle_Report {
	rep: Cycle_Report
	mu := math.LN2 / rp.t_rep_h
	G := g_of_t(rp.t_active_h, mu)
	rep.g_used = G
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			rep.n_start += 1
		}
	}
	// Phase 1 (A4): share on the dosed population, before the sweep.
	st0 := expression_stats(s)
	rep.max_share = st0.max_share
	rep.share_ok = st0.n <= 50 || st0.max_share <= RESP_SHARE_CAP
	// Phase 2: unconditional survival sweep (A1).
	sf_sum := 0.0
	n_drawn := 0
	for id in 1..<s.next_id {
		if !cell_alive(s, id) {
			continue
		}
		sf := sf_lq(cell_dose(s, id), G, rp.alpha, rp.beta)
		sf_sum += sf
		n_drawn += 1
		u := rng_f64(&s.rng)
		if u > sf {
			cell_set_state(s, id, CELL_DYING)
			rep.deaths += 1
		}
	}
	if n_drawn > 0 {
		rep.mean_sf = sf_sum / f64(n_drawn)
	}
	// Phase 3 (A2): dose consumed by the check.
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			cell_set_dose(s, id, 0.0)
		}
	}
	// Phase 4: division loop over viable cells at doubling volume.
	if s.params.v_target >= 1 {
		for id in 1..<s.next_id {
			if !cell_alive(s, id) {
				continue
			}
			if cell_state(s, id) != CELL_VIABLE {
				continue
			}
			if cell_volume(s, id) < 2 * s.params.v_target {
				continue
			}
			e := cell_expr(s, id)
			z1 := rng_normal(&s.rng)
			z2 := rng_normal(&s.rng)
			e_a := e * jexp_f64(rp.sigma_div * z1)
			e_b := e * jexp_f64(rp.sigma_div * z2)
			da, db := cell_divide(s, id, e_a, e_b)
			if da != 0 {
				rep.divisions += 1
			}
			_ = db
		}
	}
	update_centers_of_mass(s)
	resorb_sweep(s)
	st := expression_stats(s)
	rep.min_ln_e = math.ln(st.min_e) if st.n > 0 else 0
	rep.max_e = st.max_e
	rep.mean_ln_e = st.mean_ln_e
	rep.doubling_h = doubling_time_h(rep.divisions, rep.n_start, cycle_hours)
	return rep
}

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
RESP_MU      :: 0.4620981203732968 // ln2/1.5
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

// A4/G-H: refuse above 50x uniform share. Returns (share, ok).
check_share_cap :: proc(s: ^Sim, cap := RESP_SHARE_CAP) -> (share: f64, ok: bool) {
	st := expression_stats(s)
	if st.n == 0 {
		return 0, true
	}
	return st.max_share, st.max_share <= cap
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

// Realized population doubling time over a cycle window:
// T_d = cycle_hours * n_start / n_divisions (+Inf when nothing divided).
// The band is declared by the caller (default ±2x around 58.4 h); the
// check, not the number, is the gate.
doubling_time_h :: proc(n_divisions, n_start: int, cycle_hours: f64) -> f64 {
	if n_divisions <= 0 || n_start <= 0 {
		return math.INF_F64
	}
	return cycle_hours * f64(n_start) / f64(n_divisions)
}

check_doubling_band :: proc(t_d, lo, hi: f64) -> bool {
	return t_d >= lo && t_d <= hi
}

Cycle_Report :: struct {
	deaths:     int,
	divisions:  int,
	n_start:    int,
	mean_sf:    f64,
	min_ln_e:   f64,
	max_e:      f64,
	mean_ln_e:  f64,
	max_share:  f64,
	share_ok:   bool,
	doubling_h: f64,
}

// One exposure cycle (spec §4 + A1/A2/A3):
//  1. Death sweep, ascending cell id, EVERY live cell drawn exactly once
//     (A1 — unconditional, so arrested cells still die; G-Q). Survivors
//     keep state viable; failures go dying. Uniforms from s.rng (H2).
//  2. A2: every live dose resets to 0 (lived or died); SF sees only this
//     cycle's dose (G-B).
//  3. Division loop over viable cells at volume >= 2*V_target (requires
//     V_target >= 1): divide with lognormal drift; daughters born at
//     dose 0 (A2).
//  4. Centers of mass reconciled; A3 resorb sweep for <2-site dyers.
//  5. Distribution + share (A4) + doubling report.
// G is constant per cycle (precomputed by the caller from T_active).
response_cycle :: proc(s: ^Sim, rp: Response_Params, G, cycle_hours: f64) -> Cycle_Report {
	rep: Cycle_Report
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			rep.n_start += 1
		}
	}
	// Phase 1: unconditional survival sweep (A1).
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
	// Phase 2 (A2): dose consumed by the check.
	for id in 1..<s.next_id {
		if cell_alive(s, id) {
			cell_set_dose(s, id, 0.0)
		}
	}
	// Phase 3: division loop over viable cells at doubling volume.
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
			cell_set_state(s, id, CELL_MITOTIC)
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
	rep.max_share = st.max_share
	rep.share_ok = st.max_share <= RESP_SHARE_CAP
	rep.doubling_h = doubling_time_h(rep.divisions, rep.n_start, cycle_hours)
	return rep
}

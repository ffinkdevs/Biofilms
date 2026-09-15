package cpm

import "core:math"

// ─────────────────────────────────────────────────────────────
// Radiodialysis membrane-transport coupling (§12 of biofilms_potts.jl).
//
// Three-equation cylindrical system (Deep Research synthesis, Apr 2026):
//   (1) mobile contaminant:  dc/dt = (1/r)d/dr(r D_eff dc/dr)
//                                   - (k_ads X + k_red X_red) c + k_des s
//   (2) immobile phase:      ds/dt = (k_ads X + k_red X_red) c
//                                   - (k_des + k_loss) s
//   (3) membrane damage:      dm/dt = -k_dam Ddot m,
//                             P_eff(t) = P0 exp(alpha_P D_cum)
//
// Robin BC at r=R via ghost point; L'Hôpital at r=0. Forward Euler with
// automatic stability substepping (mirrors step_radiolysis! exactly,
// including the n_sub guard).
//
// Coupling direction is ONE-WAY, as in Julia: lattice biomass feeds the
// PDE sink (X_total/X_red, refreshed every 10 MCS), and membrane
// integrity scales the nutrient wall (update_nutrient's wall_scale).
// Nothing feeds back into Metropolis acceptance — the lattice
// trajectory is identical with coupling on or off (pinned by
// tests/coupled_test.odin).
//
// Units caveat inherited as-is: R = N/2 is passed in LATTICE units into
// cm/s slots (RADIODIALYSIS: BLOCKED in the Julia repo), and X_red
// counts one species' sites over ALL interior sites. Faithful port, not
// a fix; see the repo README.
// ─────────────────────────────────────────────────────────────

Radiolysis_Params :: struct {
	nr:      int,
	d_eff:   f64,
	k_ads:   f64,
	k_red:   f64,
	k_des:   f64,
	k_loss:  f64,
	x_total: f64, // biomass density slot (site-occupancy fraction here)
	x_red:   f64, // metal-reducer slot (S. oneidensis fraction here)
	p0:      f64,
	alpha_p: f64,
	k_dam:   f64,
	ddot_r:  f64, // dose rate at membrane (placeholder, Julia parity)
	c_ext:   f64,
	dt_rd:   f64,
}

default_radiolysis_params :: proc() -> Radiolysis_Params {
	return Radiolysis_Params{
		nr      = 40,
		d_eff   = 1e-3,
		k_ads   = 0.05,
		k_red   = 0.02,
		k_des   = 0.005,
		k_loss  = 0.001,
		x_total = 1.0,
		x_red   = 0.3,
		p0      = 0.01,
		alpha_p = 0.02,
		k_dam   = 0.005,
		ddot_r  = 1.0,
		c_ext   = 1.0,
		dt_rd   = 0.5,
	}
}

Radiolysis_State :: struct {
	r_grid: []f64, // Nr points over [0, R]
	c:      []f64, // mobile contaminant c(r)
	s:      []f64, // immobile (sorbed) phase s(r)
	m:      f64,   // membrane integrity in [0,1]
	t:      f64,   // solver time
	params: Radiolysis_Params,
}

radiolysis_init :: proc(p: Radiolysis_Params, R: f64, allocator := context.allocator) -> Radiolysis_State {
	g := make([]f64, p.nr, allocator)
	for i in 0..<p.nr {
		g[i] = R * f64(i) / f64(p.nr - 1)
	}
	return Radiolysis_State{
		r_grid = g,
		c      = make([]f64, p.nr, allocator),
		s      = make([]f64, p.nr, allocator),
		m      = 1.0,
		t      = 0.0,
		params = p,
	}
}

radiolysis_destroy :: proc(rd: ^Radiolysis_State, allocator := context.allocator) {
	delete(rd.r_grid, allocator)
	delete(rd.c, allocator)
	delete(rd.s, allocator)
}

membrane_peff :: proc(rd: ^Radiolysis_State) -> f64 {
	return rd.params.p0 * jexp_f64(rd.params.alpha_p * rd.params.ddot_r * rd.t)
}

radiolysis_step :: proc(rd: ^Radiolysis_State, dt: f64) {
	dr := rd.r_grid[1] - rd.r_grid[0]
	dt_stable := 0.4 * dr * dr / (2.0 * rd.params.d_eff)
	n_sub := 1
	if dt > dt_stable {
		n_sub = int(math.ceil(dt / dt_stable))
	}
	dt_sub := dt / f64(n_sub)
	for _ in 0..<n_sub {
		radiolysis_euler(rd, dt_sub)
	}
}

@(private)
radiolysis_euler :: proc(rd: ^Radiolysis_State, dt: f64) {
	rp := rd.params
	nr := rp.nr
	dr := rd.r_grid[1] - rd.r_grid[0]
	c := rd.c
	s := rd.s

	p_eff := rp.p0 * jexp_f64(rp.alpha_p * rp.ddot_r * rd.t)
	dm_dt := -rp.k_dam * rp.ddot_r * rd.m
	uptake := rp.k_ads * rp.x_total + rp.k_red * rp.x_red

	dc := make([]f64, nr, context.temp_allocator)
	ds := make([]f64, nr, context.temp_allocator)
	defer free_all(context.temp_allocator)

	// r = 0: L'Hôpital limit -> 2 d2c/dr2.
	dc[0] = rp.d_eff * 2.0 * (c[1] - c[0]) / (dr * dr) +
		(-uptake * c[0] + rp.k_des * s[0])

	// Interior finite-volume. NOTE: denominator is r*(dr*dr), NOT
	// (r*dr)*dr — Julia writes r_i*dr^2 and the association differs
	// by 1 ulp (verified). Do not "simplify".
	for i in 1..<(nr - 1) {
		r := rd.r_grid[i]
		diff := rp.d_eff *
			((r + 0.5*dr) * (c[i+1] - c[i]) -
			 (r - 0.5*dr) * (c[i] - c[i-1])) / (r * (dr * dr))
		dc[i] = diff + (-uptake * c[i] + rp.k_des * s[i])
	}

	// r = R: Robin BC via ghost point.
	c_ghost := c[nr-2] - 2.0 * dr * p_eff * (c[nr-1] - rp.c_ext) / rp.d_eff
	{
		i := nr - 1
		r := rd.r_grid[i]
		diff := rp.d_eff *
			((r + 0.5*dr) * (c_ghost - c[i]) -
			 (r - 0.5*dr) * (c[i] - c[i-1])) / (r * (dr * dr))
		dc[i] = diff + (-uptake * c[i] + rp.k_des * s[i])
	}

	for i in 0..<nr {
		ds[i] = uptake * c[i] - (rp.k_des + rp.k_loss) * s[i]
	}
	for i in 0..<nr {
		v := c[i] + dt * dc[i]
		rd.c[i] = v if v > 0 else 0
		v = s[i] + dt * ds[i]
		rd.s[i] = v if v > 0 else 0
	}
	m := rd.m + dt * dm_dt
	rd.m = m if m > 0 else 0
	rd.t += dt
}

// Radial-band biomass means feeding the PDE sink. Mirrors
// compute_radial_biomass + the call-site mean() collapse: X_total is the
// occupied-site fraction, X_red the S. oneidensis-site fraction, both
// over ALL interior sites (the documented basis quirk, kept).
radial_biomass_means :: proc(s: ^Sim, nr: int) -> (x_total, x_red: f64) {
	n := s.params.n
	r_total := f64(n) * 0.5
	dr := r_total / f64(nr - 1)
	total := make([]f64, nr, context.temp_allocator)
	red := make([]f64, nr, context.temp_allocator)
	counts := make([]int, nr, context.temp_allocator)
	defer free_all(context.temp_allocator)
	// Hoisted over z: radius, interior, and band index depend on (x,y)
	// only (cylinder axis = z). Per-site sums accumulate exact 1.0s, so
	// the visit order is bit-irrelevant; band vectors are identical.
	for y in 0..<n {
		for x in 0..<n {
			dx := site_coord(x) - f64(n)*0.5
			dy := site_coord(y) - f64(n)*0.5
			r := math.sqrt(dx*dx + dy*dy)
			if r > r_total {
				continue // exterior column: contributes nothing
			}
			idx := int(round_half_even(r / dr))
			if idx < 0 { idx = 0 }
			if idx > nr - 1 { idx = nr - 1 }
			for z in 0..<n {
				i := lidx(n, x, y, z)
				counts[idx] += 1
				sig := s.arena.lattice[i]
				if sig > 0 && cell_alive(s, int(sig)) {
					total[idx] += 1.0
					if cell_species(s, int(sig)) == SO {
						red[idx] += 1.0
					}
				}
			}
		}
	}
	// Unweighted mean over bands (call-site collapse in Julia).
	xt, xr := f64(0), f64(0)
	for i in 0..<nr {
		if counts[i] > 0 {
			xt += total[i] / f64(counts[i])
			xr += red[i] / f64(counts[i])
		}
	}
	return xt / f64(nr), xr / f64(nr)
}

// Project the 1D radial contaminant c(r) onto the 3D lattice. Mirrors
// radial_to_3d!: nearest-bin sample, interior sites only. Band index
// hoisted over z (see radial_biomass_means); pure writes, bit-exact.
radial_to_3d :: proc(s: ^Sim, rd: ^Radiolysis_State) {
	n := s.params.n
	nr := rd.params.nr
	r_total := rd.r_grid[nr-1]
	dr := rd.r_grid[1] - rd.r_grid[0]
	for y in 0..<n {
		for x in 0..<n {
			dx := site_coord(x) - f64(n)*0.5
			dy := site_coord(y) - f64(n)*0.5
			rr := math.sqrt(dx*dx + dy*dy)
			if rr > f64(n)*0.5 {
				continue // exterior column (r > R): untouched, as in Julia
			}
			r := rr / (f64(n)*0.5) * r_total
			idx := int(round_half_even(r / dr))
			if idx < 0 { idx = 0 }
			if idx > nr - 1 { idx = nr - 1 }
			cv := rd.c[idx]
			for z in 0..<n {
				i := lidx(n, x, y, z)
				s.arena.contaminant[i] = cv
			}
		}
	}
}

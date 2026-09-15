package tests

import "core:testing"
import "core:math"
import cpm "../cpm"

// The radiolysis coupling is ONE-WAY (lattice -> PDE, membrane -> wall
// pin on a field nothing reads). These tests pin that property: turning
// coupling on must not move a single lattice site, while the PDE state
// itself must be deterministic and follow its closed forms.

@(private)
run_coupled :: proc(p: cpm.CPM_Params, seed: u64, n_mcs: int) -> (cpm.Sim, cpm.Radiolysis_State) {
	s := cpm.sim_init(p, .AoS, seed)
	rd := cpm.radiolysis_init(cpm.default_radiolysis_params(), f64(p.n) * 0.5)
	for m in 1..=n_mcs {
		cpm.mcs_step(&s)
		if m % 10 == 1 {
			xt, xr := cpm.radial_biomass_means(&s, rd.params.nr)
			rd.params.x_total = xt
			rd.params.x_red = xr
		}
		cpm.radiolysis_step(&rd, rd.params.dt_rd)
		cpm.radial_to_3d(&s, &rd)
		cpm.update_fields(&s, f32(rd.m))
		cpm.update_centers_of_mass(&s)
	}
	return s, rd
}

@(private)
run_plain :: proc(p: cpm.CPM_Params, seed: u64, n_mcs: int) -> cpm.Sim {
	s := cpm.sim_init(p, .AoS, seed)
	for _ in 0..<n_mcs {
		cpm.mcs_step(&s)
		cpm.update_fields(&s)
		cpm.update_centers_of_mass(&s)
	}
	return s
}

@(test)
test_coupling_moves_no_sites :: proc(t: ^testing.T) {
	p := cpm.default_params()
	p.n = 20
	p.n_cells_per_species = 2
	a, rd_a := run_coupled(p, 42, 10)
	defer cpm.sim_destroy(&a)
	defer cpm.radiolysis_destroy(&rd_a)
	b := run_plain(p, 42, 10)
	defer cpm.sim_destroy(&b)
	testing.expect(t, len(a.arena.lattice) == len(b.arena.lattice))
	for i in 0..<len(a.arena.lattice) {
		if a.arena.lattice[i] != b.arena.lattice[i] {
			testing.expectf(t, false, "lattice differs at %d (coupled=%d plain=%d)",
				i, a.arena.lattice[i], b.arena.lattice[i])
			break
		}
	}
	sa := cpm.take_snapshot(&a)
	sb := cpm.take_snapshot(&b)
	for sp in 0..<cpm.N_SPECIES {
		testing.expectf(t, sa.species[sp].volume == sb.species[sp].volume,
			"sp %d volume differs coupled vs plain", sp)
	}
	testing.expectf(t, sa.pair_e == sb.pair_e, "pair energy differs coupled vs plain")
}

@(test)
test_radiolysis_closed_forms :: proc(t: ^testing.T) {
	rp := cpm.default_radiolysis_params()
	rd := cpm.radiolysis_init(rp, 10.0)
	defer cpm.radiolysis_destroy(&rd)
	for _ in 0..<100 {
		cpm.radiolysis_step(&rd, rp.dt_rd)
	}
	// Membrane integrity follows m(t) = exp(-k_dam Ddot t) up to the
	// forward-Euler truncation (~3e-4 relative here).
	want := math.exp(-rp.k_dam * rp.ddot_r * rd.t)
	testing.expectf(t, abs(rd.m - want) / want < 1e-3,
		"m=%.6f want %.6f", rd.m, want)
	// Permeability grows from baseline; wall sees contaminant; interior lags.
	testing.expect(t, cpm.membrane_peff(&rd) > rp.p0)
	testing.expect(t, rd.c[rp.nr-1] > 0)
	testing.expect(t, rd.c[0] < rd.c[rp.nr-1])
	testing.expect(t, rd.m < 1.0 && rd.m > 0.0)
}

@(test)
test_coupled_determinism :: proc(t: ^testing.T) {
	p := cpm.default_params()
	p.n = 20
	p.n_cells_per_species = 2
	a, rd_a := run_coupled(p, 7, 10)
	defer cpm.sim_destroy(&a)
	defer cpm.radiolysis_destroy(&rd_a)
	b, rd_b := run_coupled(p, 7, 10)
	defer cpm.sim_destroy(&b)
	defer cpm.radiolysis_destroy(&rd_b)
	testing.expectf(t, rd_a.m == rd_b.m && rd_a.t == rd_b.t, "RD clock diverged")
	for i in 0..<len(a.arena.contaminant) {
		if a.arena.contaminant[i] != b.arena.contaminant[i] {
			testing.expectf(t, false, "contaminant differs at %d", i)
			break
		}
	}
}

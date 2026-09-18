package tests

import "core:testing"
import cpm "../cpm"
import render "../render"

@(test)
test_layout_equivalence :: proc(t: ^testing.T) {
	// Same seed + same MCS must give identical registry statistics in
	// all three layouts. Integer state (volumes, counts) is exact;
	// float accumulations (COM, melanin mean) use tolerance.
	for s in 7..<10 {
		seed := u64(s)
		p := cpm.default_params()
		p.n = 20
		p.n_cells_per_species = 2
		p.snapshot_interval = 100
		a := cpm.sim_init(p, .AoS, seed)
		defer cpm.sim_destroy(&a)
		b := cpm.sim_init(p, .SoA, seed)
		defer cpm.sim_destroy(&b)
		c := cpm.sim_init(p, .AoSoA, seed)
		defer cpm.sim_destroy(&c)
		for _ in 0..<5 {
			cpm.mcs_step(&a); cpm.update_fields(&a); cpm.update_centers_of_mass(&a)
			cpm.mcs_step(&b); cpm.update_fields(&b); cpm.update_centers_of_mass(&b)
			cpm.mcs_step(&c); cpm.update_fields(&c); cpm.update_centers_of_mass(&c)
		}
		sa := cpm.take_snapshot(&a)
		sb := cpm.take_snapshot(&b)
		sc := cpm.take_snapshot(&c)
		for sp in 0..<cpm.N_SPECIES {
			testing.expectf(t, sa.species[sp].volume == sb.species[sp].volume,
				"seed %d sp %d AoS vs SoA volume %d != %d", seed, sp,
				sa.species[sp].volume, sb.species[sp].volume)
			testing.expectf(t, sa.species[sp].volume == sc.species[sp].volume,
				"seed %d sp %d AoS vs AoSoA volume %d != %d", seed, sp,
				sa.species[sp].volume, sc.species[sp].volume)
			testing.expectf(t, sa.species[sp].n_cells == sb.species[sp].n_cells,
				"seed %d sp %d cell count differs", seed, sp)
			testing.expectf(t, abs(sa.species[sp].mean_r - sb.species[sp].mean_r) < 1e-4,
				"seed %d sp %d mean_r differs", seed, sp)
			testing.expectf(t, abs(sa.species[sp].mean_mel - sc.species[sp].mean_mel) < 1e-4,
				"seed %d sp %d mean_mel differs", seed, sp)
		}
		testing.expect(t, cpm.count_alive(&a) == cpm.count_alive(&b))
		testing.expect(t, cpm.count_alive(&a) == cpm.count_alive(&c))

		// H1: one full dose cycle must also agree, including the three
		// response fields (dose/state/expression). Without this the test
		// above passes over a port whose new fields disagree by layout.
		rp := cpm.default_response_params()
		G := cpm.g_of_t(rp.t_active_h, cpm.RESP_MU)
		sims := [3]^cpm.Sim{&a, &b, &c}
		for sim in sims {
			cpm.assign_dose_uniform(sim, 5.0)
			_ = cpm.response_cycle(sim, rp, rp.cycle_hours)
		}
		for id in 1..<a.next_id {
			testing.expectf(t, cpm.cell_alive(&a, id) == cpm.cell_alive(&b, id) &&
				cpm.cell_alive(&a, id) == cpm.cell_alive(&c, id),
				"seed %d id %d liveness differs by layout", seed, id)
			if !cpm.cell_alive(&a, id) {
				continue
			}
			testing.expectf(t, cpm.cell_volume(&a, id) == cpm.cell_volume(&b, id) &&
				cpm.cell_volume(&a, id) == cpm.cell_volume(&c, id),
				"seed %d id %d volume differs by layout", seed, id)
			testing.expectf(t, cpm.cell_dose(&a, id) == cpm.cell_dose(&b, id) &&
				cpm.cell_dose(&a, id) == cpm.cell_dose(&c, id),
				"seed %d id %d dose differs by layout", seed, id)
			testing.expectf(t, cpm.cell_state(&a, id) == cpm.cell_state(&b, id) &&
				cpm.cell_state(&a, id) == cpm.cell_state(&c, id),
				"seed %d id %d state differs by layout", seed, id)
			testing.expectf(t, cpm.cell_expr(&a, id) == cpm.cell_expr(&b, id) &&
				cpm.cell_expr(&a, id) == cpm.cell_expr(&c, id),
				"seed %d id %d expression differs by layout", seed, id)
		}
		testing.expect(t, cpm.count_alive(&a) == cpm.count_alive(&b))
		testing.expect(t, cpm.count_alive(&a) == cpm.count_alive(&c))
		testing.expect(t, cpm.assert_no_dead_sigmas(&a))
		testing.expect(t, cpm.assert_no_dead_sigmas(&b))
		testing.expect(t, cpm.assert_no_dead_sigmas(&c))
	}
}

@(test)
test_simd_matches_scalar :: proc(t: ^testing.T) {
	p := cpm.default_params()
	p.n = 24
	p.n_cells_per_species = 2
	a := cpm.sim_init(p, .SoA, 1234)
	defer cpm.sim_destroy(&a)
	b := cpm.sim_init(p, .SoA, 1234)
	defer cpm.sim_destroy(&b)
	// Advance identically once so melanin is nonzero, then fork the
	// field path: scalar reference vs SIMD fast path.
	cpm.mcs_step(&a); cpm.mcs_step(&b)
	cpm.update_melanin_scalar(&a)
	cpm.update_melanin_simd(&b)
	maxd := f64(0)
	for i in 0..<len(a.arena.melanin) {
		d := abs(a.arena.melanin[i] - b.arena.melanin[i])
		if d > maxd { maxd = d }
	}
	testing.expectf(t, maxd < 1e-4, "melanin SIMD max diff %g", maxd)

	cpm.update_nutrient_scalar(&a)
	cpm.update_nutrient_simd(&b)
	maxd = 0
	for i in 0..<len(a.arena.nutrient) {
		d := abs(a.arena.nutrient[i] - b.arena.nutrient[i])
		if d > maxd { maxd = d }
	}
	testing.expectf(t, maxd < 1e-4, "nutrient SIMD max diff %g", maxd)
}

@(test)
test_determinism :: proc(t: ^testing.T) {
	p := cpm.default_params()
	p.n = 20
	p.n_cells_per_species = 2
	a := cpm.sim_init(p, .AoS, 99)
	defer cpm.sim_destroy(&a)
	b := cpm.sim_init(p, .AoS, 99)
	defer cpm.sim_destroy(&b)
	for _ in 0..<3 {
		cpm.mcs_step(&a); cpm.update_fields(&a); cpm.update_centers_of_mass(&a)
		cpm.mcs_step(&b); cpm.update_fields(&b); cpm.update_centers_of_mass(&b)
	}
	testing.expect(t, len(a.arena.lattice) == len(b.arena.lattice))
	for i in 0..<len(a.arena.lattice) {
		if a.arena.lattice[i] != b.arena.lattice[i] {
			testing.expectf(t, false, "lattice differs at %d", i)
			break
		}
	}
}

@(test)
test_voxel_collection :: proc(t: ^testing.T) {
	p := cpm.default_params()
	p.n = 20
	p.n_cells_per_species = 2
	s := cpm.sim_init(p, .AoS, 42)
	defer cpm.sim_destroy(&s)
	vl := render.collect_voxels(&s)
	defer render.free_voxel_list(&vl)
	testing.expect(t, vl.count > 0, "no voxels collected at init")
	// Every voxel must be interior + occupied.
	for i in 0..<vl.count {
		v := vl.items[i]
		testing.expect(t, v.species >= 0 && v.species < cpm.N_SPECIES)
	}
	// Face mask of an isolated single-site cell is all faces.
	// (Smoke check: mask is nonzero for colony surface.)
	m := render.face_mask(&s, int(vl.items[0].x), int(vl.items[0].y), int(vl.items[0].z))
	testing.expect(t, m != 0)
}

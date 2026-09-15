package cpm

import "core:math"

// Unified simulation. The lattice and float fields live in `arena` (the
// one huge allocation); the cell registry lives in exactly one of the
// three layout stores selected by `layout`. All layouts consume the same
// RNG stream in the same order, so same seed -> identical trajectory
// regardless of layout (covered by tests/layouts_test.odin).
Sim :: struct {
	params:      CPM_Params,
	layout:      Layout,
	arena:       Arena,
	J:           [8][8]f32,
	rng:         Rng,
	next_id:     int, // next fresh cell id (1-based); also = slots used
	n_slots:     int, // allocated cell slots
	current_mcs: int,
	// Exactly one of these is active:
	aos:         []Cell_AoS,
	soa:         Cells_SoA,
	aosoa:       Cells_AoSoA,
}

sim_cap_for :: proc(p: ^CPM_Params) -> int {
	// Founders + headroom for future division primitives. No scheduler
	// exists, so this never grows today; the cap keeps slot<->id stable.
	return p.n_cells_per_species * N_SPECIES + 64
}

sim_init :: proc(p: CPM_Params, layout: Layout, seed: u64, allocator := context.allocator) -> Sim {
	s: Sim
	s.params = p
	s.layout = layout
	s.arena = arena_init(p.n, allocator)
	s.J = build_J()
	s.rng = rng_init(seed)
	s.next_id = 1
	s.n_slots = sim_cap_for(&s.params)

	if layout == .AoS {
		s.aos = cells_aos_alloc(s.n_slots, allocator)
	} else if layout == .SoA {
		s.soa = cells_soa_alloc(s.n_slots, allocator)
	} else {
		s.aosoa = cells_aosoa_alloc(s.n_slots, allocator)
	}

	init_fields(&s)
	place_founders(&s)
	copy_drive(&s)
	return s
}

sim_destroy :: proc(s: ^Sim, allocator := context.allocator) {
	arena_destroy(&s.arena, allocator)
	if s.layout == .AoS {
		delete(s.aos, allocator)
	} else if s.layout == .SoA {
		cells_soa_free(&s.soa, allocator)
	} else {
		cells_aosoa_free(&s.aosoa, allocator)
	}
}

// ── cell registry access (layout-dispatched, inlined at call sites) ──

cell_alive :: proc(s: ^Sim, id: int) -> bool {
	slot := id - 1
	if slot < 0 || slot >= s.n_slots { return false }
	switch s.layout {
	case .AoS:   return s.aos[slot].alive
	case .SoA:   return s.soa.alive[slot]
	case .AoSoA:
		b, l := aosoa_loc(slot)
		return s.aosoa.blocks[b].alive[l]
	}
	return false
}

cell_species :: proc(s: ^Sim, id: int) -> int {
	slot := id - 1
	switch s.layout {
	case .AoS:   return int(s.aos[slot].species)
	case .SoA:   return int(s.soa.species[slot])
	case .AoSoA:
		b, l := aosoa_loc(slot)
		return int(s.aosoa.blocks[b].species[l])
	}
	return 0
}

cell_volume :: proc(s: ^Sim, id: int) -> i32 {
	slot := id - 1
	switch s.layout {
	case .AoS:   return s.aos[slot].volume
	case .SoA:   return s.soa.volume[slot]
	case .AoSoA:
		b, l := aosoa_loc(slot)
		return s.aosoa.blocks[b].volume[l]
	}
	return 0
}

cell_add_volume :: proc(s: ^Sim, id: int, dv: i32) {
	slot := id - 1
	switch s.layout {
	case .AoS:   s.aos[slot].volume += dv
	case .SoA:   s.soa.volume[slot] += dv
	case .AoSoA:
		b, l := aosoa_loc(slot)
		s.aosoa.blocks[b].volume[l] += dv
	}
}

// species_of mirrors Julia species_of(): MEDIUM (-1) for medium/wall/dead,
// else the 0-based species index 0..6. The -1 sentinel is required because
// species CN is 0-based index 0 — returning 0 for medium would collide.
MEDIUM :: -1

species_of :: proc(s: ^Sim, sigma: i32) -> int {
	if sigma <= 0 { return MEDIUM }
	if !cell_alive(s, int(sigma)) { return MEDIUM }
	return cell_species(s, int(sigma))
}

// J-matrix row/col: 0 = medium, 1..7 = species 0..6.
j_idx :: proc(sp: int) -> int {
	return 0 if sp == MEDIUM else sp + 1
}

cell_set_new :: proc(s: ^Sim, slot: int, species: int, volume: int, cx, cy, cz: f32) {
	switch s.layout {
	case .AoS:
		s.aos[slot] = Cell_AoS{
			species = u8(species), alive = true, volume = i32(volume),
			com_x = cx, com_y = cy, com_z = cz,
			lineage = i32(slot + 1), parent = 0, generation = 0, birth_mcs = 0,
		}
	case .SoA:
		s.soa.species[slot] = u8(species)
		s.soa.alive[slot] = true
		s.soa.volume[slot] = i32(volume)
		s.soa.com_x[slot] = cx
		s.soa.com_y[slot] = cy
		s.soa.com_z[slot] = cz
		s.soa.lineage[slot] = i32(slot + 1)
		s.soa.parent[slot] = 0
		s.soa.generation[slot] = 0
		s.soa.birth_mcs[slot] = 0
		s.soa.n_alive += 1
	case .AoSoA:
		b, l := aosoa_loc(slot)
		s.aosoa.blocks[b].species[l] = u8(species)
		s.aosoa.blocks[b].alive[l] = true
		s.aosoa.blocks[b].volume[l] = i32(volume)
		s.aosoa.blocks[b].com_x[l] = cx
		s.aosoa.blocks[b].com_y[l] = cy
		s.aosoa.blocks[b].com_z[l] = cz
		s.aosoa.blocks[b].lineage[l] = i32(slot + 1)
		s.aosoa.blocks[b].parent[l] = 0
		s.aosoa.blocks[b].generation[l] = 0
		s.aosoa.blocks[b].birth_mcs[l] = 0
		s.aosoa.n_alive += 1
	}
}

cell_kill :: proc(s: ^Sim, id: int) {
	slot := id - 1
	switch s.layout {
	case .AoS:   s.aos[slot].alive = false
	case .SoA:
		s.soa.alive[slot] = false
		s.soa.n_alive -= 1
	case .AoSoA:
		b, l := aosoa_loc(slot)
		if s.aosoa.blocks[b].alive[l] {
			s.aosoa.blocks[b].alive[l] = false
			s.aosoa.n_alive -= 1
		}
	}
}

// ── field + geometry init ──

@(private)
init_fields :: proc(s: ^Sim) {
	n := s.params.n
	R := f32(n) * 0.5
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := lidx(n, x, y, z)
				dx := f32(x) + 0.5 - R
				dy := f32(y) + 0.5 - R
				r := math.sqrt(dx*dx + dy*dy)
				if r <= R {
					s.arena.interior[i] = 1
					s.arena.lattice[i] = 0
				} else {
					s.arena.interior[i] = 0
					s.arena.lattice[i] = -1
				}
				// I(r) = I0 exp(-kappa r / N), maximal on axis.
				s.arena.radiation[i] = s.params.i0 * math.exp(-s.params.kappa * r / f32(n))
				// Nutrient gradient from wall: C0 * clamp(r/R).
				cr := r / R
				if cr > 1.0 { cr = 1.0 }
				s.arena.nutrient[i] = s.params.c_wall * cr
				s.arena.melanin[i] = 0
			}
		}
	}
}

@(private)
copy_drive :: proc(s: ^Sim) {
	// Legacy parity: melanin_drive starts identical to radiation.
	copy(s.arena.melanin_drive, s.arena.radiation)
}

@(private)
place_founders :: proc(s: ^Sim) {
	n := s.params.n
	R := f32(n) * 0.5
	for sp in 0..<N_SPECIES {
		for _ in 0..<s.params.n_cells_per_species {
			placed := false
			for attempt in 0..<100 {
				if placed { break }
				cx := rng_range(&s.rng, 4, n - 5)
				cy := rng_range(&s.rng, 4, n - 5)
				cz := rng_range(&s.rng, 4, n - 5)
				dx := f32(cx) + 0.5 - R
				dy := f32(cy) + 0.5 - R
				if math.sqrt(dx*dx+dy*dy) >= R - 4.0 { continue }
				radius := rng_range(&s.rng, 2, 3)
				id := s.next_id
				vol := place_sphere(s, i32(id), cx, cy, cz, radius)
				if vol > 0 {
					slot := id - 1
					cell_set_new(s, slot, sp, vol, f32(cx), f32(cy), f32(cz))
					s.next_id += 1
					placed = true
				}
				_ = attempt
			}
		}
	}
	update_centers_of_mass(s)
}

@(private)
place_sphere :: proc(s: ^Sim, id: i32, cx, cy, cz, radius: int) -> int {
	n := s.params.n
	count := 0
	for dz in -radius..=radius {
		for dy in -radius..=radius {
			for dx in -radius..=radius {
				if dx*dx + dy*dy + dz*dz > radius*radius { continue }
				x, y, z := cx + dx, cy + dy, cz + dz
				if !in_bounds(n, x, y, z) { continue }
				i := lidx(n, x, y, z)
				if s.arena.interior[i] == 1 && s.arena.lattice[i] == 0 {
					s.arena.lattice[i] = id
					count += 1
				}
			}
		}
	}
	return count
}

// ── Hamiltonian ──

// Pure adhesion of site (x,y,z) assuming it holds sigma_c. No
// write-restore (cf. serial Julia); identical value, race-free, and the
// form the GPU/checkerboard port needs.
site_adhesion :: proc(s: ^Sim, x, y, z: int, sigma_c: i32) -> f32 {
	n := s.params.n
	sp_c := species_of(s, sigma_c)
	E := f32(0)
	for o in NEIGHBOURS_26 {
		nx := x + int(o[0])
		ny := y + int(o[1])
		nz := z + int(o[2])
		if in_bounds(n, nx, ny, nz) {
			sn := s.arena.lattice[lidx(n, nx, ny, nz)]
			if sn != sigma_c {
				E += s.J[j_idx(sp_c)][j_idx(species_of(s, sn))]
			}
		} else {
			E += s.J[j_idx(sp_c)][0] // out of bounds = medium
		}
	}
	return E
}

compute_delta_H :: proc(s: ^Sim, sx, sy, sz, tx, ty, tz: int) -> f32 {
	n := s.params.n
	ti := lidx(n, tx, ty, tz)
	sigma_s := s.arena.lattice[lidx(n, sx, sy, sz)]
	sigma_t := s.arena.lattice[ti]

	dH := site_adhesion(s, tx, ty, tz, sigma_s) - site_adhesion(s, tx, ty, tz, sigma_t)

	// Volume: lambda*((V±1-Vt)^2-(V-Vt)^2).
	if sigma_s > 0 {
		V := cell_volume(s, int(sigma_s))
		d := V - s.params.v_target
		dH += s.params.lambda_v * f32(2*d + 1)
	}
	if sigma_t > 0 {
		V := cell_volume(s, int(sigma_t))
		d := V - s.params.v_target
		dH += s.params.lambda_v * f32(-2*d + 1)
	}

	// Radiation: beta * I gained/lost.
	I := s.arena.radiation[ti]
	if sigma_s > 0 {
		dH += BETA_ION[cell_species(s, int(sigma_s))] * I
	}
	if sigma_t > 0 {
		dH -= BETA_ION[cell_species(s, int(sigma_t))] * I
	}

	// Melanin: radiotropic species prefer melanin-rich sites.
	M := s.arena.melanin[ti]
	if sigma_s > 0 && is_radiotropic(cell_species(s, int(sigma_s))) {
		dH -= 0.5 * M
	}
	if sigma_t > 0 && is_radiotropic(cell_species(s, int(sigma_t))) {
		dH += 0.5 * M
	}
	return dH
}

// ── Monte Carlo step: N^3 copy attempts ──

mcs_step :: proc(s: ^Sim) {
	n := s.params.n
	n_attempts := n * n * n
	s.current_mcs += 1
	for _ in 0..<n_attempts {
		sx := rng_below(&s.rng, n)
		sy := rng_below(&s.rng, n)
		sz := rng_below(&s.rng, n)
		if s.arena.interior[lidx(n, sx, sy, sz)] == 0 { continue }
		o := NEIGHBOURS_26[rng_below(&s.rng, 26)]
		tx := sx + int(o[0])
		ty := sy + int(o[1])
		tz := sz + int(o[2])
		if !in_bounds(n, tx, ty, tz) { continue }
		if s.arena.interior[lidx(n, tx, ty, tz)] == 0 { continue }
		si := lidx(n, sx, sy, sz)
		ti := lidx(n, tx, ty, tz)
		sig_s := s.arena.lattice[si]
		sig_t := s.arena.lattice[ti]
		if sig_s == sig_t { continue }
		if sig_s <= 0 && sig_t <= 0 { continue }

		dH := compute_delta_H(s, sx, sy, sz, tx, ty, tz)
		accept := false
		if dH <= 0 {
			accept = true
		} else {
			accept = rng_f32(&s.rng) < math.exp(-dH / s.params.t_cpm)
		}
		if accept {
			if sig_t > 0 && cell_alive(s, int(sig_t)) {
				cell_add_volume(s, int(sig_t), -1)
			}
			if sig_s > 0 && cell_alive(s, int(sig_s)) {
				cell_add_volume(s, int(sig_s), +1)
			}
			s.arena.lattice[ti] = sig_s
		}
	}
	// Reap zero-volume cells (archived as died in Julia; here marked dead).
	for id in 1..<s.next_id {
		if cell_alive(s, id) && cell_volume(s, id) <= 0 {
			cell_kill(s, id)
		}
	}
}

// ── coupled fields (scalar reference; SIMD fast path in fields_simd.odin) ──

laplacian_6 :: proc(f: []f32, n, x, y, z: int) -> f32 {
	i := lidx(n, x, y, z)
	v := f[i]
	L := f32(0)
	if x > 0     { L += f[i - 1] - v }
	if x < n - 1 { L += f[i + 1] - v }
	if y > 0     { L += f[i - n] - v }
	if y < n - 1 { L += f[i + n] - v }
	if z > 0     { L += f[i - n*n] - v }
	if z < n - 1 { L += f[i + n*n] - v }
	return L
}

update_melanin_scalar :: proc(s: ^Sim) {
	n := s.params.n
	dt := s.params.dt_field
	src := s.arena.melanin
	dst := s.arena.scratch
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := lidx(n, x, y, z)
				if s.arena.interior[i] == 0 {
					dst[i] = src[i]
					continue
				}
				diff := s.params.d_m * laplacian_6(src, n, x, y, z)
				alpha := f32(0)
				sig := s.arena.lattice[i]
				if sig > 0 && cell_alive(s, int(sig)) {
					sp := cell_species(s, int(sig))
					// n_RF indicator: 1 iff occupant is a melanin producer.
					if is_melanin_producer(sp) {
						alpha = ALPHA_M[sp]
					}
				}
				prod := alpha * s.arena.melanin_drive[i]
				v := src[i] + dt * (diff + prod)
				dst[i] = v if v > 0 else 0
			}
		}
	}
	copy(src, dst)
}

update_nutrient_scalar :: proc(s: ^Sim, wall_scale := f32(1.0)) {
	n := s.params.n
	dt := s.params.dt_field
	src := s.arena.nutrient
	dst := s.arena.scratch
	wall := s.params.c_wall * wall_scale
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := lidx(n, x, y, z)
				if s.arena.interior[i] == 0 {
					dst[i] = wall
					continue
				}
				diff := s.params.d_c * laplacian_6(src, n, x, y, z)
				u := f32(0)
				sig := s.arena.lattice[i]
				if sig > 0 && cell_alive(s, int(sig)) {
					u = UPTAKE[cell_species(s, int(sig))]
				}
				v := src[i] + dt * (diff - u)
				dst[i] = v if v > 0 else 0
			}
		}
	}
	copy(src, dst)
}

// Default field update: SIMD fast path (matches scalar to ~1e-6; the
// scalar procs above remain the reference for tests).
update_fields :: proc(s: ^Sim, wall_scale := f32(1.0)) {
	update_melanin_simd(s)
	update_nutrient_simd(s, wall_scale)
}

// ── centers of mass + exact volume reconciliation ──

update_centers_of_mass :: proc(s: ^Sim) {
	n := s.params.n
	// Accumulators sized by slot count (small: <200). Stack-friendly.
	// We reuse three dense slices via scratch reinterpretation is
	// overkill; plain loops over the lattice with per-layout stores.
	switch s.layout {
	case .AoS:
		// Zero accumulators
		sx := make([]f64, s.n_slots, context.temp_allocator)
		sy := make([]f64, s.n_slots, context.temp_allocator)
		sz := make([]f64, s.n_slots, context.temp_allocator)
		cn := make([]int, s.n_slots, context.temp_allocator)
		for z in 0..<n {
			for y in 0..<n {
				for x in 0..<n {
					sig := s.arena.lattice[lidx(n, x, y, z)]
					if sig > 0 && s.aos[int(sig)-1].alive {
						slot := int(sig) - 1
						sx[slot] += f64(x)
						sy[slot] += f64(y)
						sz[slot] += f64(z)
						cn[slot] += 1
					}
				}
			}
		}
		for slot in 0..<(s.next_id - 1) {
			if !s.aos[slot].alive { continue }
			if cn[slot] > 0 {
				s.aos[slot].com_x = f32(sx[slot] / f64(cn[slot]))
				s.aos[slot].com_y = f32(sy[slot] / f64(cn[slot]))
				s.aos[slot].com_z = f32(sz[slot] / f64(cn[slot]))
				s.aos[slot].volume = i32(cn[slot])
			} else {
				s.aos[slot].alive = false
			}
		}
		free_all(context.temp_allocator)
	case .SoA:
		sx := make([]f64, s.n_slots, context.temp_allocator)
		sy := make([]f64, s.n_slots, context.temp_allocator)
		sz := make([]f64, s.n_slots, context.temp_allocator)
		cn := make([]int, s.n_slots, context.temp_allocator)
		for z in 0..<n {
			for y in 0..<n {
				for x in 0..<n {
					sig := s.arena.lattice[lidx(n, x, y, z)]
					if sig > 0 {
						slot := int(sig) - 1
						if slot < s.n_slots && s.soa.alive[slot] {
							sx[slot] += f64(x)
							sy[slot] += f64(y)
							sz[slot] += f64(z)
							cn[slot] += 1
						}
					}
				}
			}
		}
		for slot in 0..<(s.next_id - 1) {
			if !s.soa.alive[slot] { continue }
			if cn[slot] > 0 {
				s.soa.com_x[slot] = f32(sx[slot] / f64(cn[slot]))
				s.soa.com_y[slot] = f32(sy[slot] / f64(cn[slot]))
				s.soa.com_z[slot] = f32(sz[slot] / f64(cn[slot]))
				s.soa.volume[slot] = i32(cn[slot])
			} else {
				s.soa.alive[slot] = false
				s.soa.n_alive -= 1
			}
		}
		free_all(context.temp_allocator)
	case .AoSoA:
		sx := make([]f64, s.n_slots, context.temp_allocator)
		sy := make([]f64, s.n_slots, context.temp_allocator)
		sz := make([]f64, s.n_slots, context.temp_allocator)
		cn := make([]int, s.n_slots, context.temp_allocator)
		for z in 0..<n {
			for y in 0..<n {
				for x in 0..<n {
					sig := s.arena.lattice[lidx(n, x, y, z)]
					if sig > 0 {
						slot := int(sig) - 1
						b, l := aosoa_loc(slot)
						if s.aosoa.blocks[b].alive[l] {
							sx[slot] += f64(x)
							sy[slot] += f64(y)
							sz[slot] += f64(z)
							cn[slot] += 1
						}
					}
				}
			}
		}
		for slot in 0..<(s.next_id - 1) {
			b, l := aosoa_loc(slot)
			if !s.aosoa.blocks[b].alive[l] { continue }
			if cn[slot] > 0 {
				s.aosoa.blocks[b].com_x[l] = f32(sx[slot] / f64(cn[slot]))
				s.aosoa.blocks[b].com_y[l] = f32(sy[slot] / f64(cn[slot]))
				s.aosoa.blocks[b].com_z[l] = f32(sz[slot] / f64(cn[slot]))
				s.aosoa.blocks[b].volume[l] = i32(cn[slot])
			} else {
				s.aosoa.blocks[b].alive[l] = false
				s.aosoa.n_alive -= 1
			}
		}
		free_all(context.temp_allocator)
	}
}

// ── snapshots ──

Species_Stats :: struct {
	volume:   int,
	n_cells:  int,
	mean_r:   f32, // lattice units from z-axis
	mean_mel: f32, // mean melanin over occupied sites
}

Snapshot :: struct {
	mcs:     int,
	species: [N_SPECIES]Species_Stats,
	pair_e:  f32, // mutualistic pairwise energy (diagnostic; never enters dH)
}

take_snapshot :: proc(s: ^Sim) -> Snapshot {
	snap: Snapshot
	snap.mcs = s.current_mcs
	n := s.params.n
	R := f32(n) * 0.5
	// Per-cell COM radial accumulation.
	for id in 1..<s.next_id {
		if !cell_alive(s, id) { continue }
		sp := cell_species(s, id)
		cx, cy, _ := cell_com(s, id)
		dx := cx + 0.5 - R
		dy := cy + 0.5 - R
		r := math.sqrt(dx*dx + dy*dy)
		snap.species[sp].mean_r += r
		snap.species[sp].n_cells += 1
		snap.species[sp].volume += int(cell_volume(s, id))
	}
	for sp in 0..<N_SPECIES {
		if snap.species[sp].n_cells > 0 {
			snap.species[sp].mean_r /= f32(snap.species[sp].n_cells)
		}
	}
	// Mean melanin over occupied sites per species.
	mel_sum: [N_SPECIES]f64
	mel_cnt: [N_SPECIES]int
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := lidx(n, x, y, z)
				sig := s.arena.lattice[i]
				if sig > 0 && cell_alive(s, int(sig)) {
					sp := cell_species(s, int(sig))
					mel_sum[sp] += f64(s.arena.melanin[i])
					mel_cnt[sp] += 1
				}
			}
		}
	}
	for sp in 0..<N_SPECIES {
		if mel_cnt[sp] > 0 {
			snap.species[sp].mean_mel = f32(mel_sum[sp] / f64(mel_cnt[sp]))
		}
	}
	snap.pair_e = total_pairwise_energy(s)
	return snap
}

cell_com :: proc(s: ^Sim, id: int) -> (x, y, z: f32) {
	slot := id - 1
	switch s.layout {
	case .AoS:   return s.aos[slot].com_x, s.aos[slot].com_y, s.aos[slot].com_z
	case .SoA:   return s.soa.com_x[slot], s.soa.com_y[slot], s.soa.com_z[slot]
	case .AoSoA:
		b, l := aosoa_loc(slot)
		return s.aosoa.blocks[b].com_x[l], s.aosoa.blocks[b].com_y[l], s.aosoa.blocks[b].com_z[l]
	}
	return 0, 0, 0
}

// Mutualistic species pairs (Table 2 context; mirrors MUTUALISTIC_PAIRS).
// V_ij = -gamma*exp(-r^2/sigma^2) between cell centres. Diagnostic only:
// computed once per snapshot, never enters the Metropolis test.
is_mutualistic :: proc(sa, sb: int) -> bool {
	lo, hi := sa, sb
	if lo > hi { lo, hi = hi, lo }
	return (lo == CN && hi == DR) ||
	       (lo == CN && hi == CS) ||
	       (lo == CS && hi == AN) ||
	       (lo == SO && hi == OI) ||
	       (lo == DR && hi == BS)
}

total_pairwise_energy :: proc(s: ^Sim) -> f32 {
	E := f32(0)
	g2 := s.params.gamma_mutual
	sig2 := s.params.sigma_mutual * s.params.sigma_mutual
	for a in 1..<s.next_id {
		if !cell_alive(s, a) { continue }
		sa := cell_species(s, a)
		ax, ay, az := cell_com(s, a)
		for b in (a + 1)..<s.next_id {
			if !cell_alive(s, b) { continue }
			sb := cell_species(s, b)
			if !is_mutualistic(sa, sb) { continue }
			bx, by, bz := cell_com(s, b)
			dx, dy, dz := ax - bx, ay - by, az - bz
			E += -g2 * math.exp(-(dx*dx + dy*dy + dz*dz) / sig2)
		}
	}
	return E
}

count_alive :: proc(s: ^Sim) -> int {
	total := 0
	for id in 1..<s.next_id {
		if cell_alive(s, id) { total += 1 }
	}
	return total
}

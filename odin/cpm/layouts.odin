package cpm

// ─────────────────────────────────────────────────────────────
// Cell layouts: AoS (canonical), SoA (canonical), AoSoA (blocked).
//
// Cell IDs are 1-based on the lattice (0=medium, -1=wall) and index
// into these stores as id-1. `alive` is false for erased slots; the
// slot is reused by the next placed cell only at init (no division
// scheduler exists — divide_cell has no trigger, matching Julia).
// ─────────────────────────────────────────────────────────────

// Array of Structures: one struct per cell, slice of structs.
// Best for: random per-cell access (Metropolis volume updates, COM
// recompute), code clarity. Worst for: vectorised per-field sweeps.
Cell_AoS :: struct {
	species:    u8,
	alive:      bool,
	volume:     i32,
	com_x:      f64,
	com_y:      f64,
	com_z:      f64,
	lineage:    i32,
	parent:     i32,
	generation: i32,
	birth_mcs:  i32,
	dose:       f64, // accumulated Gy this cycle (reset by survival check, A2)
	state:      u8,  // CELL_* lifecycle state (1 viable default)
	expr:       f64, // heritable expression trait (1.0 default)
}

// Structure of Arrays: one array per field, uniform stride.
// Best for: streaming sweeps over a single field (volume-only pass,
// species-only gather). Idiomatic Odin: plain struct of slices, all
// slices share one backing length and are freed together.
Cells_SoA :: struct {
	species:    []u8,
	alive:      []bool,
	volume:     []i32,
	com_x:      []f64,
	com_y:      []f64,
	com_z:      []f64,
	lineage:    []i32,
	parent:     []i32,
	generation: []i32,
	birth_mcs:  []i32,
	dose:       []f64,
	state:      []u8,
	expr:       []f64,
	count:      int, // live+dead slots; live count tracked separately
	n_alive:    int,
}

// Array of Structures of Arrays: blocked SoA, BLOCK cells per block.
// BLOCK=8 matches the AVX2 #simd[4]f64 field width (two vectors) and the
// Vulkan subgroup size used in shaders/field_diffusion.comp. Within a
// field is a fixed array, so a block fits in a few cache lines and
// vectorises cleanly; across blocks it streams like SoA.
// Padding lanes (beyond n_alive) have alive=false and volume=0.
AOSOA_BLOCK :: 8

Cell_Block :: struct {
	species:    [AOSOA_BLOCK]u8,
	alive:      [AOSOA_BLOCK]bool,
	volume:     [AOSOA_BLOCK]i32,
	com_x:      [AOSOA_BLOCK]f64,
	com_y:      [AOSOA_BLOCK]f64,
	com_z:      [AOSOA_BLOCK]f64,
	lineage:    [AOSOA_BLOCK]i32,
	parent:     [AOSOA_BLOCK]i32,
	generation: [AOSOA_BLOCK]i32,
	birth_mcs:  [AOSOA_BLOCK]i32,
	dose:       [AOSOA_BLOCK]f64,
	state:      [AOSOA_BLOCK]u8,
	expr:       [AOSOA_BLOCK]f64,
}

Cells_AoSoA :: struct {
	blocks:   []Cell_Block,
	cap:      int, // cells (lanes), not blocks
	n_alive:  int,
}

Layout :: enum {
	AoS,
	SoA,
	AoSoA,
}

// ── allocation ──

cells_aos_alloc :: proc(cap: int, allocator := context.allocator) -> []Cell_AoS {
	cells := make([]Cell_AoS, cap, allocator)
	return cells
}

cells_soa_alloc :: proc(cap: int, allocator := context.allocator) -> Cells_SoA {
	return Cells_SoA{
		species    = make([]u8, cap, allocator),
		alive      = make([]bool, cap, allocator),
		volume     = make([]i32, cap, allocator),
		com_x      = make([]f64, cap, allocator),
		com_y      = make([]f64, cap, allocator),
		com_z      = make([]f64, cap, allocator),
		lineage    = make([]i32, cap, allocator),
		parent     = make([]i32, cap, allocator),
		generation = make([]i32, cap, allocator),
		birth_mcs  = make([]i32, cap, allocator),
		dose       = make([]f64, cap, allocator),
		state      = make([]u8, cap, allocator),
		expr       = make([]f64, cap, allocator),
		count      = cap,
	}
}

cells_aosoa_alloc :: proc(cap_cells: int, allocator := context.allocator) -> Cells_AoSoA {
	nb := (cap_cells + AOSOA_BLOCK - 1) / AOSOA_BLOCK
	return Cells_AoSoA{blocks = make([]Cell_Block, nb, allocator), cap = nb * AOSOA_BLOCK}
}

cells_soa_free :: proc(s: ^Cells_SoA, allocator := context.allocator) {
	delete(s.species, allocator)
	delete(s.alive, allocator)
	delete(s.volume, allocator)
	delete(s.com_x, allocator)
	delete(s.com_y, allocator)
	delete(s.com_z, allocator)
	delete(s.lineage, allocator)
	delete(s.parent, allocator)
	delete(s.generation, allocator)
	delete(s.birth_mcs, allocator)
	delete(s.dose, allocator)
	delete(s.state, allocator)
	delete(s.expr, allocator)
	s.count = 0
}

cells_aosoa_free :: proc(s: ^Cells_AoSoA, allocator := context.allocator) {
	delete(s.blocks, allocator)
	s.cap = 0
}

// Grow all three registries by doubling slots. New slots are zero-valued
// (dead, dose 0, expression 0 — set explicitly on use). Uses
// context.allocator, so callers must grow and destroy under the same
// allocator they allocated with (true for sim_init/sim_destroy/tests,
// which all use the default). Needed by cell_divide: every division is
// net +1 slot and a dividing sim outgrows its founder cap within cycles.
grow_registry_aos :: proc(cells: ^[]Cell_AoS) {
	nb := make([]Cell_AoS, len(cells) * 2)
	copy(nb, cells^)
	delete(cells^)
	cells^ = nb
}

grow_registry_soa :: proc(s: ^Cells_SoA) {
	grow :: proc(dst: ^[]f64, n: int) {
		nb := make([]f64, n * 2)
		copy(nb, dst^)
		delete(dst^)
		dst^ = nb
	}
	gi32 :: proc(dst: ^[]i32, n: int) {
		nb := make([]i32, n * 2)
		copy(nb, dst^)
		delete(dst^)
		dst^ = nb
	}
	gu8 :: proc(dst: ^[]u8, n: int) {
		nb := make([]u8, n * 2)
		copy(nb, dst^)
		delete(dst^)
		dst^ = nb
	}
	gbool :: proc(dst: ^[]bool, n: int) {
		nb := make([]bool, n * 2)
		copy(nb, dst^)
		delete(dst^)
		dst^ = nb
	}
	n := s.count
	gu8(&s.species, n)
	gbool(&s.alive, n)
	gi32(&s.volume, n)
	grow(&s.com_x, n)
	grow(&s.com_y, n)
	grow(&s.com_z, n)
	gi32(&s.lineage, n)
	gi32(&s.parent, n)
	gi32(&s.generation, n)
	gi32(&s.birth_mcs, n)
	grow(&s.dose, n)
	gu8(&s.state, n)
	grow(&s.expr, n)
	s.count = n * 2
}

grow_registry_aosoa :: proc(s: ^Cells_AoSoA) {
	nb := make([]Cell_Block, len(s.blocks) * 2)
	copy(nb, s.blocks[:])
	delete(s.blocks)
	s.blocks = nb
	s.cap = len(nb) * AOSOA_BLOCK
}

// ── lane accessors (uniform 0-based cell-slot index) ──

aos_get :: #force_inline proc(cells: []Cell_AoS, slot: int) -> Cell_AoS {
	return cells[slot]
}

soa_species :: #force_inline proc(s: ^Cells_SoA, slot: int) -> u8 {
	return s.species[slot]
}

aosoa_loc :: #force_inline proc(slot: int) -> (block: int, lane: int) {
	return slot / AOSOA_BLOCK, slot % AOSOA_BLOCK
}

aosoa_species :: proc(s: ^Cells_AoSoA, slot: int) -> u8 {
	b, l := aosoa_loc(slot)
	return s.blocks[b].species[l]
}

aosoa_alive :: #force_inline proc(s: ^Cells_AoSoA, slot: int) -> bool {
	b, l := aosoa_loc(slot)
	return s.blocks[b].alive[l]
}

aosoa_volume :: #force_inline proc(s: ^Cells_AoSoA, slot: int) -> i32 {
	b, l := aosoa_loc(slot)
	return s.blocks[b].volume[l]
}

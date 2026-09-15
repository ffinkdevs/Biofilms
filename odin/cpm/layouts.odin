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
	com_x:      f32,
	com_y:      f32,
	com_z:      f32,
	lineage:    i32,
	parent:     i32,
	generation: i32,
	birth_mcs:  i32,
}

// Structure of Arrays: one array per field, uniform stride.
// Best for: streaming sweeps over a single field (volume-only pass,
// species-only gather). Idiomatic Odin: plain struct of slices, all
// slices share one backing length and are freed together.
Cells_SoA :: struct {
	species:    []u8,
	alive:      []bool,
	volume:     []i32,
	com_x:      []f32,
	com_y:      []f32,
	com_z:      []f32,
	lineage:    []i32,
	parent:     []i32,
	generation: []i32,
	birth_mcs:  []i32,
	count:      int, // live+dead slots; live count tracked separately
	n_alive:    int,
}

// Array of Structures of Arrays: blocked SoA, BLOCK cells per block.
// BLOCK=8 matches AVX2 f32x8 / AVX-512 halves and the Vulkan subgroup
// size used in shaders/field_diffusion.comp. Within a block every
// field is a fixed array, so a block fits in a few cache lines and
// vectorises cleanly; across blocks it streams like SoA.
// Padding lanes (beyond n_alive) have alive=false and volume=0.
AOSOA_BLOCK :: 8

Cell_Block :: struct {
	species:    [AOSOA_BLOCK]u8,
	alive:      [AOSOA_BLOCK]bool,
	volume:     [AOSOA_BLOCK]i32,
	com_x:      [AOSOA_BLOCK]f32,
	com_y:      [AOSOA_BLOCK]f32,
	com_z:      [AOSOA_BLOCK]f32,
	lineage:    [AOSOA_BLOCK]i32,
	parent:     [AOSOA_BLOCK]i32,
	generation: [AOSOA_BLOCK]i32,
	birth_mcs:  [AOSOA_BLOCK]i32,
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
		com_x      = make([]f32, cap, allocator),
		com_y      = make([]f32, cap, allocator),
		com_z      = make([]f32, cap, allocator),
		lineage    = make([]i32, cap, allocator),
		parent     = make([]i32, cap, allocator),
		generation = make([]i32, cap, allocator),
		birth_mcs  = make([]i32, cap, allocator),
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
	s.count = 0
}

cells_aosoa_free :: proc(s: ^Cells_AoSoA, allocator := context.allocator) {
	delete(s.blocks, allocator)
	s.cap = 0
}

// ── lane accessors (uniform 0-based cell-slot index) ──

aos_get :: proc(cells: []Cell_AoS, slot: int) -> Cell_AoS {
	return cells[slot]
}

soa_species :: proc(s: ^Cells_SoA, slot: int) -> u8 {
	return s.species[slot]
}

aosoa_loc :: proc(slot: int) -> (block: int, lane: int) {
	return slot / AOSOA_BLOCK, slot % AOSOA_BLOCK
}

aosoa_species :: proc(s: ^Cells_AoSoA, slot: int) -> u8 {
	b, l := aosoa_loc(slot)
	return s.blocks[b].species[l]
}

aosoa_alive :: proc(s: ^Cells_AoSoA, slot: int) -> bool {
	b, l := aosoa_loc(slot)
	return s.blocks[b].alive[l]
}

aosoa_volume :: proc(s: ^Cells_AoSoA, slot: int) -> i32 {
	b, l := aosoa_loc(slot)
	return s.blocks[b].volume[l]
}

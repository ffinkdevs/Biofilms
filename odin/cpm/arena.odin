package cpm

import "core:mem"

// One huge allocation for compute/render/bindless.
//
// Layout inside `backing` (all 64-byte aligned, offsets recorded so the
// same layout maps 1:1 onto a single Vulkan VkBuffer with push-constant
// offsets — see shaders/design.md):
//
//   [lattice i32 N^3][interior u8 N^3][radiation f64][melanin f64][nutrient f64]
//   [melanin_drive f64][contaminant f64][scratch f64 N^3]
//
// Rationale: a single allocation means one map/unmap, one barrier, one
// descriptor (bindless index 0). The CPM lattice stays host-visible;
// the float fields are device-friendly (f64, 64B-aligned rows for AVX).
//Interior is u8 (not bit-packed) so GPU threads can byte-load without
// bit unpacking; the 8x memory cost is negligible vs the float fields.
Arena :: struct {
	backing:       []u8,   // the one allocation; free with arena_destroy
	n:             int,
	n3:            int,
	lattice:       []i32,  // N^3,  0=medium, -1=wall, >0=cell id
	interior:      []u8,   // N^3,  1=inside cylinder
	radiation:     []f64,  // static Beer-Lambert field (Hamiltonian signal)
	melanin:       []f64,  // reaction-diffusion field
	melanin_drive: []f64,  // copy of radiation at init (legacy parity)
	nutrient:      []f64,  // wall-fed gradient, consumed by cells
	contaminant:   []f64,  // mobile contaminant c(r) projected to 3D (coupled mode)
	scratch:       []f64,  // double-buffer target for field updates
	// Byte offsets of each region inside backing (for Vulkan parity).
	off_lattice:   int,
	off_interior:  int,
	off_radiation: int,
	off_melanin:   int,
	off_drive:     int,
	off_nutrient:  int,
	off_contam:    int,
	off_scratch:   int,
	total_bytes:   int,
}

ALIGN :: 64

@(private)
align_up :: proc(v, a: int) -> int {
	return (v + a - 1) &~ (a - 1)
}

arena_bytes_required :: proc(n: int) -> int {
	n3 := n * n * n
	total := 0
	total = align_up(total, ALIGN); total += n3 * size_of(i32)
	total = align_up(total, ALIGN); total += n3 * size_of(u8)
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // radiation
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // melanin
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // drive
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // nutrient
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // contaminant
	total = align_up(total, ALIGN); total += n3 * size_of(f64) // scratch
	total = align_up(total, ALIGN)
	return total
}

arena_init :: proc(n: int, allocator := context.allocator) -> Arena {
	a: Arena
	a.n = n
	a.n3 = n * n * n
	a.total_bytes = arena_bytes_required(n)
	raw, err := mem.alloc(a.total_bytes + ALIGN, ALIGN, allocator)
	assert(err == .None, "arena_init: out of memory")
	// Zero everything: lattice starts as medium, fields as 0.
	mem.zero(raw, a.total_bytes + ALIGN)
	base := int(uintptr(raw))
	off := 0

	off = align_up(off, ALIGN); a.off_lattice = off
	a.lattice = mem.slice_ptr((^i32)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(i32)

	off = align_up(off, ALIGN); a.off_interior = off
	a.interior = mem.slice_ptr((^u8)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(u8)

	off = align_up(off, ALIGN); a.off_radiation = off
	a.radiation = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	off = align_up(off, ALIGN); a.off_melanin = off
	a.melanin = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	off = align_up(off, ALIGN); a.off_drive = off
	a.melanin_drive = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	off = align_up(off, ALIGN); a.off_nutrient = off
	a.nutrient = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	off = align_up(off, ALIGN); a.off_contam = off
	a.contaminant = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	off = align_up(off, ALIGN); a.off_scratch = off
	a.scratch = mem.slice_ptr((^f64)(uintptr(base + off)), a.n3)
	off += a.n3 * size_of(f64)

	a.backing = mem.slice_ptr((^u8)(uintptr(base)), a.total_bytes + ALIGN)
	// Keep the raw base for free(); backing[0] == raw[0].
	return a
}

arena_destroy :: proc(a: ^Arena, allocator := context.allocator) {
	if a.backing != nil {
		mem.free(raw_data(a.backing), allocator)
		a.backing = nil
	}
}

// Linear index, x fastest (matches Julia's x,y,z loop order logically;
// Julia is column-major but we never compare raw buffers cross-language,
// only CSV statistics, so layout parity is intentionally NOT claimed).
lidx :: #force_inline proc(n, x, y, z: int) -> int {
	return x + n * (y + n * z)
}

in_bounds :: #force_inline proc(n, x, y, z: int) -> bool {
	return x >= 0 && y >= 0 && z >= 0 && x < n && y < n && z < n
}

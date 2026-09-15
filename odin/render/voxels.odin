package render

import cpm "../cpm"

// Voxel instance list: one entry per occupied lattice site.
// Built fresh each frame from the CPM lattice (N=40 -> worst case
// ~30k occupied sites; typical ~5-8k). Positions are lattice-centered
// floats so the viewer can orbit without re-quantising.
Voxel :: struct {
	x, y, z: f32, // lattice coords, voxel centre
	species:  int,  // 0..6
}

Voxel_List :: struct {
	items: []Voxel,
	count: int,
}

// collect_voxels gathers every occupied interior site. Caller owns the
// returned slice (made with context.allocator by default).
collect_voxels :: proc(s: ^cpm.Sim, allocator := context.allocator) -> Voxel_List {
	n := s.params.n
	cap_est := 8192
	items := make([]Voxel, cap_est, allocator)
	count := 0
	append_voxel :: proc(items: ^[]Voxel, count: ^int, v: Voxel, allocator := context.allocator) {
		if count^ >= len(items) {
			new_cap := len(items) * 2
			nbuf := make([]Voxel, new_cap, allocator)
			copy(nbuf, items[:count^])
			delete(items^, allocator)
			items^ = nbuf
		}
		items[count^] = v
		count^ += 1
	}
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := cpm.lidx(n, x, y, z)
				sig := s.arena.lattice[i]
				if sig > 0 && cpm.cell_alive(s, int(sig)) {
					sp := cpm.cell_species(s, int(sig))
					append_voxel(&items, &count,
						Voxel{f32(x), f32(y), f32(z), sp}, allocator)
				}
			}
		}
	}
	return Voxel_List{items = items, count = count}
}

free_voxel_list :: proc(v: ^Voxel_List, allocator := context.allocator) {
	delete(v.items, allocator)
	v.count = 0
}

// Face-visibility test for a filled-voxel renderer: a cube face is drawn
// only if the neighbour in that direction is empty (medium) or out of
// bounds. Wall neighbours count as empty (colony surface at the
// membrane is still rendered). Returns a 6-bit mask (+x,-x,+y,-y,+z,-z).
face_mask :: proc(s: ^cpm.Sim, x, y, z: int) -> u8 {
	n := s.params.n
	mask: u8 = 0
	occ :: proc(s: ^cpm.Sim, x, y, z: int) -> bool {
		n := s.params.n
		if x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n {
			return false
		}
		sig := s.arena.lattice[cpm.lidx(n, x, y, z)]
		return sig > 0 && cpm.cell_alive(s, int(sig))
	}
	if !occ(s, x + 1, y, z) { mask |= 1 }
	if !occ(s, x - 1, y, z) { mask |= 2 }
	if !occ(s, x, y + 1, z) { mask |= 4 }
	if !occ(s, x, y - 1, z) { mask |= 8 }
	if !occ(s, x, y, z + 1) { mask |= 16 }
	if !occ(s, x, y, z - 1) { mask |= 32 }
	return mask
}

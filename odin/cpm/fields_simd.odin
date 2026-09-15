package cpm

import "base:intrinsics"
import "core:simd"

// SIMD field updates: the vectorised fast path for the two
// reaction-diffusion sweeps. Scalar reference lives in sim.odin.
//
// What vectorises: the 6-point Laplacian over contiguous x-runs
// (#simd[8]f32 = AVX2 width, also the AoSoA block width and the Vulkan
// subgroup size, so one width everywhere). What does NOT vectorise:
// the Metropolis copy attempts — random gather/scatter, branchy
// Metropolis test, per-site adhesion over 26 neighbours. That part
// stays scalar on CPU by design (see shaders/design.md).
//
// Precision note: SIMD reassociates the Laplacian sum, so results
// agree with the scalar path to ~1e-6, not bit-identically. Layout
// equivalence (AoS vs SoA vs AoSoA) IS exact — integer paths are
// shared. Tests assert tolerance, not bitwise equality.

W8 :: 8
Vec8 :: #simd[W8]f32

update_melanin_simd :: proc(s: ^Sim) {
	n := s.params.n
	dt := s.params.dt_field
	dm := s.params.d_m
	src := s.arena.melanin
	dst := s.arena.scratch
	drive := s.arena.melanin_drive
	lat := s.arena.lattice
	interior := s.arena.interior
	n2 := n * n

	for z in 0..<n {
		for y in 0..<n {
			x := 0
			// Scalar prologue until 8-aligned AND safely interior-adjacent.
			// Alignment is opportunistic: rows are 64B-aligned when n is
			// even (N=40 -> row = 160B), so most chunks are aligned; we
			// do unaligned loads anyway (MOVUPS) and skip align prologue.
			for x < n {
				// Fast vector chunk: needs x-1..x+8 valid and y,z interior
				// neighbours present. Edges fall through to scalar.
				if x >= 1 && x + W8 - 1 < n - 1 && y >= 1 && y < n - 1 && z >= 1 && z < n - 1 {
					i := lidx(n, x, y, z)
					// Quick reject: if the whole chunk + halo is interior,
					// take the vector path; else scalar per-lane (cylinder edge).
					chunk_interior := true
					for k in 0..<W8 {
						if interior[i + k] == 0 {
							chunk_interior = false
							break
						}
					}
					if chunk_interior {
						c  := intrinsics.unaligned_load((^Vec8)(&src[i]))
						xl := intrinsics.unaligned_load((^Vec8)(&src[i - 1]))
						xr := intrinsics.unaligned_load((^Vec8)(&src[i + 1]))
						yu := intrinsics.unaligned_load((^Vec8)(&src[i - n]))
						yd := intrinsics.unaligned_load((^Vec8)(&src[i + n]))
						zf := intrinsics.unaligned_load((^Vec8)(&src[i - n2]))
						zb := intrinsics.unaligned_load((^Vec8)(&src[i + n2]))
						lap := (xl - c) + (xr - c) + (yu - c) + (yd - c) + (zf - c) + (zb - c)

						// Per-lane production: gather alpha from occupant.
						alpha_arr: [W8]f32
						dr := intrinsics.unaligned_load((^Vec8)(&drive[i]))
						for k in 0..<W8 {
							sig := lat[i + k]
							a := f32(0)
							if sig > 0 && cell_alive(s, int(sig)) {
								sp := cell_species(s, int(sig))
								if is_melanin_producer(sp) {
									a = ALPHA_M[sp]
								}
							}
							alpha_arr[k] = a
						}
						alphas := simd.from_array(alpha_arr)
						v := c + dt * (dm * lap + alphas * dr)
						// Clamp negatives lane-wise.
						v_arr := simd.to_array(v)
						for k in 0..<W8 {
							if v_arr[k] < 0 { v_arr[k] = 0 }
						}
						intrinsics.unaligned_store((^Vec8)(&dst[i]), simd.from_array(v_arr))
						x += W8
						continue
					}
				}
				// Scalar lane (edges, cylinder wall neighbourhood).
				i := lidx(n, x, y, z)
				if interior[i] == 0 {
					dst[i] = src[i]
				} else {
					diff := s.params.d_m * laplacian_6(src, n, x, y, z)
					a := f32(0)
					sig := lat[i]
					if sig > 0 && cell_alive(s, int(sig)) {
						sp := cell_species(s, int(sig))
						if is_melanin_producer(sp) { a = ALPHA_M[sp] }
					}
					v := src[i] + dt * (diff + a * drive[i])
					dst[i] = v if v > 0 else 0
				}
				x += 1
			}
		}
	}
	copy(src, dst)
}

update_nutrient_simd :: proc(s: ^Sim, wall_scale := f32(1.0)) {
	n := s.params.n
	dt := s.params.dt_field
	dc := s.params.d_c
	src := s.arena.nutrient
	dst := s.arena.scratch
	lat := s.arena.lattice
	interior := s.arena.interior
	wall := s.params.c_wall * wall_scale
	n2 := n * n

	for z in 0..<n {
		for y in 0..<n {
			x := 0
			for x < n {
				if x >= 1 && x + W8 - 1 < n - 1 && y >= 1 && y < n - 1 && z >= 1 && z < n - 1 {
					i := lidx(n, x, y, z)
					chunk_interior := true
					for k in 0..<W8 {
						if interior[i + k] == 0 {
							chunk_interior = false
							break
						}
					}
					if chunk_interior {
						c  := intrinsics.unaligned_load((^Vec8)(&src[i]))
						xl := intrinsics.unaligned_load((^Vec8)(&src[i - 1]))
						xr := intrinsics.unaligned_load((^Vec8)(&src[i + 1]))
						yu := intrinsics.unaligned_load((^Vec8)(&src[i - n]))
						yd := intrinsics.unaligned_load((^Vec8)(&src[i + n]))
						zf := intrinsics.unaligned_load((^Vec8)(&src[i - n2]))
						zb := intrinsics.unaligned_load((^Vec8)(&src[i + n2]))
						lap := (xl - c) + (xr - c) + (yu - c) + (yd - c) + (zf - c) + (zb - c)
						up_arr: [W8]f32
						for k in 0..<W8 {
							sig := lat[i + k]
							u := f32(0)
							if sig > 0 && cell_alive(s, int(sig)) {
								u = UPTAKE[cell_species(s, int(sig))]
							}
							up_arr[k] = u
						}
						ups := simd.from_array(up_arr)
						v := c + dt * (dc * lap - ups)
						v_arr := simd.to_array(v)
						for k in 0..<W8 {
							if v_arr[k] < 0 { v_arr[k] = 0 }
						}
						intrinsics.unaligned_store((^Vec8)(&dst[i]), simd.from_array(v_arr))
						x += W8
						continue
					}
				}
				i := lidx(n, x, y, z)
				if interior[i] == 0 {
					dst[i] = wall
				} else {
					diff := s.params.d_c * laplacian_6(src, n, x, y, z)
					u := f32(0)
					sig := lat[i]
					if sig > 0 && cell_alive(s, int(sig)) {
						u = UPTAKE[cell_species(s, int(sig))]
					}
					v := src[i] + dt * (diff - u)
					dst[i] = v if v > 0 else 0
				}
				x += 1
			}
		}
	}
	copy(src, dst)
}

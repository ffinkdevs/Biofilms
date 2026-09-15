package viewer_main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import rl "vendor:raylib"

import cpm "../cpm"
import render "../render"

// Interactive voxel viewer: renders the CPM lattice as shaded cubes with
// outlines (like the screenshot), orbit camera, MCS stepping, species
// legend, and axes gizmo.
//
// Controls:
//   SPACE  advance 10 MCS     R  reset to seed
//   1/2/3  layout aos/soa/aosoa (rebuilds, same seed+MCS)
//   +/-    cube scale          Q/Esc quit
//
// Usage:
//   biofilm-viewer [--n 30] [--seed 42] [--layout aos] [--mcs 4]

rl_color :: proc(c: render.Species_RGB, a: u8 = 255) -> rl.Color {
	return rl.Color{c.r, c.g, c.b, a}
}

main :: proc() {
	n := 30
	seed := 42
	layout := cpm.Layout.AoS
	mcs_target := 4
	cells := 6
	for i := 1; i < len(os.args); i += 1 {
		a := os.args[i]
		val :: proc(i: int) -> (int, bool) {
			if i + 1 >= len(os.args) { return 0, false }
			v, ok := strconv.parse_int(os.args[i + 1])
			return v, ok
		}
		if a == "--n" {
			if v, ok := val(i); ok { n = v; i += 1 }
		} else if a == "--seed" {
			if v, ok := val(i); ok { seed = v; i += 1 }
		} else if a == "--mcs" {
			if v, ok := val(i); ok { mcs_target = v; i += 1 }
		} else if a == "--layout" {
			if i + 1 < len(os.args) {
				v := strings.to_lower(os.args[i + 1])
				if v == "soa" { layout = .SoA }
				else if v == "aosoa" { layout = .AoSoA }
				i += 1
			}
		}
	}

	params := cpm.default_params()
	params.n = n
	params.n_cells_per_species = cells
	sim := cpm.sim_init(params, layout, u64(seed))
	defer cpm.sim_destroy(&sim)
	run_until(&sim, mcs_target)

	rl.InitWindow(1280, 800, "Biofilm CPM — voxel viewer (Odin/raylib)")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	cam := rl.Camera3D{
		position   = {f32(n) * 1.6, f32(n) * 1.35, f32(n) * 1.6},
		target     = {f32(n) * 0.5, f32(n) * 0.45, f32(n) * 0.5},
		up         = {0, 1, 0},
		fovy       = 40,
		projection = .PERSPECTIVE,
	}
	cube_scale: f32 = 0.92

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(.SPACE) {
			run_until(&sim, sim.current_mcs + 10)
		}
		if rl.IsKeyPressed(.R) {
			cpm.sim_destroy(&sim)
			sim = cpm.sim_init(params, layout, u64(seed))
			run_until(&sim, mcs_target)
		}
		if rl.IsKeyPressed(.ONE)   { switch_layout(&sim, params, u64(seed), .AoS) }
		if rl.IsKeyPressed(.TWO)   { switch_layout(&sim, params, u64(seed), .SoA) }
		if rl.IsKeyPressed(.THREE) { switch_layout(&sim, params, u64(seed), .AoSoA) }
		if rl.IsKeyPressed(.KP_ADD) || rl.IsKeyPressed(.EQUAL) { cube_scale += 0.04 }
		if rl.IsKeyPressed(.KP_SUBTRACT) || rl.IsKeyPressed(.MINUS) { cube_scale -= 0.04 }
		if rl.IsKeyPressed(.Q) { break }
		rl.UpdateCamera(&cam, .ORBITAL)

		rl.BeginDrawing()
		rl.ClearBackground(rl.RAYWHITE)
		rl.BeginMode3D(cam)
		draw_voxels(&sim, cube_scale)
		draw_axes_gizmo(n)
		rl.EndMode3D()
		draw_hud(&sim, layout)
		rl.EndDrawing()
	}
}

run_until :: proc(sim: ^cpm.Sim, target: int) {
	for sim.current_mcs < target {
		cpm.mcs_step(sim)
		cpm.update_fields(sim)
		cpm.update_centers_of_mass(sim)
	}
}

switch_layout :: proc(sim: ^cpm.Sim, p: cpm.CPM_Params, seed: u64, l: cpm.Layout) {
	mcs := sim.current_mcs
	cpm.sim_destroy(sim)
	sim^ = cpm.sim_init(p, l, seed)
	run_until(sim, mcs)
	fmt.printf("layout -> %v (rebuilt to MCS %d)\n", l, mcs)
}

draw_voxels :: proc(sim: ^cpm.Sim, s: f32) {
	n := sim.params.n
	// N=30 -> ~4k occupied sites; per-cube DrawCube+Wires is fine.
	// N=60 (~30k sites) wants instancing: see shaders/design.md
	// (single SSBO -> IndirectDraw). This loop is the correctness
	// reference that path must match.
	c := f32(n) * 0.5
	for z in 0..<n {
		for y in 0..<n {
			for x in 0..<n {
				i := cpm.lidx(n, x, y, z)
				sig := sim.arena.lattice[i]
				if sig <= 0 || !cpm.cell_alive(sim, int(sig)) { continue }
				sp := cpm.cell_species(sim, int(sig))
				// World mapping matches render/ppm.odin: sim Z (axial)
				// points up.
				pos := rl.Vector3{f32(x) - c, f32(z) - c, f32(y) - c}
				rl.DrawCube(pos, s, s, s, rl_color(render.SPECIES_COLORS[sp]))
				rl.DrawCubeWires(pos, s, s, s, rl.Color{70, 70, 70, 255})
			}
		}
	}
}

draw_axes_gizmo :: proc(n: int) {
	c := f32(n) * 0.5
	o := rl.Vector3{-c - 4, -c - 2, -c - 4}
	rl.DrawLine3D(o, o + rl.Vector3{3, 0, 0}, rl.RED)
	rl.DrawLine3D(o, o + rl.Vector3{0, 3, 0}, rl.YELLOW)
	rl.DrawLine3D(o, o + rl.Vector3{0, 0, 3}, rl.GREEN)
}

draw_hud :: proc(sim: ^cpm.Sim, layout: cpm.Layout) {
	rl.DrawText(rl.TextFormat("MCS %d", sim.current_mcs), 16, 14, 22, rl.DARKGRAY)
	// Legend (right side, matches screenshot order + colours).
	lx := rl.GetScreenWidth() - 220
	ly := rl.GetScreenHeight() - 7*30 - 24
	for sp in 0..<cpm.N_SPECIES {
		y := ly + i32(sp*30)
		rl.DrawRectangle(i32(lx), y, 16, 20, rl_color(render.SPECIES_COLORS[sp]))
		rl.DrawRectangleLines(i32(lx), y, 16, 20, rl.DARKGRAY)
		rl.DrawText(fmt.ctprintf("%s", cpm.SPECIES_NAMES[sp]), i32(lx) + 24, y + 2, 16, rl.BLACK)
	}
	rl.DrawText(fmt.ctprintf("layout %v  alive %d  [SPACE]+10 [R]eset [1/2/3]layout [Q]uit",
		layout, cpm.count_alive(sim)), 16, rl.GetScreenHeight() - 28, 16, rl.DARKGRAY)
}

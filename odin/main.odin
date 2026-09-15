package main

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import "core:crypto/sha2"

import cpm "cpm"
import render "render"

// Headless CPM driver: runs the biofilm Potts model with the selected
// cell layout, prints Julia-compatible CSV lines (see validate_serial.jl),
// and optionally writes an isometric voxel PPM that looks like the
// screenshot (white background, shaded cubes, legend bars, MCS label).
//
// Coupled mode (default, mirrors main_coupled()) additionally steps the
// 1D radiodialysis PDE per MCS, projects contaminant to the 3D lattice,
// scales the nutrient wall by membrane integrity, and reports the pairwise
// energy diagnostic. The PDE never feeds back into Metropolis acceptance,
// so lattice trajectories are identical with coupling on or off.
//
// Usage:
//   biofilm [--n 40] [--mcs 100] [--seed 42] [--layout aos|soa|aosoa]
//           [--ppm out.ppm] [--width 1280] [--height 800] [--no-coupled]
//
// Defaults mirror main_coupled(): N=40, 6 cells/species, 100 MCS, seed 42.

Config :: struct {
	n:       int,
	mcs:     int,
	seed:    u64,
	layout:  cpm.Layout,
	ppm:     string,
	frames:  string, // prefix for per-frame PPMs (animation); "" = off
	every:   int,    // dump a frame every K MCS
	width:   int,
	height:  int,
	stats:   bool,
	cells:   int,
	coupled: bool,
	rng:     cpm.Rng_Kind,
	trace:   bool, // machine-checkable per-MCS trace (Julia cross-check)
	ensemble: bool, // machine-readable per-snapshot rows for ensembles
}

parse_args :: proc() -> Config {
	c := Config{n = 40, mcs = 100, seed = 42, layout = .AoS,
		width = 1280, height = 800, stats = true, cells = 6, coupled = true,
		every = 4}
	args := os.args[1:]
	i := 0
	for i < len(args) {
		a := args[i]
		take_int :: proc(args: []string, i: ^int, def: int) -> int {
			if i^ + 1 >= len(args) { return def }
			v, ok := strconv.parse_int(args[i^ + 1])
			i^ += 2
			return v if ok else def
		}
		if a == "--n" {
			c.n = take_int(args, &i, c.n)
		} else if a == "--mcs" {
			c.mcs = take_int(args, &i, c.mcs)
		} else if a == "--seed" {
			v := take_int(args, &i, int(c.seed))
			c.seed = u64(v)
		} else if a == "--cells" {
			c.cells = take_int(args, &i, c.cells)
		} else if a == "--width" {
			c.width = take_int(args, &i, c.width)
		} else if a == "--height" {
			c.height = take_int(args, &i, c.height)
		} else if a == "--layout" {
			if i + 1 < len(args) {
				v := strings.to_lower(args[i + 1])
				if v == "soa" { c.layout = .SoA }
				else if v == "aosoa" { c.layout = .AoSoA }
				else { c.layout = .AoS }
				i += 2
			} else { i += 1 }
		} else if a == "--ppm" {
			if i + 1 < len(args) { c.ppm = args[i + 1]; i += 2 } else { i += 1 }
		} else if a == "--frames" {
			if i + 1 < len(args) { c.frames = args[i + 1]; i += 2 } else { i += 1 }
		} else if a == "--every" {
			c.every = take_int(args, &i, c.every)
			if c.every < 1 { c.every = 1 }
		} else if a == "--no-stats" {
			c.stats = false
			i += 1
		} else if a == "--no-coupled" || a == "--no-radiolysis" {
			c.coupled = false
			i += 1
		} else if a == "--coupled" {
			c.coupled = true
			i += 1
		} else if a == "--rng" {
			if i + 1 < len(args) {
				v := strings.to_lower(args[i + 1])
				c.rng = .Julia if v == "julia" else .Splitmix
				i += 2
			} else { i += 1 }
		} else if a == "--trace" {
			c.trace = true
			i += 1
		} else if a == "--ensemble" {
			c.ensemble = true
			c.stats = false
			i += 1
		} else if a == "--help" || a == "-h" {
			fmt.println(usage())
			os.exit(0)
		} else {
			fmt.printf("unknown flag %q\n%s", a, usage())
			os.exit(2)
		}
	}
	return c
}

usage :: proc() -> string {
	return `biofilm — Cellular Potts biofilm (Odin port of biofilms_potts.jl)

  biofilm [--n 40] [--mcs 100] [--seed 42] [--layout aos|soa|aosoa]
          [--ppm out.ppm] [--width 1280] [--height 800] [--cells 6]
          [--no-coupled | --no-radiolysis]
          [--frames anim/frame --every 4]   # PPM per K MCS for video
          [--rng splitmix|julia]            # julia = Julia 1.12 MT stream
          [--ensemble]                      # ENSEMBLE rows per snapshot
          [--trace] [--no-stats]            # byte-identical trace / quiet

  CSV lines match validate_serial.jl: CSV,seed,species,vol,ncells,mel,survived
`
}

// Unweighted node means over the 1D contaminant/sorbate profiles
// (Julia parity: mean(rd.c), mean(rd.s) — not volume-weighted).
rd_mean :: proc(rd: cpm.Radiolysis_State) -> (c_mean, s_mean: f64) {
	for i in 0..<rd.params.nr {
		c_mean += rd.c[i]
		s_mean += rd.s[i]
	}
	return c_mean / f64(rd.params.nr), s_mean / f64(rd.params.nr)
}

// SHA-256 hex of a byte slice (trace lattice/field identity).
sha_hex :: proc(data: []u8) -> string {
	ctx: sha2.Context_256
	sha2.init_256(&ctx)
	sha2.update(&ctx, data)
	d: [32]u8
	sha2.final(&ctx, d[:])
	hex := "0123456789abcdef"
	out := make([]u8, 64, context.temp_allocator)
	for i in 0..<32 {
		out[2*i + 0] = hex[d[i] >> 4]
		out[2*i + 1] = hex[d[i] & 0xf]
	}
	return string(out)
}

bytes_of :: proc(s: $T/[]$E) -> []u8 {
	return mem.slice_ptr((^u8)(raw_data(s)), len(s) * size_of(E))
}

// Machine-checkable trace: lattice/field hashes, per-cell volumes/COMs
// (as u64 bit patterns), per-species snapshot stats. Byte-identical in
// format to jl_trace.jl output for diffing.
dump_trace :: proc(sim: ^cpm.Sim, rd: ^cpm.Radiolysis_State, seed: u64, m: int, coupled: bool) {
	fmt.printf("TRACE %d %d LAT %s MEL %s NUT %s RAD %s ALIVE %d\n",
		seed, m,
		sha_hex(bytes_of(sim.arena.lattice)),
		sha_hex(bytes_of(sim.arena.melanin)),
		sha_hex(bytes_of(sim.arena.nutrient)),
		sha_hex(bytes_of(sim.arena.radiation)),
		cpm.count_alive(sim))
	for id in 1..<sim.next_id {
		if !cpm.cell_alive(sim, id) { continue }
		cx, cy, cz := cpm.cell_com(sim, id)
		fmt.printf("TRACECELL %d %d %d %d %d %d %d %d\n",
			seed, m, id, cpm.cell_species(sim, id) + 1,
			cpm.cell_volume(sim, id),
			transmute(u64)cx, transmute(u64)cy, transmute(u64)cz)
	}
	snap := cpm.take_snapshot(sim)
	for sp in 0..<cpm.N_SPECIES {
		st := snap.species[sp]
		fmt.printf("TRACESPEC %d %d %d %d %d %d %d\n",
			seed, m, sp + 1, st.volume, st.n_cells,
			transmute(u64)st.mean_r, transmute(u64)st.mean_mel)
	}
	if coupled && m > 0 {
		fmt.printf("RDC %d %d %d %d %s %s %s\n",
			seed, m, transmute(u64)rd.t, transmute(u64)rd.m,
			sha_hex(bytes_of(rd.c)), sha_hex(bytes_of(rd.s)),
			sha_hex(bytes_of(sim.arena.contaminant)))
	}
	free_all(context.temp_allocator)
}

main :: proc() {
	cfg := parse_args()
	layout_name := "aos" if cfg.layout == .AoS else ("soa" if cfg.layout == .SoA else "aosoa")

	params := cpm.default_params()
	params.n = cfg.n
	params.n_cells_per_species = cfg.cells

	t0 := time.tick_now()
	sim := cpm.sim_init(params, cfg.layout, cfg.seed, cfg.rng)
	defer cpm.sim_destroy(&sim)

	rd := cpm.radiolysis_init(cpm.default_radiolysis_params(), f64(cfg.n) * 0.5)
	defer cpm.radiolysis_destroy(&rd)

	mode := "coupled" if cfg.coupled else "CPM-only"
	rng_name := "julia" if cfg.rng == .Julia else "splitmix"
	fmt.printf("CPM biofilm (%s, %s) — N=%d cells/species=%d layout=%s seed=%d MCS=%d\n",
		mode, rng_name, cfg.n, cfg.cells, layout_name, cfg.seed, cfg.mcs)


	if cfg.trace {
		dump_trace(&sim, &rd, cfg.seed, 0, cfg.coupled)
	}

	frame_idx := 0
	dump_frame :: proc(sim: ^cpm.Sim, cfg: ^Config, mcs, idx: int) {
		vl := render.collect_voxels(sim)
		defer render.free_voxel_list(&vl)
		img := render.image_make(cfg.width, cfg.height)
		defer render.image_free(&img)
		render.render_voxels(&img, sim, &vl, mcs)
		path := fmt.tprintf("%s_%d.ppm", cfg.frames, idx)
		if !render.write_ppm(&img, path) {
			fmt.printf("FAILED to write %s\n", path)
			os.exit(1)
		}
	}

	for m in 1..=cfg.mcs {
		cpm.mcs_step(&sim)
		if cfg.coupled {
			// Biomass -> PDE sink, refreshed every 10 MCS (Julia parity).
			if m % 10 == 1 {
				xt, xr := cpm.radial_biomass_means(&sim, rd.params.nr)
				rd.params.x_total = xt
				rd.params.x_red = xr
			}
			cpm.radiolysis_step(&rd, rd.params.dt_rd)
			cpm.radial_to_3d(&sim, &rd)
		}
		cpm.update_fields(&sim, rd.m if cfg.coupled else 1.0)
		cpm.update_centers_of_mass(&sim)
		if cfg.trace {
			dump_trace(&sim, &rd, cfg.seed, m, cfg.coupled)
		}
		if len(cfg.frames) > 0 && (m % cfg.every == 0 || m == cfg.mcs) {
			dump_frame(&sim, &cfg, m, frame_idx)
			frame_idx += 1
		}
		if m % params.snapshot_interval == 0 || m == cfg.mcs {
			snap := cpm.take_snapshot(&sim)
			if cfg.ensemble {
				for sp in 0..<cpm.N_SPECIES {
					st := snap.species[sp]
					fmt.printf("ENSEMBLE,%d,%d,%d,%d,%d,%.6f,%.6f\n",
						cfg.seed, m, sp + 1, st.volume, st.n_cells,
						st.mean_mel, st.mean_mel_parcel)
				}
			} else {
				fmt.printf("MCS %d alive=%d pair_e=%.2f\n", m, cpm.count_alive(&sim), snap.pair_e)
			}
			if cfg.stats {
				for sp in 0..<cpm.N_SPECIES {
					st := snap.species[sp]
					fmt.printf("  %-18s vol=%d cells=%d mean_r=%.2f mel=%.4f\n",
						cpm.SPECIES_NAMES[sp], st.volume, st.n_cells, st.mean_r, st.mean_mel)
				}
			}
			if cfg.coupled {
				c_mean, s_mean := rd_mean(rd)
				fmt.printf("  [RD] t=%.1f m=%.4f P_eff=%.5f c_wall=%.4f c_mean=%.4f s_mean=%.4f\n",
					rd.t, rd.m, cpm.membrane_peff(&rd), rd.c[rd.params.nr-1],
					c_mean, s_mean)
			}
		}
	}

	// Final CSV (Julia-compatible columns).
	snap := cpm.take_snapshot(&sim)
	for sp in 0..<cpm.N_SPECIES {
		st := snap.species[sp]
		fmt.printf("CSV,%d,%d,%d,%d,%.5f,%d\n",
			cfg.seed, sp + 1, st.volume, st.n_cells, st.mean_mel,
			1 if st.n_cells > 0 else 0)
	}
	fmt.printf("CSVTOT,%d,%d\n", cfg.seed, cpm.count_alive(&sim))
	if cfg.coupled {
		c_mean, s_mean := rd_mean(rd)
		fmt.printf("membrane m=%.4f P_eff=%.5f (x%.2f baseline) dose=%.1f c(R)=%.4f c_mean=%.4f\n",
			rd.m, cpm.membrane_peff(&rd),
			cpm.membrane_peff(&rd) / rd.params.p0,
			rd.params.ddot_r * rd.t, rd.c[rd.params.nr-1], c_mean)
	}
	fmt.printf("done in %.1fs layout=%s\n", time.duration_seconds(time.tick_since(t0)), layout_name)

	if len(cfg.ppm) > 0 {
		vl := render.collect_voxels(&sim)
		defer render.free_voxel_list(&vl)
		img := render.image_make(cfg.width, cfg.height)
		defer render.image_free(&img)
		render.render_voxels(&img, &sim, &vl, cfg.mcs)
		if render.write_ppm(&img, cfg.ppm) {
			fmt.printf("wrote %s (%dx%d, %d voxels)\n", cfg.ppm, cfg.width, cfg.height, vl.count)
		} else {
			fmt.printf("FAILED to write %s\n", cfg.ppm)
			os.exit(1)
		}
	}
}

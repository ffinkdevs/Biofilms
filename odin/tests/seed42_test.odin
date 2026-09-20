package tests

import "core:testing"
import "core:os"
import "core:fmt"
import "core:strings"
import "core:strconv"
import cpm "../cpm"

// End-to-end golden contract: N=40, 6 cells/species, 100 MCS, seed 42,
// coupled, `--rng julia` must reproduce tests/fixtures/serial_seed42.csv
// (the validate_serial.jl seed-42 output) 7/7 plus the membrane line.
// This used to be a manual run the suite never read; now the suite reads
// the file, so a regressed stream or model fails here loudly.

@(private)
SEED42_PATHS := []string{
	"../tests/fixtures/serial_seed42.csv", // `odin test tests/` from odin/
	"tests/fixtures/serial_seed42.csv",    // built binary from repo root
}

@(test)
test_seed42_golden_csv :: proc(t: ^testing.T) {
	raw: []u8
	ok := false
	for path in SEED42_PATHS {
		data, err := os.read_entire_file_from_path(path, context.allocator)
		if err == os.ERROR_NONE {
			raw = data
			ok = true
			break
		}
	}
	testing.expect(t, ok)
	if !ok { return }
	defer delete(raw)
	text := string(raw)

	// Parse the fixture first, into stack arrays (temp allocator is
	// owned by take_snapshot below, so keep no heap here).
	want_vol:   [7]int
	want_ncells: [7]int
	want_mel:   [7]string
	want_surv:  [7]int
	want_alive := -1
	want_mem := ""
	n_csv := 0
	{
		defer free_all(context.temp_allocator)
		for line in strings.split_lines(text, context.temp_allocator) {
			line := strings.trim_space(line)
			if len(line) == 0 || line[0] == '#' { continue }
			f := strings.split(line, ",", context.temp_allocator)
			if f[0] == "CSV" {
				testing.expectf(t, len(f) == 7, "bad CSV line %q", line)
				sp, _ := strconv.parse_int(f[2])
				vol, _ := strconv.parse_int(f[3])
				ncells, _ := strconv.parse_int(f[4])
				surv, _ := strconv.parse_int(f[6])
				want_vol[sp - 1] = vol
				want_ncells[sp - 1] = ncells
				want_mel[sp - 1] = strings.clone(f[5])
				want_surv[sp - 1] = surv
				n_csv += 1
			} else if f[0] == "CSVTOT" {
				alive, _ := strconv.parse_int(f[2])
				want_alive = alive
				want_mem = strings.clone(f[3])
				n_csv += 1
			}
		}
	}
	defer {
		for m in want_mel { delete(m) }
		delete(want_mem)
	}
	testing.expectf(t, n_csv == 8, "fixture has %d/8 data lines", n_csv)

	// Run the exact main.odin coupled loop.
	p := cpm.default_params()
	p.n = 40
	p.n_cells_per_species = 6
	s := cpm.sim_init(p, .AoS, 42, .Julia)
	defer cpm.sim_destroy(&s)
	rd := cpm.radiolysis_init(cpm.default_radiolysis_params(), f64(p.n) * 0.5)
	defer cpm.radiolysis_destroy(&rd)
	for m in 1..=100 {
		cpm.mcs_step(&s)
		if m % 10 == 1 {
			xt, xr := cpm.radial_biomass_means(&s, rd.params.nr)
			rd.params.x_total = xt
			rd.params.x_red = xr
		}
		cpm.radiolysis_step(&rd, rd.params.dt_rd)
		cpm.radial_to_3d(&s, &rd)
		cpm.update_fields(&s, rd.m)
		cpm.update_centers_of_mass(&s)
	}
	snap := cpm.take_snapshot(&s)

	for sp in 1..=7 {
		st := snap.species[sp - 1]
		testing.expectf(t, st.volume == want_vol[sp - 1],
			"sp %d volume %d, want %d", sp, st.volume, want_vol[sp - 1])
		testing.expectf(t, st.n_cells == want_ncells[sp - 1],
			"sp %d ncells %d, want %d", sp, st.n_cells, want_ncells[sp - 1])
		testing.expectf(t, fmt.tprintf("%.5f", st.mean_mel) == want_mel[sp - 1],
			"sp %d mel %.5f, want %s", sp, st.mean_mel, want_mel[sp - 1])
		testing.expectf(t, (1 if st.n_cells > 0 else 0) == want_surv[sp - 1],
			"sp %d survived flag mismatch", sp)
	}
	testing.expectf(t, cpm.count_alive(&s) == want_alive,
		"alive %d, want %d", cpm.count_alive(&s), want_alive)
	testing.expectf(t, fmt.tprintf("%.5f", rd.m) == want_mem,
		"membrane %.5f, want %s", rd.m, want_mem)
}

package cpm

import "core:math"

// Canonical species registry. Authoritative order matches biofilms_potts.jl
// SPECIES_NAMES[1..7] and FIG_COLORS. Do not reorder: HDF5 lineage labels,
// J-matrix indices, and CSV columns all assume this order.
//
// Precision: all compute floats are f64 (Julia parity — Julia's CPMParams,
// fields, and Hamiltonian are Float64 throughout). Rendering keeps its own
// f32 palette; it never feeds back into dynamics.
N_SPECIES :: 7

CN :: 0 // C. neoformans      (radiotropic, melanin producer)
DR :: 1 // D. radiodurans
CS :: 2 // C. sphaerospermum  (radiotropic, melanin producer)
BS :: 3 // B. subtilis
AN :: 4 // A. niger           (melanin producer)
SO :: 5 // S. oneidensis
OI :: 6 // O. intermedium AM7

SPECIES_NAMES := [N_SPECIES]string{
	"C. neoformans",
	"D. radiodurans",
	"C. sphaerospermum",
	"B. subtilis",
	"A. niger",
	"S. oneidensis",
	"O. intermedium",
}

// Short labels used by the voxel legend (matches screenshot, right side).
SPECIES_SHORT := [N_SPECIES]string{
	"C. neoformans",
	"D. radiodurans",
	"C. sphaerospermum",
	"B. subtilis",
	"A. niger",
	"S. oneidensis",
	"O. intermedium",
}

// Per-species tropism coefficient beta_ion (Table 2 midpoints).
// Negative = drifts UP the dose gradient (toward axis). Dimensionless
// in practice; see biofilms_potts.jl §1. Shipped values are inert at
// T_cpm=5 (bias 1.00001 for CN/CS); the melanin term dominates.
BETA_ION := [N_SPECIES]f64{-5e-5, 2.5e-5, -5e-5, 3e-3, 2.5e-4, 7.5e-2, 1e-2}

// Melanin production rate alpha_M per species (Table 2 midpoints).
ALPHA_M := [N_SPECIES]f64{0.10, 0.0, 0.14, 0.0, 0.065, 0.0, 0.0}

// Melanin coupling in dH: 0.5 for radiotropic species, else 0.
// Ledgered as cpm.melanin_coupling; the term that actually moves the model.
MEL_COEF := [N_SPECIES]f64{0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0}

// Nutrient uptake per species.
UPTAKE := [N_SPECIES]f64{0.01, 0.02, 0.01, 0.03, 0.01, 0.03, 0.02}

// Species diffusion Ds (Table 2 midpoints). Currently informational:
// the CPM has no per-species motility temperature; kept for parity.
DIFF_S := [N_SPECIES]f64{0.05, 0.25, 0.025, 0.50, 0.025, 0.40, 0.20}

is_radiotropic :: #force_inline proc(s: int) -> bool {
	return s == CN || s == CS
}

is_melanin_producer :: #force_inline proc(s: int) -> bool {
	return s == CN || s == CS || s == AN
}

// Simulation parameters. Mirrors CPMParams in biofilms_potts.jl.
CPM_Params :: struct {
	n:                   int,     // lattice edge (N x N x N, cylindrical mask)
	t_cpm:               f64,     // Metropolis temperature
	lambda_v:            f64,     // volume constraint strength
	v_target:            i32,     // target volume per cell (sites)
	i0:                  f64,     // source intensity at axis
	kappa:               f64,     // radial attenuation
	d_m:                 f64,     // melanin diffusion coefficient
	d_c:                 f64,     // nutrient diffusion coefficient
	dt_field:            f64,     // field update time step
	c_wall:              f64,     // nutrient concentration at wall
	gamma_mutual:        f64,     // pairwise attraction strength (diagnostic only)
	sigma_mutual:        f64,     // pairwise range (diagnostic only)
	n_cells_per_species: int,
	snapshot_interval:   int,
}

default_params :: proc() -> CPM_Params {
	return CPM_Params{
		n = 40,
		t_cpm = 5.0,
		lambda_v = 10.0,
		v_target = 120,
		i0 = 1.0,
		kappa = 2.0,
		d_m = 0.1,
		d_c = 0.2,
		dt_field = 0.5,
		c_wall = 1.0,
		gamma_mutual = 2.0,
		sigma_mutual = 10.0,
		n_cells_per_species = 6,
		snapshot_interval = 20,
	}
}

// Cell lifecycle states for the growth/survival response module
// (spec/response_growth_survival.md §1). Numeric values are pinned:
// zero-valued registry slots must read as viable so pre-module code
// paths (which never set the field) keep working unchanged.
CELL_VIABLE :: u8(1)
CELL_MITOTIC :: u8(2) // transient within the division step
CELL_DYING  :: u8(3) // mitotic catastrophe; resorbing toward cull

// Growth/survival response parameters. Values are DECLARED, not tuned
// (spec §2, Tamborino et al. 2025 EBRT fit on NCI-H69; reference vectors
// in PRRT-spatial-CPM spec/reference_vectors.json):
//   alpha = 0.24 Gy^-1, beta = 0.06 Gy^-2 (NOT the PRRT-fitted pair —
//     those already embed protraction and G would double-count it)
//   T_rep = 1.5 h (declared choice; Tamborino publishes no explicit value)
//   T_active = 96 h exposure window for the per-cycle Lea-Catcheside G
//   sigma_div = 0.10 heritable lognormal drift (study3_run.py:47)
//   T_doubling = 58.4 h reference doubling (report band, not dynamics)
//   cycle_hours = 1344 h (56-day study3 window) for the doubling report
Response_Params :: struct {
	alpha:        f64,
	beta:         f64,
	t_rep_h:      f64,
	t_active_h:   f64,
	sigma_div:    f64,
	t_doubling_h: f64,
	cycle_hours:  f64,
}

default_response_params :: proc() -> Response_Params {
	return Response_Params{
		alpha        = 0.24,
		beta         = 0.06,
		t_rep_h      = 1.5,
		t_active_h   = 96.0,
		sigma_div    = 0.10,
		t_doubling_h = 58.4,
		cycle_hours  = 1344.0,
	}
}

// Adhesion matrix J, (N_SPECIES+1) x (N_SPECIES+1), row/col 0 = medium.
// Lower J = more adhesive. Matches build_J_matrix() in both Julia ports.
build_J :: proc() -> [8][8]f64 {
	J: [8][8]f64
	for i in 0..<8 {
		for j in 0..<8 {
			J[i][j] = 12.0
		}
		J[i][i] = 0.0
	}
	for s in 1..<8 {
		J[0][s] = 16.0
		J[s][0] = 16.0
	}
	// mutualistic pairs (attractive)
	mut := [5][2]int{{CN, DR}, {CN, CS}, {CS, AN}, {SO, OI}, {DR, BS}}
	for e in mut {
		a := e[0] + 1
		b := e[1] + 1
		J[a][b] = 4.0
		J[b][a] = 4.0
	}
	// antagonistic pairs
	ant := [3][2]int{{CN, SO}, {CS, SO}, {BS, AN}}
	for e in ant {
		a := e[0] + 1
		b := e[1] + 1
		J[a][b] = 20.0
		J[b][a] = 20.0
	}
	return J
}

// 26-connected Moore neighbourhood offsets, dz fastest.
// Order matches Julia's `[(dx,dy,dz) for dx in -1:1 for dy in -1:1
// for dz in -1:1 ...]` (last generator fastest) — the adhesion sum order
// matters for bitwise parity, as does the rand(1:26) index mapping.
NEIGHBOURS_26 := [26][3]i8{
	{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
	{-1,  0, -1}, {-1,  0, 0}, {-1,  0, 1},
	{-1,  1, -1}, {-1,  1, 0}, {-1,  1, 1},
	{ 0, -1, -1}, { 0, -1, 0}, { 0, -1, 1},
	{ 0,  0, -1},             { 0,  0, 1},
	{ 0,  1, -1}, { 0,  1, 0}, { 0,  1, 1},
	{ 1, -1, -1}, { 1, -1, 0}, { 1, -1, 1},
	{ 1,  0, -1}, { 1,  0, 0}, { 1,  0, 1},
	{ 1,  1, -1}, { 1,  1, 0}, { 1,  1, 1},
}

NEIGHBOURS_6 := [6][3]i8{
	{1, 0, 0}, {-1, 0, 0},
	{0, 1, 0}, {0, -1, 0},
	{0, 0, 1}, {0, 0, -1},
}

// Site coordinate in Julia's 1-based convention as Float64: lattice index
// `c` (0-based) holds Julia coordinate `c+1`. All field values, COMs, and
// radial distances use this convention for bitwise Julia parity.
site_coord :: #force_inline proc(c: int) -> f64 {
	return f64(c) + 1.0
}

// Banker's rounding (round-half-to-even) for non-negative values,
// matching Julia's round(Int, x). Odin's math.round rounds half away
// from zero — identical except on exact .5 fractions, which do occur in
// radial binning. Only used where Julia rounds (radiolysis binning).
round_half_even :: proc(x: f64) -> i64 {
	fl := math.floor(x)
	f := x - fl
	if f < 0.5 {
		return i64(fl)
	} else if f > 0.5 {
		return i64(fl) + 1
	}
	fi := i64(fl)
	return fi if fi & 1 == 0 else fi + 1
}

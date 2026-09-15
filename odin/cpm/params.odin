package cpm

// Canonical species registry. Authoritative order matches biofilms_potts.jl
// SPECIES_NAMES[1..7] and FIG_COLORS. Do not reorder: HDF5 lineage labels,
// J-matrix indices, and CSV columns all assume this order.
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
BETA_ION := [N_SPECIES]f32{-5e-5, 2.5e-5, -5e-5, 3e-3, 2.5e-4, 7.5e-2, 1e-2}

// Melanin production rate alpha_M per species (Table 2 midpoints).
ALPHA_M := [N_SPECIES]f32{0.10, 0.0, 0.14, 0.0, 0.065, 0.0, 0.0}

// Melanin coupling in dH: 0.5 for radiotropic species, else 0.
// Ledgered as cpm.melanin_coupling; the term that actually moves the model.
MEL_COEF := [N_SPECIES]f32{0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0}

// Nutrient uptake per species.
UPTAKE := [N_SPECIES]f32{0.01, 0.02, 0.01, 0.03, 0.01, 0.03, 0.02}

// Species diffusion Ds (Table 2 midpoints). Currently informational:
// the CPM has no per-species motility temperature; kept for parity.
DIFF_S := [N_SPECIES]f32{0.05, 0.25, 0.025, 0.50, 0.025, 0.40, 0.20}

is_radiotropic :: proc(s: int) -> bool {
	return s == CN || s == CS
}

is_melanin_producer :: proc(s: int) -> bool {
	return s == CN || s == CS || s == AN
}

// Simulation parameters. Mirrors CPMParams in biofilms_potts.jl.
CPM_Params :: struct {
	n:                   int,     // lattice edge (N x N x N, cylindrical mask)
	t_cpm:               f32,     // Metropolis temperature
	lambda_v:            f32,     // volume constraint strength
	v_target:            i32,     // target volume per cell (sites)
	i0:                  f32,     // source intensity at axis
	kappa:               f32,     // radial attenuation
	d_m:                 f32,     // melanin diffusion coefficient
	d_c:                 f32,     // nutrient diffusion coefficient
	dt_field:            f32,     // field update time step
	c_wall:              f32,     // nutrient concentration at wall
	gamma_mutual:        f32,     // pairwise attraction strength (diagnostic only)
	sigma_mutual:        f32,     // pairwise range (diagnostic only)
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

// Adhesion matrix J, (N_SPECIES+1) x (N_SPECIES+1), row/col 0 = medium.
// Lower J = more adhesive. Matches build_J_matrix() in both Julia ports.
build_J :: proc() -> [8][8]f32 {
	J: [8][8]f32
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

// 26-connected Moore neighbourhood offsets, precomputed once.
NEIGHBOURS_26 := [26][3]i8{
	{-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
	{-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
	{-1,  1, -1}, {0,  1, -1}, {1,  1, -1},
	{-1, -1,  0}, {0, -1,  0}, {1, -1,  0},
	{-1,  0,  0},             {1,  0,  0},
	{-1,  1,  0}, {0,  1,  0}, {1,  1,  0},
	{-1, -1,  1}, {0, -1,  1}, {1, -1,  1},
	{-1,  0,  1}, {0,  0,  1}, {1,  0,  1},
	{-1,  1,  1}, {0,  1,  1}, {1,  1,  1},
}

NEIGHBOURS_6 := [6][3]i8{
	{1, 0, 0}, {-1, 0, 0},
	{0, 1, 0}, {0, -1, 0},
	{0, 0, 1}, {0, 0, -1},
}

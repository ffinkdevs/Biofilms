package cpm

// Deterministic RNG: splitmix64 counter stream + float helpers.
// Julia uses MersenneTwister; we do NOT claim bit-identical streams.
// What we guarantee: same seed -> same trajectory within this Odin port,
// across AoS/SoA/AoSoA layouts (tested), and across scalar/SIMD field
// paths. Cross-language validation is statistical (see validate_serial.jl
// contract), never pathwise.
Rng :: struct {
	state: u64,
}

rng_init :: proc(seed: u64) -> Rng {
	// Avoid the fixed point at 0.
	s := seed + 0x9e3779b97f4a7c15
	if s == 0 { s = 0x243f6a8885a308d3 }
	return Rng{state = s}
}

@(private)
splitmix64_next :: proc(r: ^Rng) -> u64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ~ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ~ (z >> 27)) * 0x94d049bb133111eb
	return z ~ (z >> 31)
}

// Uniform u64.
rng_u64 :: proc(r: ^Rng) -> u64 {
	return splitmix64_next(r)
}

// Uniform float in [0, 1). 24-bit mantissa, matching JACC u01().
rng_f32 :: proc(r: ^Rng) -> f32 {
	v := splitmix64_next(r) >> 40 // top 24 bits
	return f32(v) * (1.0 / 16777216.0)
}

// Uniform int in [lo, hi] inclusive. Rejection-free modulo is fine here:
// the CPM consumes this stream identically in all layouts, so any bias is
// shared and the layout-equivalence tests hold exactly.
rng_range :: proc(r: ^Rng, lo, hi: int) -> int {
	span := u64(hi - lo + 1)
	return lo + int(splitmix64_next(r) % span)
}

// Uniform int in [0, n).
rng_below :: proc(r: ^Rng, n: int) -> int {
	return int(splitmix64_next(r) % u64(n))
}

// Hash two streams (seed + step + site) into an independent substream,
// used by the checkerboard/GPU-style update if enabled.
rng_hash3 :: proc(a, b, c: u64) -> u64 {
	h := a ~ (b &+ 0x9e3779b97f4a7c15) ~ (c &+ 0xbf58476d1ce4e5b9)
	h = (h ~ (h >> 30)) * 0xbf58476d1ce4e5b9
	h = (h ~ (h >> 27)) * 0x94d049bb133111eb
	return h ~ (h >> 31)
}

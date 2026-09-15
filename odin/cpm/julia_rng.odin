package cpm

import "core:crypto/sha2"
import "core:mem"

// Julia_Rng replicates Julia 1.12's MersenneTwister(seed::Integer) stream
// bit-for-bit: SHA-256 seed digest -> dSFMT19937 -> Julia's float/int
// block caches -> NDL range sampler. Verified against Julia 1.12.6 output
// (see tests/julia_rng_test.odin).
//
// What is mirrored (stdlib/Random/src, Julia 1.12):
//   hash_seed(::Integer)  — SHA-256 over LE UInt32 words (RNGs.jl)
//   initstate!            — dsfmt_init_by_array over the 8 digest words
//   float cache           — 1002 CloseOpen12 doubles, CloseOpen01 = v - 1.0
//   ints cache            — 627xUInt128 fill condensed to 501 (mt_setfull!)
//   mt_pop!(UInt64)       — byte-indexed pop from the ints cache
//   rand(1:N) / Int64 ranges — SamplerRangeNDL (default since Julia 1.5)
//
// What is NOT mirrored: bulk rand! paths, Float32/16 masking, Bool,
// randperm, BigInt, randjump. The CPM trajectory needs only Float64,
// UInt64, and Int64 ranges, all covered here.

JL_CACHE_F :: 1002 // MT_CACHE_F = 501 << 1
JL_CACHE_I :: 501  // live UInt128 words after condensation

Julia_Rng :: struct {
	ds:    Dsfmt_State,
	vals:  [JL_CACHE_F]f64,
	idx_f: int, // next float to pop; JL_CACHE_F = cache empty
	ints:  [JL_CACHE_I]u128,
	idx_i: int, // bytes remaining in ints cache; 0 = empty
}

// Seed from a non-negative integer (Julia Integer seeds hash to a 256-bit
// digest; negative seeds pre-negate and append a 0x01 marker — supported).
julia_rng_init :: proc(seed: i64) -> Julia_Rng {
	r: Julia_Rng
	data: [16]u8 // up to 64-bit magnitude + sign marker
	nbytes := 0
	mag := u64(seed)
	neg := seed < 0
	if neg {
		mag = ~mag
	}
	for {
		word := u32(mag & 0xffffffff)
		data[nbytes + 0] = u8(word)
		data[nbytes + 1] = u8(word >> 8)
		data[nbytes + 2] = u8(word >> 16)
		data[nbytes + 3] = u8(word >> 24)
		nbytes += 4
		mag >>= 32
		if mag == 0 {
			break
		}
	}
	if neg {
		data[nbytes] = 0x01
		nbytes += 1
	}
	ctx: sha2.Context_256
	sha2.init_256(&ctx)
	sha2.update(&ctx, data[:nbytes])
	digest: [32]u8
	sha2.final(&ctx, digest[:])
	key: [8]u32
	for i in 0..<8 {
		key[i] = u32(digest[4*i]) | u32(digest[4*i+1]) << 8 |
			u32(digest[4*i+2]) << 16 | u32(digest[4*i+3]) << 24
	}
	dsfmt_init_by_array(&r.ds, key[:])
	r.idx_f = JL_CACHE_F
	r.idx_i = 0
	return r
}

// rand(rng) :: Float64 in [0,1).
jrand_f64 :: #force_inline proc(r: ^Julia_Rng) -> f64 {
	if r.idx_f == JL_CACHE_F {
		dsfmt_fill_c1o2(&r.ds, r.vals[:])
		r.idx_f = 0
	}
	v := r.vals[r.idx_f]
	r.idx_f += 1
	return v - 1.0
}

// mt_setfull!: refill the ints cache from 627 fresh UInt128 (= 1254
// CloseOpen12 doubles) condensed to 501 fully-randomized words.
@(private)
jints_refill :: proc(r: ^Julia_Rng) {
	buf: [627]u128
	f := mem.slice_ptr((^f64)(&buf[0]), 627 * 2)
	dsfmt_fill_c1o2(&r.ds, f)
	k := 500
	n := 0
	for n != 500 {
		k += 1
		u := buf[k]
		buf[n] = buf[n] ~ (u << 48); n += 1
		buf[n] = buf[n] ~ (u << 36); n += 1
		buf[n] = buf[n] ~ (u << 24); n += 1
		buf[n] = buf[n] ~ (u << 12); n += 1
	}
	buf[500] = buf[500] ~ (buf[626] << 48)
	for i in 0..<JL_CACHE_I {
		r.ints[i] = buf[i]
	}
	r.idx_i = JL_CACHE_I * 16
}

// rand(rng, UInt64).
jrand_u64 :: #force_inline proc(r: ^Julia_Rng) -> u64 {
	if r.idx_i < 8 {
		jints_refill(r)
	}
	r.idx_i -= 8
	i := r.idx_i
	x := r.ints[i >> 4]
	lane := (i >> 3) & 1
	return u64(x >> u32(lane * 64))
}

// rand(rng, lo:hi) for Int64 ranges (SamplerRangeNDL / Lemire).
// The rejection threshold t is computed lazily: it needs a 64-bit
// division but is only consulted when low < s (probability ~s/2^64,
// i.e. never for our ranges) — same values, skipped dead division.
jrand_range :: #force_inline proc(r: ^Julia_Rng, lo, hi: i64) -> i64 {
	s := u64(hi - lo) + 1
	m := u128(jrand_u64(r)) * u128(s)
	low := u64(m)
	if low < s {
		t := (0 - s) % s
		for low < t {
			m = u128(jrand_u64(r)) * u128(s)
			low = u64(m)
		}
	}
	return i64(m >> 64) + lo
}

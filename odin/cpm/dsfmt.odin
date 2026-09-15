package cpm

import "core:mem"

// Pure-Odin dSFMT19937 (double-precision SIMD Mersenne Twister, MEXP=19937).
// This is the exact algorithm behind Julia's MersenneTwister, which calls
// libdSFMT's C implementation of it. Constants and structure verified
// against the dSFMT reference (BSD, Saito/Matsumoto) and Julia 1.12's
// stdlib/Random/src/{DSFMT,RNGs}.jl:
//
//   N=191, POS1=117, SL1=19, SR=12,
//   MSK1=0x000ffafffffffb3f, MSK2=0x000ffdfffc90fffd,
//   FIX1/2, PCV1/2, LOW_MASK, HIGH_CONST as below.
//
// Scalar (non-SIMD) recurrence: dSFMT guarantees every SIMD variant
// produces the identical stream, so the scalar form is bit-exact.
// State words are always valid [1,2) doubles — the state doubles as output.

DSFMT_N    :: 191
DSFMT_POS1 :: 117
DSFMT_SL1  :: 19
DSFMT_SR   :: 12
DSFMT_N64  :: DSFMT_N * 2 // 382, minimum fill size (doubles)

DSFMT_MSK1      :: u64(0x000ffafffffffb3f)
DSFMT_MSK2      :: u64(0x000ffdfffc90fffd)
DSFMT_FIX1      :: u64(0x90014964b32f4329)
DSFMT_FIX2      :: u64(0x3b8d12ac548a7c7a)
DSFMT_PCV1      :: u64(0x3d84e1ac0dc82880)
DSFMT_LOW_MASK  :: u64(0x000fffffffffffff)
DSFMT_HIGH_CONST:: u64(0x3ff0000000000000)

Dsfmt_W128 :: [2]u64

Dsfmt_State :: struct {
	status: [DSFMT_N + 1]Dsfmt_W128, // 192 words; word 191 is the lung
	idx:    i32,
}

dsfmt_rec :: #force_inline proc(r, a, b, lung: ^Dsfmt_W128) {
	t0 := a[0]
	t1 := a[1]
	l0 := lung[0]
	l1 := lung[1]
	lung[0] = (t0 << DSFMT_SL1) ~ (l1 >> 32) ~ (l1 << 32) ~ b[0]
	lung[1] = (t1 << DSFMT_SL1) ~ (l0 >> 32) ~ (l0 << 32) ~ b[1]
	r[0] = (lung[0] >> DSFMT_SR) ~ (lung[0] & DSFMT_MSK1) ~ t0
	r[1] = (lung[1] >> DSFMT_SR) ~ (lung[1] & DSFMT_MSK2) ~ t1
}

@(private)
dsfmt_ini_f1 :: proc(x: u32) -> u32 {
	return (x ~ (x >> 27)) * 1664525
}

@(private)
dsfmt_ini_f2 :: proc(x: u32) -> u32 {
	return (x ~ (x >> 27)) * 1566083941
}

// Seed from an array of UInt32 words (Julia passes the 8 words of the
// SHA-256 seed digest). Verbatim port of dsfmt_chk_init_by_array.
dsfmt_init_by_array :: proc(s: ^Dsfmt_State, key: []u32) {
	SIZE :: (DSFMT_N + 1) * 4 // 768
	LAG  :: 11               // SIZE >= 623
	MID  :: (SIZE - LAG) / 2 // 378

	st := mem.slice_ptr((^u32)(&s.status[0][0]), SIZE)
	raw := mem.slice_ptr((^u8)(&s.status[0][0]), (DSFMT_N + 1) * 16)
	for &b in raw {
		b = 0x8b
	}
	count := SIZE
	if len(key) + 1 > SIZE {
		count = len(key) + 1
	}
	r := dsfmt_ini_f1(st[0] ~ st[MID] ~ st[SIZE-1])
	st[MID] += r
	r += u32(len(key))
	st[(MID + LAG) % SIZE] += r
	st[0] = r
	count -= 1

	i := 1
	j := 0
	for j < count && j < len(key) {
		r = dsfmt_ini_f1(st[i] ~ st[(i+MID)%SIZE] ~ st[(i+SIZE-1)%SIZE])
		st[(i+MID)%SIZE] += r
		r += key[j] + u32(i)
		st[(i+MID+LAG)%SIZE] += r
		st[i] = r
		i = (i + 1) % SIZE
		j += 1
	}
	for j < count {
		r = dsfmt_ini_f1(st[i] ~ st[(i+MID)%SIZE] ~ st[(i+SIZE-1)%SIZE])
		st[(i+MID)%SIZE] += r
		r += u32(i)
		st[(i+MID+LAG)%SIZE] += r
		st[i] = r
		i = (i + 1) % SIZE
		j += 1
	}
	for _ in 0..<SIZE {
		r = dsfmt_ini_f2(st[i] + st[(i+MID)%SIZE] + st[(i+SIZE-1)%SIZE])
		st[(i+MID)%SIZE] ~= r
		r -= u32(i)
		st[(i+MID+LAG)%SIZE] ~= r
		st[i] = r
		i = (i + 1) % SIZE
	}

	// initial_mask: first 191 words only (lung excluded).
	u := mem.slice_ptr((^u64)(&s.status[0][0]), DSFMT_N * 2)
	for k in 0..<DSFMT_N * 2 {
		u[k] = (u[k] & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST
	}

	// period_certification (PCV2 is odd: single-bit fixup path).
	tmp0 := s.status[DSFMT_N][0] ~ DSFMT_FIX1
	tmp1 := s.status[DSFMT_N][1] ~ DSFMT_FIX2
	inner := (tmp0 & DSFMT_PCV1) ~ (tmp1 & 1)
	inner = inner ~ (inner >> 32)
	inner = inner ~ (inner >> 16)
	inner = inner ~ (inner >> 8)
	inner = inner ~ (inner >> 4)
	inner = inner ~ (inner >> 2)
	inner = inner ~ (inner >> 1)
	if inner & 1 == 0 {
		s.status[DSFMT_N][1] ~= 1
	}
	s.idx = DSFMT_N64
}

// Fill `out` (even length >= 382) with [1,2) doubles.
// Verbatim port of gen_rand_array_c1o2.
dsfmt_fill_c1o2 :: proc(s: ^Dsfmt_State, out: []f64) {
	assert(len(out) % 2 == 0 && len(out) >= DSFMT_N64)
	n := len(out) / 2 // w128 words
	aw := mem.slice_ptr((^Dsfmt_W128)(raw_data(out)), n)
	N := DSFMT_N
	P := DSFMT_POS1

	lung := s.status[N]
	dsfmt_rec(&aw[0], &s.status[0], &s.status[P], &lung)
	i := 1
	for i < N - P {
		dsfmt_rec(&aw[i], &s.status[i], &s.status[i+P], &lung)
		i += 1
	}
	for i < N {
		dsfmt_rec(&aw[i], &s.status[i], &aw[i+P-N], &lung)
		i += 1
	}
	for i < n - N {
		dsfmt_rec(&aw[i], &aw[i-N], &aw[i+P-N], &lung)
		i += 1
	}
	j := 0
	for j < 2*N - n {
		s.status[j] = aw[j + n - N]
		j += 1
	}
	for i < n {
		dsfmt_rec(&aw[i], &aw[i-N], &aw[i+P-N], &lung)
		s.status[j] = aw[i]
		i += 1
		j += 1
	}
	s.status[N] = lung
}

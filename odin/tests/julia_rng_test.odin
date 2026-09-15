package tests

import "core:testing"
import cpm "../cpm"

// Bit-exact Julia 1.12 MersenneTwister parity. Expected values were
// emitted by Julia 1.12.6 (`MersenneTwister(seed)`, sequential draws)
// and cross-checked across two independent runs. A wrong implementation
// cannot hit the post-refill values (float refill at draw 1003, ints
// refill inside the uint64 block).

@(test)
test_julia_rng_stream_42 :: proc(t: ^testing.T) {
	r := cpm.julia_rng_init(42)

	// First three Float64 bit patterns.
	want_f := []u64{4604577751753464416, 4589311072708242000, 4602279667678908688}
	for want in want_f {
		got := transmute(u64)cpm.jrand_f64(&r)
		testing.expectf(t, got == want, "float bits: got %d want %d", got, want)
	}
	// Draw through the float-cache refill (1002 live + refill at #1003).
	for _ in 0..<998 {
		cpm.jrand_f64(&r)
	}
	testing.expect(t, transmute(u64)cpm.jrand_f64(&r) == u64(4605180353918274470)) // #1002
	testing.expect(t, transmute(u64)cpm.jrand_f64(&r) == u64(4604628642923413984)) // #1003, post-refill
	testing.expect(t, transmute(u64)cpm.jrand_f64(&r) == u64(4602720123268723474)) // #1004
	for _ in 0..<46 {
		cpm.jrand_f64(&r)
	}
	// NDL range draws 1:40.
	want_i := []i64{23, 14, 14}
	for want in want_i {
		got := cpm.jrand_range(&r, 1, 40)
		testing.expectf(t, got == want, "range: got %d want %d", got, want)
	}
	for _ in 0..<57 {
		cpm.jrand_range(&r, 1, 40)
	}
	// Raw UInt64 pops, including across the ints-cache refill.
	testing.expect(t, cpm.jrand_u64(&r) == u64(9230128771654023970))
	testing.expect(t, cpm.jrand_u64(&r) == u64(10048489451684382574))
	for _ in 0..<999 {
		cpm.jrand_u64(&r)
	}
	testing.expect(t, cpm.jrand_u64(&r) == u64(12674727154709227483)) // post-refill
	testing.expect(t, cpm.jrand_u64(&r) == u64(15533950497023209428))
}

@(test)
test_julia_rng_stream_123 :: proc(t: ^testing.T) {
	r := cpm.julia_rng_init(123)
	testing.expect(t, transmute(u64)cpm.jrand_f64(&r) == u64(4600418692738201888))
	for _ in 0..<1049 {
		cpm.jrand_f64(&r)
	}
	testing.expect(t, cpm.jrand_range(&r, 1, 40) == 12)
	for _ in 0..<59 {
		cpm.jrand_range(&r, 1, 40)
	}
	testing.expect(t, cpm.jrand_u64(&r) == u64(7048500180185517165))
}

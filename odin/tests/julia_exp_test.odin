package tests

import "core:testing"
import cpm "../cpm"

// Bit-exact Julia 1.12 Base.Math exp(::Float64) parity. Expected outputs
// were emitted by Julia 1.12.6. Julia does NOT call libm/openlibm here —
// it evaluates base/special/exp.jl inline (table reduction + minimax
// kernel with fused multiply-adds), which differs from libm by 1 ulp on
// ~10% of inputs. The port replicates it with C fma() (correctly
// rounded, identical to hardware vfmadd).
@(test)
test_julia_exp :: proc(t: ^testing.T) {
	pairs := [][2]u64{
		{13870862551929061376, 1363076674615447813}, // t=-499.25 (libm differs here)
		{13870440339463995392, 1518986234140517167},
		{13866380942534246400, 2981170516415555511},
		{4642991313493426176, 6229569789378573007},
		{0, 4607182418800017408}, // exp(0) = 1
		{9223372036854775808, 4607182418800017408}, // exp(-0) = 1
		{4607182418800017408, 4613303445314885481}, // exp(1) = e
		{4158027847206421152, 4607182418800017408}, // 1e-30 -> 1
		{13381399884061196960, 4607182418800017408}, // -1e-30 -> 1
		{1, 4607182418800017408}, // 5e-324 -> 1
		{118622047889322841, 4607182418800017408}, // 1e-300 -> 1
		{4649454524316470215, 9218862018342910611}, // 709.782, just below overflow
		{4649454533112563237, 9218868437227405312}, // 709.783 -> +Inf
		{13873137485467395031, 1}, // -745.13 -> smallest subnormal
		{13873137573428325253, 0}, // -745.14 -> +0
		{9214871658872686752, 9218868437227405312}, // 1e308 -> +Inf
		{18438243695727462560, 0}, // -1e308 -> +0
		{9218868437227405312, 9218868437227405312}, // +Inf -> +Inf
		{18442240474082181120, 0}, // -Inf -> +0
		{0xBFE3333333333333, 4603118475304895716}, // exp(-0.6), the N=20 lane
	}
	for p in pairs {
		got := transmute(u64)cpm.jexp_f64(transmute(f64)p[0])
		testing.expectf(t, got == p[1], "exp(bits %d): got %d want %d", p[0], got, p[1])
	}
}

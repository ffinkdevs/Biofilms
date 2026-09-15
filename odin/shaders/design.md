# Compute vs SIMD vs raylib — where each part of this model belongs

This port ships three execution paths and one viewer. The mapping is
deliberate, not fashionable: each kernel runs where its memory pattern wins.

## 1. CPM Metropolis (`mcs_step`) — scalar CPU, all layouts

Per attempt: 2 random gathers (source + target site), a 26-neighbour
adhesion gather around the target (52 J-matrix lookups with a `(1-δ)`
branch each), 2 volume lookups, 2 `β·I` / melanin taps, one `exp`, one
branch. Arithmetic intensity ≈ 0.2 FLOP/byte, access pattern =
pointer-chase into a 64³–216³ lattice. On GPU this is latency-bound:
every thread diverges on the Metropolis test and atomics on cell volumes
serialize same-cell copies. The JACC port (`biofilms_potts_jacc.jl`)
already proved the point: it needs an 8-colour checkerboard to avoid
races, which changes the dynamics from random-sequential to
checkerboard-parallel (statistical, not pathwise, equivalence).

So: `mcs_step` stays scalar CPU here. All three layouts (AoS / SoA /
AoSoA) share the identical RNG consumption order, so the trajectory is
layout-independent — the thing the tests pin.

## 2. Field diffusion (melanin, nutrient) — SIMD on CPU, compute on GPU

`∂M/∂t = D∇²M + α·drive`, `∂C/∂t = D∇²C − uptake`. Per site: 6
neighbour loads, ~10 FLOPs, streaming access, no branches except the
wall pin. This is the bandwidth-bound stencil that SIMD and compute both
love:

- **CPU default (shipped):** `cpm/fields_simd.odin` processes 8-wide
  x-runs with `#simd[8]f32` (AVX2 width). Same width as the AoSoA block
  (`AOSOA_BLOCK = 8`) and the compute subgroup below, so tuning one
  width tunes all three. Scalar reference in `cpm/sim.odin` is kept for
  tests (tolerance 1e-5; SIMD reassociates the Laplacian sum).
- **GPU upgrade (provided, opt-in):** `shaders/field_diffusion.comp`
  runs the same stencil as a Vulkan compute dispatch
  (`8×4×4 = 128` threads/workgroup = 16 f32x8 vectors). Expected win:
  ~4–8× on the field sweeps at N=60, where the two sweeps dominate
  profile (~60% of wall time). The Metropolis step still runs on CPU
  and the lattice uploads once per MCS (N=40: 256 KB — negligible).

## 3. Single huge allocation (bindless)

`cpm/arena.odin` allocates **one** backing buffer and slices it:

```
[lattice i32][interior u8][radiation f32][melanin f32][drive f32][nutrient f32][contaminant f32][scratch f32]
```

every region 64-byte aligned, offsets recorded in the struct. The Vulkan
mapping is 1:1: one `VkBuffer` (host-visible, coherent), regions bound
by offset (push constants) or by `buffer_device_address` under
`VK_EXT_buffer_device_address` — one descriptor, one barrier, one
map/unmap per MCS. The render path reuses the same buffer: the voxel
instance list is a view over `lattice`, never a copy (raylib path
streams it; a full bindless renderer would issue
`vkCmdDrawIndirect` from a compacted index buffer — see §4).

## 4. Rendering: raylib today, bindless tomorrow

- **Shipped (`viewer/`):** raylib `DrawCube` + `DrawCubeWires` per
  occupied site. Correctness reference. Fine to ~10k sites at 60 fps;
  N=40 colonies (~5–8k sites) orbit smoothly. N=60 dense runs (~30k)
  will dip — that is the documented cue to switch to §5.
- **Headless (`render/ppm.odin`):** isometric software rasterizer,
  zero dependencies. CI geometry check + the PPM in this README.
- **Bindless (design, not shipped):** compact occupied indices on GPU
  (one compute pass over `lattice`), then instanced cube mesh with per-
  instance species colour via `gl_InstanceIndex` into a 7-entry palette
  UBO. One vertex buffer (unit cube), one instance buffer, one draw.
  The face-culling mask (`render.face_mask`) becomes a geometry-shader /
  fragment-stage `discard` on interior faces.

## 5. When to switch paths

| N     | Metropolis | Fields (SIMD) | Fields (compute) | Render (raylib) | Render (bindless) |
|-------|------------|---------------|------------------|-----------------|-------------------|
| ≤40   | CPU ✓      | CPU ✓         | no win          | ✓ 60 fps        | overkill          |
| 60    | CPU ✓      | CPU ✓         | ~2× if GPU idle | ~30 fps         | worth it          |
| ≥96   | CPU ✓      | consider GPU  | ✓               | instancing req. | required          |

Bottom line: **SIMD fields + raylib voxels is the sweet spot for this
model at publication scales.** Compute shaders pay off only for the
field sweeps at N≥60, and never for the Metropolis core.

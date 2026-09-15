package render

import "core:os"
import "core:fmt"
import "core:sort"
import cpm "../cpm"

// Headless software rasterizer: isometric voxel view -> binary PPM (P6).
// No GPU, no window, no font dependency — the CI geometry check. The
// raylib viewer (viewer/main.odin) is the publication view with text,
// orbit camera, and per-face lighting; this file proves the voxel list
// projects to something shaped like the screenshot (white background,
// shaded cubes, legend bars, MCS counter).

Image :: struct {
	w, h: int,
	px:   []u8, // RGB, row-major
}

image_make :: proc(w, h: int, allocator := context.allocator) -> Image {
	return Image{w = w, h = h, px = make([]u8, w * h * 3, allocator)}
}

image_free :: proc(img: ^Image, allocator := context.allocator) {
	delete(img.px, allocator)
}

put_pixel :: proc(img: ^Image, x, y: int, c: Species_RGB) {
	if x < 0 || y < 0 || x >= img.w || y >= img.h { return }
	i := (y * img.w + x) * 3
	img.px[i + 0] = c.r
	img.px[i + 1] = c.g
	img.px[i + 2] = c.b
}

clear :: proc(img: ^Image, c: Species_RGB) {
	for y in 0..<img.h {
		for x in 0..<img.w {
			put_pixel(img, x, y, c)
		}
	}
}

fill_rect :: proc(img: ^Image, x0, y0, w, h: int, c: Species_RGB) {
	for y in y0..<y0 + h {
		for x in x0..<x0 + w {
			put_pixel(img, x, y, c)
		}
	}
}

// Bresenham line, for cube edges + axes gizmo.
draw_line :: proc(img: ^Image, x0, y0, x1, y1: int, c: Species_RGB) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx := 1 if x0 < x1 else -1
	sy := 1 if y0 < y1 else -1
	err := dx + dy
	x, y := x0, y0
	for {
		put_pixel(img, x, y, c)
		if x == x1 && y == y1 { break }
		e2 := 2 * err
		if e2 >= dy { err += dy; x += sx }
		if e2 <= dx { err += dx; y += sy }
	}
}

P2 :: struct { x, y: f32 }

fill_triangle :: proc(img: ^Image, a, b, c: P2, col: Species_RGB) {
	// Bounding-box barycentric fill. Voxel faces are tiny (<40px), so
	// per-pixel barycentric is plenty fast and branchless-safe.
	minx := int(min(a.x, b.x, c.x))
	maxx := int(max(a.x, b.x, c.x)) + 1
	miny := int(min(a.y, b.y, c.y))
	maxy := int(max(a.y, b.y, c.y)) + 1
	den := (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
	if den == 0 { return }
	for y in miny..=maxy {
		for x in minx..=maxx {
			w1 := ((b.y - c.y) * (f32(x) - c.x) + (c.x - b.x) * (f32(y) - c.y)) / den
			w2 := ((c.y - a.y) * (f32(x) - c.x) + (a.x - c.x) * (f32(y) - c.y)) / den
			w3 := 1 - w1 - w2
			if w1 >= 0 && w2 >= 0 && w3 >= 0 {
				put_pixel(img, x, y, col)
			}
		}
	}
}

fill_quad :: proc(img: ^Image, a, b, c, d: P2, col: Species_RGB) {
	fill_triangle(img, a, b, c, col)
	fill_triangle(img, a, c, d, col)
}

stroke_quad :: proc(img: ^Image, a, b, c, d: P2, col: Species_RGB) {
	draw_line(img, int(a.x), int(a.y), int(b.x), int(b.y), col)
	draw_line(img, int(b.x), int(b.y), int(c.x), int(c.y), col)
	draw_line(img, int(c.x), int(c.y), int(d.x), int(d.y), col)
	draw_line(img, int(d.x), int(d.y), int(a.x), int(a.y), col)
}

// ── tiny 3x5 font (headless labels: MCS counter + legend names) ──
// Each glyph is 3 wide, 5 tall, row-major bits, bit 2 = left pixel.
// Black-on-white at scale 2-3; the legend chips alone can't name species.
font_glyph :: proc(ch: u8) -> [5]u8 {
	switch ch {
	case 'A': return {0b010, 0b101, 0b111, 0b101, 0b101}
	case 'B': return {0b110, 0b101, 0b110, 0b101, 0b110}
	case 'C': return {0b011, 0b100, 0b100, 0b100, 0b011}
	case 'D': return {0b110, 0b101, 0b101, 0b101, 0b110}
	case 'E': return {0b111, 0b100, 0b110, 0b100, 0b111}
	case 'F': return {0b111, 0b100, 0b110, 0b100, 0b100}
	case 'G': return {0b011, 0b100, 0b101, 0b101, 0b011}
	case 'H': return {0b101, 0b101, 0b111, 0b101, 0b101}
	case 'I': return {0b111, 0b010, 0b010, 0b010, 0b111}
	case 'J': return {0b001, 0b001, 0b001, 0b101, 0b010}
	case 'K': return {0b101, 0b101, 0b110, 0b101, 0b101}
	case 'L': return {0b100, 0b100, 0b100, 0b100, 0b111}
	case 'M': return {0b101, 0b111, 0b111, 0b101, 0b101}
	case 'N': return {0b110, 0b101, 0b101, 0b101, 0b101}
	case 'O': return {0b010, 0b101, 0b101, 0b101, 0b010}
	case 'P': return {0b110, 0b101, 0b110, 0b100, 0b100}
	case 'Q': return {0b010, 0b101, 0b101, 0b110, 0b011}
	case 'R': return {0b110, 0b101, 0b110, 0b101, 0b101}
	case 'S': return {0b011, 0b100, 0b010, 0b001, 0b110}
	case 'T': return {0b111, 0b010, 0b010, 0b010, 0b010}
	case 'U': return {0b101, 0b101, 0b101, 0b101, 0b111}
	case 'V': return {0b101, 0b101, 0b101, 0b101, 0b010}
	case 'W': return {0b101, 0b101, 0b101, 0b111, 0b101}
	case 'X': return {0b101, 0b101, 0b010, 0b101, 0b101}
	case 'Y': return {0b101, 0b101, 0b010, 0b010, 0b010}
	case 'Z': return {0b111, 0b001, 0b010, 0b100, 0b111}
	case 'a': return {0b000, 0b011, 0b001, 0b011, 0b011}
	case 'b': return {0b100, 0b100, 0b110, 0b101, 0b110}
	case 'c': return {0b000, 0b011, 0b100, 0b100, 0b011}
	case 'd': return {0b001, 0b001, 0b011, 0b101, 0b011}
	case 'e': return {0b000, 0b011, 0b111, 0b100, 0b011}
	case 'f': return {0b011, 0b100, 0b110, 0b100, 0b100}
	case 'g': return {0b000, 0b011, 0b101, 0b011, 0b001}
	case 'h': return {0b100, 0b100, 0b110, 0b101, 0b101}
	case 'i': return {0b010, 0b000, 0b010, 0b010, 0b010}
	case 'j': return {0b001, 0b000, 0b001, 0b001, 0b100}
	case 'k': return {0b100, 0b101, 0b110, 0b101, 0b101}
	case 'l': return {0b010, 0b010, 0b010, 0b010, 0b010}
	case 'm': return {0b000, 0b101, 0b111, 0b101, 0b101}
	case 'n': return {0b000, 0b110, 0b101, 0b101, 0b101}
	case 'o': return {0b000, 0b011, 0b101, 0b101, 0b011}
	case 'p': return {0b000, 0b110, 0b101, 0b110, 0b100}
	case 'q': return {0b000, 0b011, 0b101, 0b011, 0b001}
	case 'r': return {0b000, 0b101, 0b110, 0b100, 0b100}
	case 's': return {0b000, 0b011, 0b110, 0b001, 0b110}
	case 't': return {0b010, 0b010, 0b111, 0b010, 0b010}
	case 'u': return {0b000, 0b101, 0b101, 0b101, 0b011}
	case 'v': return {0b000, 0b101, 0b101, 0b101, 0b010}
	case 'w': return {0b000, 0b101, 0b101, 0b111, 0b101}
	case 'x': return {0b000, 0b101, 0b010, 0b101, 0b101}
	case 'y': return {0b000, 0b101, 0b101, 0b011, 0b001}
	case 'z': return {0b000, 0b111, 0b010, 0b100, 0b111}
	case '0': return {0b111, 0b101, 0b101, 0b101, 0b111}
	case '1': return {0b010, 0b110, 0b010, 0b010, 0b111}
	case '2': return {0b111, 0b001, 0b111, 0b100, 0b111}
	case '3': return {0b111, 0b001, 0b111, 0b001, 0b111}
	case '4': return {0b101, 0b101, 0b111, 0b001, 0b001}
	case '5': return {0b111, 0b100, 0b111, 0b001, 0b111}
	case '6': return {0b111, 0b100, 0b111, 0b101, 0b111}
	case '7': return {0b111, 0b001, 0b001, 0b010, 0b010}
	case '8': return {0b111, 0b101, 0b111, 0b101, 0b111}
	case '9': return {0b111, 0b101, 0b111, 0b001, 0b111}
	case '.': return {0b000, 0b000, 0b000, 0b000, 0b010}
	case '-': return {0b000, 0b000, 0b111, 0b000, 0b000}
	}
	return {0, 0, 0, 0, 0} // unknown -> space
}

draw_glyph :: proc(img: ^Image, x, y: int, g: [5]u8, col: Species_RGB, scale := 2) {
	for r in 0..<5 {
		for c in 0..<3 {
			if (g[r] >> u8(2 - c)) & 1 == 1 {
				fill_rect(img, x + c * scale, y + r * scale, scale, scale, col)
			}
		}
	}
}

// Advance per character, including 1px tracking.
text_width :: proc(s: string, scale := 2) -> int {
	return len(s) * 4 * scale - scale
}

draw_text :: proc(img: ^Image, x, y: int, s: string, col: Species_RGB, scale := 2) {
	for i in 0..<len(s) {
		draw_glyph(img, x + i * 4 * scale, y, font_glyph(s[i]), col, scale)
	}
}

draw_mcs_label :: proc(img: ^Image, mcs: int, col: Species_RGB) {
	buf: [16]u8
	draw_text(img, 16, 14, fmt.bprintf(buf[:], "MCS %d", mcs), col, 3)
}

// ── isometric projection ──
// Camera looks from (+x, +y, +z) toward the lattice centre (classic
// 30°/30° dimetric used by the screenshot: X runs down-right, Z runs
// down-left, Y runs up). View direction ~ normalize(1, 0.9, 1).
Iso_Params :: struct {
	cx, cy:   f32, // screen centre of lattice centre
	hs:       f32, // half-size of a voxel edge in px (cube scale)
	n:        int,
}

iso_project :: proc(p: Iso_Params, x, y, z: f32) -> P2 {
	c := f32(p.n) * 0.5
	dx := x - c
	dy := y - c
	dz := z - c
	// Dimetric: screen_x = (dx - dz) * cos30, screen_y = (dx + dz) * sin30 - dy.
	sx := p.cx + (dx - dz) * 0.8660254 * p.hs
	sy := p.cy + (dx + dz) * 0.5 * p.hs - dy * p.hs
	return P2{sx, sy}
}

iso_depth :: proc(n: int, x, y, z: f32) -> f32 {
	// Far (small depth) first: camera sits at +x,+y,+z, so depth grows
	// toward the camera. Painter's algorithm sorts ascending.
	return x * 0.577 + y * 0.577 + z * 0.577
}

Voxel_Depth :: struct {
	v: Voxel,
	d: f32,
}

// render_voxels draws every voxel as 3 shaded faces (top/left/right)
// with dark edges, back-to-front. Matches the screenshot's white
// background + outlined-cube look.
render_voxels :: proc(img: ^Image, s: ^cpm.Sim, vl: ^Voxel_List, mcs: int) {
	WHITE := Species_RGB{255, 255, 255}
	EDGE  := Species_RGB{60, 60, 60}
	INK   := Species_RGB{30, 30, 30}
	clear(img, WHITE)

	n := s.params.n
	// Fit lattice diagonal into ~62% of frame (leave right margin for legend).
	hs := f32(img.h) / (f32(n) * 2.1)
	if f32(img.w) * 0.62 / (f32(n) * 1.75) < hs {
		hs = f32(img.w) * 0.62 / (f32(n) * 1.75)
	}
	if hs < 2.0 { hs = 2.0 }
	ip := Iso_Params{cx = f32(img.w) * 0.38, cy = f32(img.h) * 0.52, hs = hs, n = n}

	// Depth sort (insertion into temp buffer + sort).
	order := make([]Voxel_Depth, vl.count, context.temp_allocator)
	defer free_all(context.temp_allocator)
	for i in 0..<vl.count {
		v := vl.items[i]
		order[i] = Voxel_Depth{v = v, d = iso_depth(n, v.x, v.y, v.z)}
	}
	sort.heap_sort_proc(order[:], proc(a, b: Voxel_Depth) -> int {
		if a.d < b.d { return -1 }
		if a.d > b.d { return 1 }
		return 0
	})

	hx := P2{hs * 0.5 * 0.8660254, hs * 0.25}
	hz := P2{-hs * 0.5 * 0.8660254, hs * 0.25}
	hy := P2{0, -hs * 0.5}

	for e in order {
		v := e.v
		// World mapping: sim Z (the cylinder/axial axis) points up, so
		// the bioreactor stands vertically like the reference view.
		base := iso_project(ip, v.x + 0.5, v.z + 0.5, v.y + 0.5)
		col := SPECIES_COLORS[v.species]
		top_c  := shade(col, SHADE_TOP)
		left_c := shade(col, SHADE_FRONT)
		right_c := shade(col, SHADE_SIDE)
		// 8 corners
		c000 := P2{base.x - hx.x - hz.x - hy.x, base.y - hx.y - hz.y - hy.y}
		_ = c000
		// Face centres from half-vectors: top face corners
		t0 := P2{base.x + hy.x - hx.x - hz.x, base.y + hy.y - hx.y - hz.y}
		t1 := P2{base.x + hy.x + hx.x - hz.x, base.y + hy.y + hx.y - hz.y}
		t2 := P2{base.x + hy.x + hx.x + hz.x, base.y + hy.y + hx.y + hz.y}
		t3 := P2{base.x + hy.x - hx.x + hz.x, base.y + hy.y - hx.y + hz.y}
		// Left face (+x side): corners t1,t2 shifted down by hy*2? Recompute:
		// bottom corners = top corners - 2*hy (hy points up).
		b1 := P2{t1.x - 2*hy.x, t1.y - 2*hy.y}
		b2 := P2{t2.x - 2*hy.x, t2.y - 2*hy.y}
		b3 := P2{t3.x - 2*hy.x, t3.y - 2*hy.y}
		b0 := P2{t0.x - 2*hy.x, t0.y - 2*hy.y}
		// Visible faces for this camera: top (t0..t3), +x side (t1,t2,b2,b1),
		// +z side (t2,t3,b3,b2). Back faces culled by construction.
		fill_quad(img, t1, t2, b2, b1, left_c)
		fill_quad(img, t2, t3, b3, b2, right_c)
		fill_quad(img, t0, t1, t2, t3, top_c)
		stroke_quad(img, t0, t1, t2, t3, EDGE)
		stroke_quad(img, t1, t2, b2, b1, EDGE)
		stroke_quad(img, t2, t3, b3, b2, EDGE)
		_ = b0
	}

	// MCS label (top-left, like screenshot).
	draw_mcs_label(img, mcs, INK)

	// Legend (right side): colour chip + species name in near-black,
	// matching the screenshot. Text is the label — chips alone can't
	// name species.
	BLACK := Species_RGB{20, 20, 20}
	lx := int(f32(img.w) * 0.78)
	ly := img.h - 7*26 - 20
	for sp in 0..<7 {
		fill_rect(img, lx, ly + sp*26, 14, 18, SPECIES_COLORS[sp])
		draw_line(img, lx, ly + sp*26, lx + 14, ly + sp*26, EDGE)
		draw_line(img, lx, ly + sp*26, lx, ly + sp*26 + 18, EDGE)
		draw_line(img, lx, ly + sp*26 + 18, lx + 14, ly + sp*26 + 18, EDGE)
		draw_line(img, lx + 14, ly + sp*26, lx + 14, ly + sp*26 + 18, EDGE)
		draw_text(img, lx + 22, ly + sp*26 + 4, cpm.SPECIES_NAMES[sp], BLACK, 2)
	}

	// Axes gizmo (bottom-left): X red down-right, Y pale up, Z green down-left.
	ox, oy := 70, img.h - 70
	draw_line(img, ox, oy, ox + 34, oy + 16, Species_RGB{200, 40, 40})  // X
	draw_line(img, ox, oy, ox, oy - 38, Species_RGB{180, 180, 60})      // Y
	draw_line(img, ox, oy, ox - 30, oy + 14, Species_RGB{40, 160, 60})  // Z
}

write_ppm :: proc(img: ^Image, path: string) -> bool {
	f, err := os.open(path, {.Write, .Create, .Trunc})
	if err != nil { return false }
	defer os.close(f)
	head := fmt.tprintf("P6\n%d %d\n255\n", img.w, img.h)
	os.write_string(f, head)
	os.write(f, img.px)
	return true
}

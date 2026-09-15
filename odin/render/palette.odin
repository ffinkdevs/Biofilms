package render

// Species palette. Matches FIG_COLORS in biofilms_potts.jl §13 and the
// screenshot legend (right side). sRGB 8-bit; the viewer converts to
// raylib Color, the PPM path writes bytes directly.
Species_RGB :: struct {
	r, g, b: u8,
}

SPECIES_COLORS := [7]Species_RGB{
	{0xe6, 0x19, 0x4b}, // CN — red     (C. neoformans)
	{0x3c, 0xb4, 0x4b}, // DR — green   (D. radiodurans)
	{0x43, 0x63, 0xd8}, // CS — blue    (C. sphaerospermum)
	{0xf5, 0x82, 0x31}, // BS — orange  (B. subtilis)
	{0x91, 0x1e, 0xb4}, // AN — purple  (A. niger)
	{0x42, 0xd4, 0xf4}, // SO — cyan    (S. oneidensis)
	{0xf0, 0x32, 0xe6}, // OI — magenta (O. intermedium)
}

// Face shading factors for the software rasterizer: top faces brightest
// (screenshot is lit from above), matching raylib default ambient.
SHADE_TOP   :: f32(1.00)
SHADE_FRONT :: f32(0.86)
SHADE_SIDE  :: f32(0.72)

shade :: proc(c: Species_RGB, f: f32) -> Species_RGB {
	return Species_RGB{
		u8(f32(c.r) * f),
		u8(f32(c.g) * f),
		u8(f32(c.b) * f),
	}
}

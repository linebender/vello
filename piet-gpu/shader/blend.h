// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Mode definitions and functions for blending and composition.

#define Blend_Normal 0
#define Blend_Multiply 1
#define Blend_Screen 2
#define Blend_Overlay 3
#define Blend_Darken 4
#define Blend_Lighten 5
#define Blend_ColorDodge 6
#define Blend_ColorBurn 7
#define Blend_HardLight 8
#define Blend_SoftLight 9
#define Blend_Difference 10
#define Blend_Exclusion 11
#define Blend_Hue 12
#define Blend_Saturation 13
#define Blend_Color 14
#define Blend_Luminosity 15
#define Blend_Clip 128

vec3 screen(vec3 cb, vec3 cs) {
	return cb + cs - (cb * cs);
}

float color_dodge(float cb, float cs) {
    if (cb == 0.0)
        return 0.0;
    else if (cs == 1.0)
        return 1.0;
    else
        return min(1.0, cb / (1.0 - cs));
}

float color_burn(float cb, float cs) {
    if (cb == 1.0)
        return 1.0;
    else if (cs == 0.0)
        return 0.0;
    else
        return 1.0 - min(1.0, (1.0 - cb) / cs);
}

vec3 hard_light(vec3 cb, vec3 cs) {
	return mix(
		screen(cb, 2.0 * cs - 1.0),
		cb * 2.0 * cs, 
		lessThanEqual(cs, vec3(0.5))
	);
}

vec3 soft_light(vec3 cb, vec3 cs) {
	vec3 d = mix(
		sqrt(cb),
		((16.0 * cb - vec3(12.0)) * cb + vec3(4.0)) * cb,
		lessThanEqual(cb, vec3(0.25))
	);
	return mix(
		cb + (2.0 * cs - vec3(1.0)) * (d - cb),
		cb - (vec3(1.0) - 2.0 * cs) * cb * (vec3(1.0) - cb),
		lessThanEqual(cs, vec3(0.5))
	);
}

float sat(vec3 c) {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

float lum(vec3 c) {
    vec3 f = vec3(0.3, 0.59, 0.11);
    return dot(c, f);
}

vec3 clip_color(vec3 c) {
    float L = lum(c);
    float n = min(c.r, min(c.g, c.b));
    float x = max(c.r, max(c.g, c.b));
    if (n < 0.0)
        c = L + (((c - L) * L) / (L - n));
    if (x > 1.0)
        c = L + (((c - L) * (1.0 - L)) / (x - L));
    return c;
}

vec3 set_lum(vec3 c, float l) {
    return clip_color(c + (l - lum(c)));
}

void set_sat_inner(inout float cmin, inout float cmid, inout float cmax, float s) {
    if (cmax > cmin) {
        cmid = (((cmid - cmin) * s) / (cmax - cmin));
        cmax = s;
    } else {
        cmid = 0.0;
        cmax = 0.0;
    }
    cmin = 0.0;
}

vec3 set_sat(vec3 c, float s) {
    if (c.r <= c.g) {
        if (c.g <= c.b) {
            set_sat_inner(c.r, c.g, c.b, s);
        } else {
            if (c.r <= c.b) {
                set_sat_inner(c.r, c.b, c.g, s);
            } else {
                set_sat_inner(c.b, c.r, c.g, s);
            }
        }
    } else {
        if (c.r <= c.b) {
            set_sat_inner(c.g, c.r, c.b, s);
        } else {
            if (c.g <= c.b) {
                set_sat_inner(c.g, c.b, c.r, s);
            } else {
                set_sat_inner(c.b, c.g, c.r, s);
            }
        }
    }
    return c;
}

// Blends two RGB colors together. The colors are assumed to be in sRGB
// color space, and this function does not take alpha into account.
vec3 mix_blend(vec3 cb, vec3 cs, uint mode) {
	vec3 b = vec3(0.0);
	switch (mode) {
	case Blend_Multiply:
		b = cb * cs;
		break;
	case Blend_Screen:
		b = screen(cb, cs);
		break;
	case Blend_Overlay:
		b = hard_light(cs, cb);
		break;
	case Blend_Darken:
		b = min(cb, cs);
		break;
	case Blend_Lighten:
		b = max(cb, cs);
		break;
	case Blend_ColorDodge:
		b = vec3(color_dodge(cb.x, cs.x), color_dodge(cb.y, cs.y), color_dodge(cb.z, cs.z));
		break;
	case Blend_ColorBurn:
		b = vec3(color_burn(cb.x, cs.x), color_burn(cb.y, cs.y), color_burn(cb.z, cs.z));
		break;
	case Blend_HardLight:
		b = hard_light(cb, cs);
		break;
	case Blend_SoftLight:
		b = soft_light(cb, cs);
		break;
	case Blend_Difference:
		b = abs(cb - cs);
		break;
	case Blend_Exclusion:
		b = cb + cs - 2 * cb * cs;
		break;
	case Blend_Hue:
		b = set_lum(set_sat(cs, sat(cb)), lum(cb));
		break;
	case Blend_Saturation:
		b = set_lum(set_sat(cb, sat(cs)), lum(cb));
		break;
	case Blend_Color:
		b = set_lum(cs, lum(cb));
		break;
	case Blend_Luminosity:
		b = set_lum(cb, lum(cs));
		break;
	default:
		b = cs;
		break;
	}
	return b;
}

#define Comp_Clear 0
#define Comp_Copy 1
#define Comp_Dest 2
#define Comp_SrcOver 3
#define Comp_DestOver 4
#define Comp_SrcIn 5
#define Comp_DestIn 6
#define Comp_SrcOut 7
#define Comp_DestOut 8
#define Comp_SrcAtop 9
#define Comp_DestAtop 10
#define Comp_Xor 11
#define Comp_Plus 12
#define Comp_PlusLighter 13

// Apply general compositing operation.
// Inputs are separated colors and alpha, output is premultiplied.
vec4 mix_compose(vec3 cb, vec3 cs, float ab, float as, uint mode) {
	float fa = 0.0;
	float fb = 0.0;
	switch (mode) {
	case Comp_Copy:
		fa = 1.0;
		fb = 0.0;
		break;
	case Comp_Dest:
		fa = 0.0;
		fb = 1.0;
		break;
	case Comp_SrcOver:
		fa = 1.0;
		fb = 1.0 - as;
		break;
	case Comp_DestOver:
		fa = 1.0 - ab;
		fb = 1.0;
		break;
	case Comp_SrcIn:
		fa = ab;
		fb = 0.0;
		break;
	case Comp_DestIn:
		fa = 0.0;
		fb = as;
		break;
	case Comp_SrcOut:
		fa = 1.0 - ab;
		fb = 0.0;
		break;
	case Comp_DestOut:
		fa = 0.0;
		fb = 1.0 - as;
		break;
	case Comp_SrcAtop:
		fa = ab;
		fb = 1.0 - as;
		break;
	case Comp_DestAtop:
		fa = 1.0 - ab;
		fb = as;
		break;
	case Comp_Xor:
		fa = 1.0 - ab;
		fb = 1.0 - as;
		break;
	case Comp_Plus:
		fa = 1.0;
		fb = 1.0;
		break;
	case Comp_PlusLighter:
		return min(vec4(1.0), vec4(as * cs + ab * cb, as + ab));
	default:
		break;
	}
	float as_fa = as * fa;
	float ab_fb = ab * fb;
	vec3 co = as_fa * cs + ab_fb * cb;
	return vec4(co, as_fa + ab_fb);
}

#define BlendComp_default (Blend_Normal << 8 | Comp_SrcOver)
#define BlendComp_clip (Blend_Clip << 8 | Comp_SrcOver)

// This is added to alpha to prevent divide-by-zero
#define EPSILON 1e-15

// Apply blending and composition. Both input and output colors are
// premultiplied RGB.
vec4 mix_blend_compose(vec4 backdrop, vec4 src, uint mode) {
	if ((mode & 0x7fff) == BlendComp_default) {
		// Both normal+src_over blend and clip case
		return backdrop * (1.0 - src.a) + src;
	}
	// Un-premultiply colors for blending
	float inv_src_a = 1.0 / (src.a + EPSILON);
	vec3 cs = src.rgb * inv_src_a;
	float inv_backdrop_a = 1.0 / (backdrop.a + EPSILON);
	vec3 cb = backdrop.rgb * inv_backdrop_a;
	uint blend_mode = mode >> 8;
	vec3 blended = mix_blend(cb, cs, blend_mode);
	cs = mix(cs, blended, backdrop.a);
	uint comp_mode = mode & 0xff;
	if (comp_mode == Comp_SrcOver) {
		vec3 co = mix(backdrop.rgb, cs, src.a);
		return vec4(co, src.a + backdrop.a * (1 - src.a));
	} else {
		return mix_compose(cb, cs, backdrop.a, src.a, comp_mode);
	}
}

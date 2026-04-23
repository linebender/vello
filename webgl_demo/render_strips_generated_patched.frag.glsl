#version 300 es

precision highp float;
precision highp int;

struct Config {
    uint width;
    uint height;
    uint strip_height;
    uint alphas_tex_width_bits;
    uint encoded_paints_tex_width_bits;
    int strip_offset_x;
    int strip_offset_y;
    uint ndc_y_negate;
};
struct StripInstance {
    uint xy;
    uint widths_or_rect_height;
    uint col_idx_or_rect_frac;
    uint payload;
    uint paint_and_rect_flag;
    uint depth_index;
};
struct VertexOutput {
    uint paint_and_rect_flag;
    vec2 tex_coord;
    vec2 sample_xy;
    uint dense_end_or_rect_size;
    uint payload;
    uint rect_frac;
    vec4 position;
};
struct EncodedImage {
    uint quality;
    uvec2 extend_modes;
    vec2 image_size;
    vec2 image_offset;
    uint atlas_index;
    mat2x2 transform;
    vec2 translate;
    vec4 tint;
    uint tint_mode;
    float image_padding;
};
struct LinearGradient {
    uint extend_mode;
    uint gradient_start;
    uint texture_width;
    mat2x2 transform;
    vec2 translate;
};
struct RadialGradient {
    uint extend_mode;
    uint gradient_start;
    uint texture_width;
    mat2x2 transform;
    vec2 translate;
    float bias;
    float scale;
    float fp0_;
    float fp1_;
    float fr1_;
    float f_focal_x;
    uint f_is_swapped;
    float scaled_r0_squared;
    uint kind;
};
struct RadialGradientResult {
    float t_value;
    bool is_valid;
};
struct SweepGradient {
    uint extend_mode;
    uint gradient_start;
    uint texture_width;
    mat2x2 transform;
    vec2 translate;
    float start_angle;
    float inv_angle_delta;
};
const uint COLOR_SOURCE_PAYLOAD = 0u;
const uint COLOR_SOURCE_SLOT = 1u;
const uint COLOR_SOURCE_BLEND = 2u;
const uint PAINT_TYPE_SOLID = 0u;
const uint PAINT_TYPE_IMAGE = 1u;
const uint PAINT_TYPE_LINEAR_GRADIENT = 2u;
const uint PAINT_TYPE_RADIAL_GRADIENT = 3u;
const uint PAINT_TYPE_SWEEP_GRADIENT = 4u;
const uint PAINT_TEXTURE_INDEX_MASK = 67108863u;
const uint RECT_STRIP_FLAG = 2147483648u;
const uint IMAGE_QUALITY_LOW = 0u;
const uint IMAGE_QUALITY_MEDIUM = 1u;
const uint IMAGE_QUALITY_HIGH = 2u;
const uint GRADIENT_TYPE_LINEAR = 0u;
const uint GRADIENT_TYPE_RADIAL = 1u;
const uint GRADIENT_TYPE_SWEEP = 2u;
const uint RADIAL_GRADIENT_TYPE_STANDARD = 0u;
const uint RADIAL_GRADIENT_TYPE_STRIP = 1u;
const uint RADIAL_GRADIENT_TYPE_FOCAL = 2u;
const float PI = 3.1415927;
const float TWO_PI = 6.2831855;
const float NEARLY_ZERO_TOLERANCE = 0.00024414063;
const uint COMPOSE_CLEAR = 0u;
const uint COMPOSE_COPY = 1u;
const uint COMPOSE_DEST = 2u;
const uint COMPOSE_SRC_OVER = 3u;
const uint COMPOSE_DEST_OVER = 4u;
const uint COMPOSE_SRC_IN = 5u;
const uint COMPOSE_DEST_IN = 6u;
const uint COMPOSE_SRC_OUT = 7u;
const uint COMPOSE_DEST_OUT = 8u;
const uint COMPOSE_SRC_ATOP = 9u;
const uint COMPOSE_DEST_ATOP = 10u;
const uint COMPOSE_XOR = 11u;
const uint COMPOSE_PLUS = 12u;
const uint COMPOSE_PLUS_LIGHTER = 13u;
const uint MIX_NORMAL = 0u;
const uint MIX_MULTIPLY = 1u;
const uint MIX_SCREEN = 2u;
const uint MIX_OVERLAY = 3u;
const uint MIX_DARKEN = 4u;
const uint MIX_LIGHTEN = 5u;
const uint MIX_COLOR_DODGE = 6u;
const uint MIX_COLOR_BURN = 7u;
const uint MIX_HARD_LIGHT = 8u;
const uint MIX_SOFT_LIGHT = 9u;
const uint MIX_DIFFERENCE = 10u;
const uint MIX_EXCLUSION = 11u;
const uint MIX_HUE = 12u;
const uint MIX_SATURATION = 13u;
const uint MIX_COLOR = 14u;
const uint MIX_LUMINOSITY = 15u;
const uint TINT_MODE_MULTIPLY = 1u;
const uint EXTEND_PAD = 0u;
const uint EXTEND_REPEAT = 1u;
const uint EXTEND_REFLECT = 2u;
const vec4 MF[4] = vec4[4](vec4(0.055555556, -0.5, 0.8333333, -0.3888889), vec4(0.8888889, 0.0, -2.0, 1.1666666), vec4(0.055555556, 0.5, 1.5, -1.1666666), vec4(0.0, 0.0, -0.33333334, 0.3888889));
const uint TINT_MODE_ALPHA_MASK = 0u;

layout(std140) uniform Config_block_0Fragment { Config _group_0_binding_1_fs; };

uniform highp sampler2DArray _group_1_binding_0_fs;

uniform highp usampler2D _group_2_binding_0_fs;

uniform highp sampler2D _group_3_binding_0_fs;

uniform highp usampler2D _group_0_binding_0_fs;

uniform highp sampler2D _group_0_binding_2_fs;

flat in uint _vs2fs_location0;
smooth in vec2 _vs2fs_location1;
smooth in vec2 _vs2fs_location2;
flat in uint _vs2fs_location3;
flat in uint _vs2fs_location4;
flat in uint _vs2fs_location5;
layout(location = 0) out vec4 _fs2p_location0;

uvec2 encoded_paint_coord(uint flat_idx) {
    uint _e4 = _group_0_binding_1_fs.encoded_paints_tex_width_bits;
    uint _e11 = _group_0_binding_1_fs.encoded_paints_tex_width_bits;
    return uvec2((flat_idx & ((1u << _e4) - 1u)), (flat_idx >> _e11));
}

EncodedImage unpack_encoded_image(uint paint_tex_idx) {
    uvec2 _e2 = encoded_paint_coord(paint_tex_idx);
    uvec4 texel0_ = texelFetch(_group_2_binding_0_fs, ivec2(_e2), 0);
    uvec2 _e8 = encoded_paint_coord((paint_tex_idx + 1u));
    uvec4 texel1_ = texelFetch(_group_2_binding_0_fs, ivec2(_e8), 0);
    uvec2 _e14 = encoded_paint_coord((paint_tex_idx + 2u));
    uvec4 texel2_ = texelFetch(_group_2_binding_0_fs, ivec2(_e14), 0);
    uint quality = (texel0_.x & 3u);
    uint extend_x = ((texel0_.x >> 2u) & 3u);
    uint extend_y = ((texel0_.x >> 4u) & 3u);
    uint atlas_index = ((texel0_.x >> 6u) & 255u);
    vec2 image_size_2 = vec2(float((texel0_.y >> 16u)), float((texel0_.y & 65535u)));
    vec2 image_offset_2 = vec2(float((texel0_.z >> 16u)), float((texel0_.z & 65535u)));
    mat2x2 transform = mat2x2(vec2(uintBitsToFloat(texel0_.w), uintBitsToFloat(texel1_.x)), vec2(uintBitsToFloat(texel1_.y), uintBitsToFloat(texel1_.z)));
    vec2 translate = vec2(uintBitsToFloat(texel1_.w), uintBitsToFloat(texel2_.x));
    uint packed_tint = texel2_.y;
    vec4 tint = ((packed_tint != 0u) ? (vec4(packed_tint & 0xFFu, packed_tint >> 8 & 0xFFu, packed_tint >> 16 & 0xFFu, packed_tint >> 24) / 255.0) : vec4(1.0));
    uint tint_mode = ((packed_tint != 0u) ? texel2_.z : TINT_MODE_MULTIPLY);
    float image_padding_2 = float(texel2_.w);
    return EncodedImage(quality, uvec2(extend_x, extend_y), image_size_2, image_offset_2, atlas_index, transform, translate, tint, tint_mode, image_padding_2);
}

uint unpack_alphas_from_channel(uvec4 rgba, uint channel_index) {
    switch(channel_index) {
        case 0u: {
            return rgba.x;
        }
        case 1u: {
            return rgba.y;
        }
        case 2u: {
            return rgba.z;
        }
        case 3u: {
            return rgba.w;
        }
        default: {
            return rgba.x;
        }
    }
}

float extend_mode_normalized(float t, uint mode) {
    switch(mode) {
        case 0u: {
            return clamp(t, 0.0, 1.0);
        }
        case 1u: {
            return fract(t);
        }
        case 2u:
        default: {
            return abs((t - (2.0 * roundEven((0.5 * t)))));
        }
    }
}

float extend_mode(float t_1, uint mode_1, float max_) {
    switch(mode_1) {
        case 0u: {
            return clamp(t_1, 0.0, (max_ - 1.0));
        }
        case 1u: {
            float _e8 = extend_mode_normalized((t_1 / max_), mode_1);
            return (_e8 * max_);
        }
        case 2u:
        default: {
            float _e11 = extend_mode_normalized((t_1 / max_), mode_1);
            return (_e11 * max_);
        }
    }
}

float single_weight(float t_2, float a, float b, float c, float d) {
    return ((t_2 * ((t_2 * ((t_2 * d) + c)) + b)) + a);
}

vec4 cubic_weights(float fract_) {
    float _e5 = single_weight(fract_, 0.055555556, -0.5, 0.8333333, -0.3888889);
    float _e10 = single_weight(fract_, 0.8888889, 0.0, -2.0, 1.1666666);
    float _e15 = single_weight(fract_, 0.055555556, 0.5, 1.5, -1.1666666);
    float _e20 = single_weight(fract_, 0.0, 0.0, -0.33333334, 0.3888889);
    return vec4(_e5, _e10, _e15, _e20);
}

vec4 bicubic_sample(highp sampler2DArray tex, vec2 coords, int atlas_idx, vec2 image_offset, vec2 image_size, uvec2 extend_modes, float image_padding) {
    vec2 atlas_max = ((image_offset + image_size) - vec2(1.0));
    vec2 frac_coords = fract((coords + vec2(0.5)));
    vec4 _e16 = cubic_weights(frac_coords.x);
    vec4 _e18 = cubic_weights(frac_coords.y);
    vec4 s00_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-1.5, -1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s10_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-0.5, -1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s20_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(0.5, -1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s30_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(1.5, -1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s01_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-1.5, -0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s11_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-0.5, -0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s21_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(0.5, -0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s31_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(1.5, -0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s02_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-1.5, 0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s12_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-0.5, 0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s22_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(0.5, 0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s32_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(1.5, 0.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s03_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-1.5, 1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s13_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(-0.5, 1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s23_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(0.5, 1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 s33_ = texelFetch(tex, ivec3(ivec2(clamp((coords + vec2(1.5, 1.5)), image_offset, atlas_max)), atlas_idx), 0);
    vec4 row0_ = ((((_e16.x * s00_) + (_e16.y * s10_)) + (_e16.z * s20_)) + (_e16.w * s30_));
    vec4 row1_ = ((((_e16.x * s01_) + (_e16.y * s11_)) + (_e16.z * s21_)) + (_e16.w * s31_));
    vec4 row2_ = ((((_e16.x * s02_) + (_e16.y * s12_)) + (_e16.z * s22_)) + (_e16.w * s32_));
    vec4 row3_ = ((((_e16.x * s03_) + (_e16.y * s13_)) + (_e16.z * s23_)) + (_e16.w * s33_));
    vec4 result = ((((_e18.x * row0_) + (_e18.y * row1_)) + (_e18.z * row2_)) + (_e18.w * row3_));
    float a_1 = clamp(result.w, 0.0, 1.0);
    return vec4(clamp(result.xyz, vec3(0.0), vec3(a_1)), a_1);
}

vec4 bilinear_sample(highp sampler2DArray tex_1, vec2 coords_1, int atlas_idx_1, vec2 image_offset_1, vec2 image_size_1, uvec2 extend_modes_1, float image_padding_1) {
    vec2 atlas_max_1 = ((image_offset_1 + image_size_1) - vec2(1.0));
    vec2 atlas_uv_clamped = clamp(coords_1, image_offset_1, atlas_max_1);
    vec4 uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
    vec2 uv_frac = fract(coords_1);
    vec4 a_2 = texelFetch(tex_1, ivec3(ivec2(uv_quad.xy), atlas_idx_1), 0);
    vec4 b_3 = texelFetch(tex_1, ivec3(ivec2(uv_quad.xw), atlas_idx_1), 0);
    vec4 c_6 = texelFetch(tex_1, ivec3(ivec2(uv_quad.zy), atlas_idx_1), 0);
    vec4 d_1 = texelFetch(tex_1, ivec3(ivec2(uv_quad.zw), atlas_idx_1), 0);
    return mix(mix(a_2, b_3, uv_frac.y), mix(c_6, d_1, uv_frac.y), uv_frac.x);
}

uvec2 unpack_texture_width_and_extend_mode(uint packed_) {
    uint texture_width_1 = (packed_ & 268435455u);
    uint extend_mode_2 = ((packed_ >> 30u) & 3u);
    return uvec2(texture_width_1, extend_mode_2);
}

LinearGradient unpack_linear_gradient(uint paint_tex_idx_1) {
    uvec2 _e2 = encoded_paint_coord(paint_tex_idx_1);
    uvec4 texel0_1 = texelFetch(_group_2_binding_0_fs, ivec2(_e2), 0);
    uvec2 _e8 = encoded_paint_coord((paint_tex_idx_1 + 1u));
    uvec4 texel1_1 = texelFetch(_group_2_binding_0_fs, ivec2(_e8), 0);
    uvec2 _e12 = unpack_texture_width_and_extend_mode(texel0_1.x);
    uint texture_width_2 = _e12.x;
    uint extend_mode_3 = _e12.y;
    uint gradient_start_1 = texel0_1.y;
    mat2x2 transform_1 = mat2x2(vec2(uintBitsToFloat(texel0_1.z), uintBitsToFloat(texel0_1.w)), vec2(uintBitsToFloat(texel1_1.x), uintBitsToFloat(texel1_1.y)));
    vec2 translate_1 = vec2(uintBitsToFloat(texel1_1.z), uintBitsToFloat(texel1_1.w));
    return LinearGradient(extend_mode_3, gradient_start_1, texture_width_2, transform_1, translate_1);
}

vec4 sample_gradient_lut(float t_value, uint extend_mode_1, uint gradient_start, uint texture_width) {
    float _e4 = extend_mode_normalized(t_value, extend_mode_1);
    uint t_offset = uint((_e4 * float((texture_width - 1u))));
    uint flat_coord = (gradient_start + t_offset);
    uint gradient_tex_width = uvec2(textureSize(_group_3_binding_0_fs, 0).xy).x;
    uint tex_x = (flat_coord % gradient_tex_width);
    uint tex_y = (flat_coord / gradient_tex_width);
    vec4 gradient_color = texelFetch(_group_3_binding_0_fs, ivec2(uvec2(tex_x, tex_y)), 0);
    return gradient_color;
}

uvec2 unpack_radial_kind_and_swapped(uint packed_1) {
    uint kind = (packed_1 & 3u);
    uint f_is_swapped = ((packed_1 >> 2u) & 1u);
    return uvec2(kind, f_is_swapped);
}

RadialGradient unpack_radial_gradient(uint paint_tex_idx_2) {
    uvec2 _e2 = encoded_paint_coord(paint_tex_idx_2);
    uvec4 texel0_2 = texelFetch(_group_2_binding_0_fs, ivec2(_e2), 0);
    uvec2 _e8 = encoded_paint_coord((paint_tex_idx_2 + 1u));
    uvec4 texel1_2 = texelFetch(_group_2_binding_0_fs, ivec2(_e8), 0);
    uvec2 _e14 = encoded_paint_coord((paint_tex_idx_2 + 2u));
    uvec4 texel2_1 = texelFetch(_group_2_binding_0_fs, ivec2(_e14), 0);
    uvec2 _e20 = encoded_paint_coord((paint_tex_idx_2 + 3u));
    uvec4 texel3_ = texelFetch(_group_2_binding_0_fs, ivec2(_e20), 0);
    uvec2 _e24 = unpack_texture_width_and_extend_mode(texel0_2.x);
    uint texture_width_3 = _e24.x;
    uint extend_mode_4 = _e24.y;
    uint gradient_start_2 = texel0_2.y;
    mat2x2 transform_2 = mat2x2(vec2(uintBitsToFloat(texel0_2.z), uintBitsToFloat(texel0_2.w)), vec2(uintBitsToFloat(texel1_2.x), uintBitsToFloat(texel1_2.y)));
    vec2 translate_2 = vec2(uintBitsToFloat(texel1_2.z), uintBitsToFloat(texel1_2.w));
    uvec2 _e45 = unpack_radial_kind_and_swapped(texel2_1.x);
    uint kind_1 = _e45.x;
    uint f_is_swapped_1 = _e45.y;
    float bias = uintBitsToFloat(texel2_1.y);
    float scale = uintBitsToFloat(texel2_1.z);
    float fp0_ = uintBitsToFloat(texel2_1.w);
    float fp1_ = uintBitsToFloat(texel3_.x);
    float fr1_ = uintBitsToFloat(texel3_.y);
    float f_focal_x = uintBitsToFloat(texel3_.z);
    float scaled_r0_squared = uintBitsToFloat(texel3_.w);
    return RadialGradient(extend_mode_4, gradient_start_2, texture_width_3, transform_2, translate_2, bias, scale, fp0_, fp1_, fr1_, f_focal_x, f_is_swapped_1, scaled_r0_squared, kind_1);
}

RadialGradientResult calculate_radial_gradient(vec2 grad_pos_1, RadialGradient radial_gradient) {
    float t_value_1 = 0.0;
    bool is_valid = false;
    float t_3 = 0.0;
    bool local_4 = false;
    bool local_5 = false;
    bool local_6 = false;
    bool local_7 = false;
    float x_pos = grad_pos_1.x;
    float y_pos = grad_pos_1.y;
    switch(radial_gradient.kind) {
        case 0u: {
            float radius = sqrt(((x_pos * x_pos) + (y_pos * y_pos)));
            t_value_1 = (radial_gradient.bias + (radial_gradient.scale * radius));
            is_valid = true;
            break;
        }
        case 1u: {
            float p1_ = (radial_gradient.scaled_r0_squared - (y_pos * y_pos));
            is_valid = (p1_ >= 0.0);
            bool _e21 = is_valid;
            if (_e21) {
                t_value_1 = (x_pos + sqrt(p1_));
            } else {
                t_value_1 = 0.0;
            }
            break;
        }
        case 2u:
        default: {
            float fp0_1 = radial_gradient.fp0_;
            float fp1_1 = radial_gradient.fp1_;
            float fr1_1 = radial_gradient.fr1_;
            float f_focal_x_1 = radial_gradient.f_focal_x;
            uint is_swapped = radial_gradient.f_is_swapped;
            bool is_focal_on_circle = (abs((1.0 - fr1_1)) <= NEARLY_ZERO_TOLERANCE);
            if (!(is_focal_on_circle)) {
                local_4 = (fr1_1 > 1.0);
            } else {
                local_4 = false;
            }
            bool is_well_behaved = local_4;
            bool is_natively_focal = (abs(f_focal_x_1) <= NEARLY_ZERO_TOLERANCE);
            is_valid = true;
            if (is_focal_on_circle) {
                t_3 = (x_pos + ((y_pos * y_pos) / x_pos));
                float _e51 = t_3;
                if ((_e51 >= 0.0)) {
                    local_5 = (x_pos != 0.0);
                } else {
                    local_5 = false;
                }
                bool _e59 = local_5;
                is_valid = _e59;
            } else {
                if (is_well_behaved) {
                    t_3 = (sqrt(((x_pos * x_pos) + (y_pos * y_pos))) - (x_pos * fp0_1));
                } else {
                    float xx = (x_pos * x_pos);
                    float yy = (y_pos * y_pos);
                    float discriminant = (xx - yy);
                    if (!((is_swapped != 0u))) {
                        local_6 = ((1.0 - f_focal_x_1) < 0.0);
                    } else {
                        local_6 = true;
                    }
                    bool _e79 = local_6;
                    if (_e79) {
                        t_3 = (-(sqrt(discriminant)) - (x_pos * fp0_1));
                    } else {
                        t_3 = (sqrt(discriminant) - (x_pos * fp0_1));
                    }
                    if ((discriminant >= 0.0)) {
                        float _e91 = t_3;
                        local_7 = (_e91 >= 0.0);
                    } else {
                        local_7 = false;
                    }
                    bool _e95 = local_7;
                    is_valid = _e95;
                }
            }
            bool _e96 = is_valid;
            if (_e96) {
                if (((1.0 - f_focal_x_1) < 0.0)) {
                    float _e101 = t_3;
                    t_3 = -(_e101);
                }
                if (!(is_natively_focal)) {
                    float _e104 = t_3;
                    t_3 = (_e104 + fp1_1);
                }
                if ((is_swapped != 0u)) {
                    float _e108 = t_3;
                    t_3 = (1.0 - _e108);
                }
            }
            float _e111 = t_3;
            t_value_1 = _e111;
            break;
        }
    }
    float _e112 = t_value_1;
    bool _e113 = is_valid;
    return RadialGradientResult(_e112, _e113);
}

SweepGradient unpack_sweep_gradient(uint paint_tex_idx_3) {
    uvec2 _e2 = encoded_paint_coord(paint_tex_idx_3);
    uvec4 texel0_3 = texelFetch(_group_2_binding_0_fs, ivec2(_e2), 0);
    uvec2 _e8 = encoded_paint_coord((paint_tex_idx_3 + 1u));
    uvec4 texel1_3 = texelFetch(_group_2_binding_0_fs, ivec2(_e8), 0);
    uvec2 _e14 = encoded_paint_coord((paint_tex_idx_3 + 2u));
    uvec4 texel2_2 = texelFetch(_group_2_binding_0_fs, ivec2(_e14), 0);
    uvec2 _e18 = unpack_texture_width_and_extend_mode(texel0_3.x);
    uint texture_width_4 = _e18.x;
    uint extend_mode_5 = _e18.y;
    uint gradient_start_3 = texel0_3.y;
    mat2x2 transform_3 = mat2x2(vec2(uintBitsToFloat(texel0_3.z), uintBitsToFloat(texel0_3.w)), vec2(uintBitsToFloat(texel1_3.x), uintBitsToFloat(texel1_3.y)));
    vec2 translate_3 = vec2(uintBitsToFloat(texel1_3.z), uintBitsToFloat(texel1_3.w));
    float start_angle = uintBitsToFloat(texel2_2.x);
    float inv_angle_delta = uintBitsToFloat(texel2_2.y);
    return SweepGradient(extend_mode_5, gradient_start_3, texture_width_4, transform_3, translate_3, start_angle, inv_angle_delta);
}

float xy_to_unit_angle(float x, float y) {
    float phi = 0.0;
    float xabs = abs(x);
    float yabs = abs(y);
    float slope = (min(xabs, yabs) / max(xabs, yabs));
    float s_2 = (slope * slope);
    phi = (slope * (0.15912117 + (s_2 * (-0.05185397 + (s_2 * (0.02476102 + (s_2 * -0.0070547382)))))));
    float _e20 = phi;
    float _e21 = phi;
    phi = ((xabs < yabs) ? (0.25 - _e21) : _e20);
    float _e26 = phi;
    float _e27 = phi;
    phi = ((x < 0.0) ? (0.5 - _e27) : _e26);
    float _e33 = phi;
    float _e34 = phi;
    phi = ((y < 0.0) ? (1.0 - _e34) : _e33);
    float _e40 = phi;
    float _e41 = phi;
    float _e42 = phi;
    phi = ((_e41 != _e42) ? 0.0 : _e40);
    float _e46 = phi;
    return _e46;
}

vec3 screen(vec3 cb, vec3 cs) {
    return ((cb + cs) - (cb * cs));
}

vec3 hard_light(vec3 cb_1, vec3 cs_1) {
    vec3 _e7 = screen(cb_1, ((2.0 * cs_1) - vec3(1.0)));
    return mix(_e7, ((cb_1 * 2.0) * cs_1), lessThanEqual(cs_1, vec3(0.5)));
}

float color_dodge(float cb_2, float cs_2) {
    if ((cb_2 == 0.0)) {
        return 0.0;
    } else {
        if ((cs_2 == 1.0)) {
            return 1.0;
        } else {
            return min(1.0, (cb_2 / (1.0 - cs_2)));
        }
    }
}

float color_burn(float cb_3, float cs_3) {
    if ((cb_3 == 1.0)) {
        return 1.0;
    } else {
        if ((cs_3 == 0.0)) {
            return 0.0;
        } else {
            return (1.0 - min(1.0, ((1.0 - cb_3) / cs_3)));
        }
    }
}

vec3 soft_light(vec3 cb_4, vec3 cs_4) {
    vec3 d_2 = mix(sqrt(cb_4), (((((16.0 * cb_4) - vec3(12.0)) * cb_4) + vec3(4.0)) * cb_4), lessThanEqual(cb_4, vec3(0.25)));
    return mix((cb_4 + (((2.0 * cs_4) - vec3(1.0)) * (d_2 - cb_4))), (cb_4 - (((vec3(1.0) - (2.0 * cs_4)) * cb_4) * (vec3(1.0) - cb_4))), lessThanEqual(cs_4, vec3(0.5)));
}

float sat(vec3 c_1) {
    return (max(c_1.x, max(c_1.y, c_1.z)) - min(c_1.x, min(c_1.y, c_1.z)));
}

void set_sat_inner(inout float cmin, inout float cmid, inout float cmax, float s) {
    float _e4 = cmax;
    float _e5 = cmin;
    if ((_e4 > _e5)) {
        float _e7 = cmid;
        float _e8 = cmin;
        float _e11 = cmax;
        float _e12 = cmin;
        cmid = (((_e7 - _e8) * s) / (_e11 - _e12));
        cmax = s;
    } else {
        cmid = 0.0;
        cmax = 0.0;
    }
    cmin = 0.0;
    return;
}

vec3 set_sat(vec3 c_2, float s_1) {
    float r = 0.0;
    float g = 0.0;
    float b_1 = 0.0;
    r = c_2.x;
    g = c_2.y;
    b_1 = c_2.z;
    float _e8 = r;
    float _e9 = g;
    if ((_e8 <= _e9)) {
        float _e11 = g;
        float _e12 = b_1;
        if ((_e11 <= _e12)) {
            set_sat_inner(r, g, b_1, s_1);
        } else {
            float _e14 = r;
            float _e15 = b_1;
            if ((_e14 <= _e15)) {
                set_sat_inner(r, b_1, g, s_1);
            } else {
                set_sat_inner(b_1, r, g, s_1);
            }
        }
    } else {
        float _e17 = r;
        float _e18 = b_1;
        if ((_e17 <= _e18)) {
            set_sat_inner(g, r, b_1, s_1);
        } else {
            float _e20 = g;
            float _e21 = b_1;
            if ((_e20 <= _e21)) {
                set_sat_inner(g, b_1, r, s_1);
            } else {
                set_sat_inner(b_1, g, r, s_1);
            }
        }
    }
    float _e23 = r;
    float _e24 = g;
    float _e25 = b_1;
    return vec3(_e23, _e24, _e25);
}

float lum(vec3 c_3) {
    vec3 f = vec3(0.3, 0.59, 0.11);
    return dot(c_3, f);
}

vec3 clip_color(vec3 c_in) {
    vec3 c_4 = vec3(0.0);
    c_4 = c_in;
    vec3 _e2 = c_4;
    float _e3 = lum(_e2);
    float _e5 = c_4.x;
    float _e7 = c_4.y;
    float _e9 = c_4.z;
    float n = min(_e5, min(_e7, _e9));
    float _e13 = c_4.x;
    float _e15 = c_4.y;
    float _e17 = c_4.z;
    float x_1 = max(_e13, max(_e15, _e17));
    if ((n < 0.0)) {
        vec3 _e22 = c_4;
        c_4 = (vec3(_e3) + (((_e22 - vec3(_e3)) * _e3) / vec3((_e3 - n))));
    }
    if ((x_1 > 1.0)) {
        vec3 _e33 = c_4;
        c_4 = (vec3(_e3) + (((_e33 - vec3(_e3)) * (1.0 - _e3)) / vec3((x_1 - _e3))));
    }
    vec3 _e44 = c_4;
    return _e44;
}

vec3 set_lum(vec3 c_5, float l) {
    float _e2 = lum(c_5);
    vec3 _e6 = clip_color((c_5 + vec3((l - _e2))));
    return _e6;
}

vec3 blend_mix(vec3 cb_5, vec3 cs_5, uint mode_2) {
    vec3 b_2 = vec3(0.0);
    switch(mode_2) {
        case 1u: {
            b_2 = (cb_5 * cs_5);
            break;
        }
        case 2u: {
            vec3 _e7 = screen(cb_5, cs_5);
            b_2 = _e7;
            break;
        }
        case 3u: {
            vec3 _e8 = hard_light(cs_5, cb_5);
            b_2 = _e8;
            break;
        }
        case 4u: {
            b_2 = min(cb_5, cs_5);
            break;
        }
        case 5u: {
            b_2 = max(cb_5, cs_5);
            break;
        }
        case 6u: {
            float _e13 = color_dodge(cb_5.x, cs_5.x);
            float _e16 = color_dodge(cb_5.y, cs_5.y);
            float _e19 = color_dodge(cb_5.z, cs_5.z);
            b_2 = vec3(_e13, _e16, _e19);
            break;
        }
        case 7u: {
            float _e23 = color_burn(cb_5.x, cs_5.x);
            float _e26 = color_burn(cb_5.y, cs_5.y);
            float _e29 = color_burn(cb_5.z, cs_5.z);
            b_2 = vec3(_e23, _e26, _e29);
            break;
        }
        case 8u: {
            vec3 _e31 = hard_light(cb_5, cs_5);
            b_2 = _e31;
            break;
        }
        case 9u: {
            vec3 _e32 = soft_light(cb_5, cs_5);
            b_2 = _e32;
            break;
        }
        case 10u: {
            b_2 = abs((cb_5 - cs_5));
            break;
        }
        case 11u: {
            b_2 = ((cb_5 + cs_5) - ((2.0 * cb_5) * cs_5));
            break;
        }
        case 12u: {
            float _e40 = sat(cb_5);
            vec3 _e41 = set_sat(cs_5, _e40);
            float _e42 = lum(cb_5);
            vec3 _e43 = set_lum(_e41, _e42);
            b_2 = _e43;
            break;
        }
        case 13u: {
            float _e44 = sat(cs_5);
            vec3 _e45 = set_sat(cb_5, _e44);
            float _e46 = lum(cb_5);
            vec3 _e47 = set_lum(_e45, _e46);
            b_2 = _e47;
            break;
        }
        case 14u: {
            float _e48 = lum(cb_5);
            vec3 _e49 = set_lum(cs_5, _e48);
            b_2 = _e49;
            break;
        }
        case 15u: {
            float _e50 = lum(cs_5);
            vec3 _e51 = set_lum(cb_5, _e50);
            b_2 = _e51;
            break;
        }
        default: {
            b_2 = cs_5;
            break;
        }
    }
    vec3 _e52 = b_2;
    return _e52;
}

vec4 blend_compose_unpremul(vec3 cb_6, vec3 cs_6, float ab, float as, uint mode_3) {
    float fa = 0.0;
    float fb = 0.0;
    switch(mode_3) {
        case 1u: {
            fa = 1.0;
            fb = 0.0;
            break;
        }
        case 2u: {
            fa = 0.0;
            fb = 1.0;
            break;
        }
        case 3u: {
            fa = 1.0;
            fb = (1.0 - as);
            break;
        }
        case 4u: {
            fa = (1.0 - ab);
            fb = 1.0;
            break;
        }
        case 5u: {
            fa = ab;
            fb = 0.0;
            break;
        }
        case 6u: {
            fa = 0.0;
            fb = as;
            break;
        }
        case 7u: {
            fa = (1.0 - ab);
            fb = 0.0;
            break;
        }
        case 8u: {
            fa = 0.0;
            fb = (1.0 - as);
            break;
        }
        case 9u: {
            fa = ab;
            fb = (1.0 - as);
            break;
        }
        case 10u: {
            fa = (1.0 - ab);
            fb = as;
            break;
        }
        case 11u: {
            fa = (1.0 - ab);
            fb = (1.0 - as);
            break;
        }
        case 12u: {
            fa = 1.0;
            fb = 1.0;
            break;
        }
        case 13u: {
            return min(vec4(1.0), vec4(((as * cs_6) + (ab * cb_6)), (as + ab)));
        }
        default: {
            break;
        }
    }
    float _e45 = fa;
    float as_fa = (as * _e45);
    float _e47 = fb;
    float ab_fb = (ab * _e47);
    vec3 co = ((as_fa * cs_6) + (ab_fb * cb_6));
    return vec4(co, min((as_fa + ab_fb), 1.0));
}

vec4 blend_mix_compose(vec4 backdrop, vec4 src, uint compose_mode, uint mix_mode) {
    vec3 cs_7 = vec3(0.0);
    uint mode_4 = ((mix_mode << 8u) | compose_mode);
    if ((mode_4 == 3u)) {
        return ((backdrop * (1.0 - src.w)) + src);
    }
    float inv_src_a = (1.0 / max(src.w, 1e-15));
    cs_7 = (src.xyz * inv_src_a);
    float inv_backdrop_a = (1.0 / max(backdrop.w, 1e-15));
    vec3 cb_7 = (backdrop.xyz * inv_backdrop_a);
    vec3 _e28 = cs_7;
    vec3 _e29 = blend_mix(cb_7, _e28, mix_mode);
    vec3 _e30 = cs_7;
    cs_7 = mix(_e30, _e29, backdrop.w);
    if ((compose_mode == COMPOSE_SRC_OVER)) {
        vec3 _e36 = cs_7;
        vec3 co_1 = mix(backdrop.xyz, _e36, src.w);
        return vec4(co_1, (src.w + (backdrop.w * (1.0 - src.w))));
    } else {
        vec3 _e47 = cs_7;
        vec4 _e50 = blend_compose_unpremul(cb_7, _e47, backdrop.w, src.w, compose_mode);
        return _e50;
    }
}

void main() {
    uint paint_and_rect_flag = _vs2fs_location0;
    vec2 tex_coord = _vs2fs_location1;
    vec2 sample_xy = _vs2fs_location2;
    uint dense_end_or_rect_size = _vs2fs_location3;
    uint payload = _vs2fs_location4;
    uint rect_frac = _vs2fs_location5;
    vec4 position = gl_FragCoord;
    float alpha = 1.0;
    bool local_2 = false;
    bool local_3 = false;
    vec4 final_color = vec4(0.0);
    vec4 sample_color = vec4(0.0);
    vec2 grad_pos = vec2(0.0);
    bool is_rect = ((paint_and_rect_flag & RECT_STRIP_FLAG) != 0u);
    if (is_rect) {
        local_2 = (rect_frac != 0u);
    } else {
        local_2 = false;
    }
    bool _e14 = local_2;
    if (_e14) {
        uint _e15 = rect_frac;
        vec4 frac = (vec4(_e15 & 0xFFu, _e15 >> 8 & 0xFFu, _e15 >> 16 & 0xFFu, _e15 >> 24) / 255.0);
        vec2 rect_size = vec2(float((dense_end_or_rect_size & 65535u)), float((dense_end_or_rect_size >> 16u)));
        vec2 tc = tex_coord;
        vec2 bottom_and_right = min((tc + vec2(0.5)), (rect_size - frac.zw));
        vec2 top_and_left = max((tc - vec2(0.5)), frac.xy);
        vec2 a_3 = clamp((bottom_and_right - top_and_left), vec2(0.0), vec2(1.0));
        alpha = (a_3.x * a_3.y);
    } else {
        if (!(is_rect)) {
            local_3 = (dense_end_or_rect_size != 0u);
        } else {
            local_3 = false;
        }
        bool _e54 = local_3;
        if (_e54) {
            uint alphas_index = uint(floor(tex_coord.x));
            uint y_1 = uint(floor(tex_coord.y));
            uvec2 tex_dimensions = uvec2(textureSize(_group_0_binding_0_fs, 0).xy);
            uint alphas_tex_width = tex_dimensions.x;
            uint texel_index = (alphas_index / 4u);
            uint channel_index_1 = (alphas_index % 4u);
            uint tex_x_1 = (texel_index & (alphas_tex_width - 1u));
            uint _e75 = _group_0_binding_1_fs.alphas_tex_width_bits;
            uint tex_y_1 = (texel_index >> _e75);
            uvec4 rgba_values = texelFetch(_group_0_binding_0_fs, ivec2(uvec2(tex_x_1, tex_y_1)), 0);
            uint _e81 = unpack_alphas_from_channel(rgba_values, channel_index_1);
            alpha = (float(((_e81 >> (y_1 * 8u)) & 255u)) * 0.003921569);
        }
    }
    uint color_source = ((paint_and_rect_flag >> 29u) & 3u);
    if ((color_source == COLOR_SOURCE_PAYLOAD)) {
        uint paint_type = ((paint_and_rect_flag >> 26u) & 7u);
        if ((paint_type == PAINT_TYPE_SOLID)) {
            float _e105 = alpha;
            uint _e106 = payload;
            final_color = (_e105 * (vec4(_e106 & 0xFFu, _e106 >> 8 & 0xFFu, _e106 >> 16 & 0xFFu, _e106 >> 24) / 255.0));
        } else {
            if ((paint_type == PAINT_TYPE_IMAGE)) {
                uint paint_tex_idx_4 = (paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK);
                EncodedImage _e114 = unpack_encoded_image(paint_tex_idx_4);
                vec2 image_offset_3 = _e114.image_offset;
                vec2 image_size_3 = _e114.image_size;
                vec2 local_xy = (sample_xy - image_offset_3);
                float _e125 = extend_mode((local_xy.x + 1e-5), _e114.extend_modes.x, image_size_3.x);
                float _e131 = extend_mode((local_xy.y + 1e-5), _e114.extend_modes.y, image_size_3.y);
                vec2 extended_xy = vec2(_e125, _e131);
                if ((_e114.quality == IMAGE_QUALITY_HIGH)) {
                    vec2 final_xy = (image_offset_3 + extended_xy);
                    vec4 _e143 = bicubic_sample(_group_1_binding_0_fs, final_xy, int(_e114.atlas_index), image_offset_3, image_size_3, _e114.extend_modes, _e114.image_padding);
                    sample_color = _e143;
                } else {
                    if ((_e114.quality == IMAGE_QUALITY_MEDIUM)) {
                        vec2 final_xy_1 = ((image_offset_3 + extended_xy) - vec2(0.5));
                        vec4 _e156 = bilinear_sample(_group_1_binding_0_fs, final_xy_1, int(_e114.atlas_index), image_offset_3, image_size_3, _e114.extend_modes, _e114.image_padding);
                        sample_color = _e156;
                    } else {
                        vec2 final_xy_2 = (image_offset_3 + extended_xy);
                        vec4 _e163 = texelFetch(_group_1_binding_0_fs, ivec3(uvec2(final_xy_2), int(_e114.atlas_index)), 0);
                        sample_color = _e163;
                    }
                }
                bool is_multiply = bool(_e114.tint_mode);
                float _e166 = alpha;
                float _e169 = sample_color.w;
                vec4 _e171 = sample_color;
                final_color = (_e166 * (is_multiply ? (_e171 * _e114.tint) : (_e114.tint * _e169)));
            } else {
                if ((paint_type == PAINT_TYPE_LINEAR_GRADIENT)) {
                    uint paint_tex_idx_5 = (paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK);
                    LinearGradient _e181 = unpack_linear_gradient(paint_tex_idx_5);
                    vec2 fragment_pos = sample_xy;
                    vec2 grad_pos_2 = ((_e181.transform * fragment_pos) + _e181.translate);
                    float t_value_2 = (grad_pos_2.x + 1e-5);
                    vec4 _e193 = sample_gradient_lut(t_value_2, _e181.extend_mode, _e181.gradient_start, _e181.texture_width);
                    float _e194 = alpha;
                    final_color = (_e194 * _e193);
                } else {
                    if ((paint_type == PAINT_TYPE_RADIAL_GRADIENT)) {
                        uint paint_tex_idx_6 = (paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK);
                        RadialGradient _e201 = unpack_radial_gradient(paint_tex_idx_6);
                        vec2 fragment_pos_1 = sample_xy;
                        vec2 grad_pos_3 = ((_e201.transform * fragment_pos_1) + _e201.translate);
                        RadialGradientResult _e207 = calculate_radial_gradient(grad_pos_3, _e201);
                        vec4 _e212 = sample_gradient_lut(_e207.t_value, _e201.extend_mode, _e201.gradient_start, _e201.texture_width);
                        float _e218 = alpha;
                        final_color = (_e207.is_valid ? (_e218 * _e212) : vec4(0.0, 0.0, 0.0, 0.0));
                    } else {
                        if ((paint_type == PAINT_TYPE_SWEEP_GRADIENT)) {
                            uint paint_tex_idx_7 = (paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK);
                            SweepGradient _e227 = unpack_sweep_gradient(paint_tex_idx_7);
                            vec2 fragment_pos_2 = sample_xy;
                            grad_pos = ((_e227.transform * fragment_pos_2) + _e227.translate);
                            vec2 _e234 = grad_pos;
                            vec2 _e235 = grad_pos;
                            grad_pos = mix(_e234, vec2(0.0), lessThan(abs(_e235), vec2(0.00024414063)));
                            float _e244 = grad_pos.x;
                            float _e246 = grad_pos.y;
                            float _e247 = xy_to_unit_angle(_e244, _e246);
                            float angle = (_e247 * TWO_PI);
                            float t_value_3 = ((angle - _e227.start_angle) * _e227.inv_angle_delta);
                            vec4 _e257 = sample_gradient_lut(t_value_3, _e227.extend_mode, _e227.gradient_start, _e227.texture_width);
                            float _e258 = alpha;
                            final_color = (_e258 * _e257);
                        }
                    }
                }
            }
        }
    } else {
        if ((color_source == COLOR_SOURCE_SLOT)) {
            uint _e266 = _group_0_binding_1_fs.height;
            uint _e273 = _group_0_binding_1_fs.ndc_y_negate;
            float sample_y = ((_e273 != 0u) ? (float(_e266) - position.y) : position.y);
            int _e282 = _group_0_binding_1_fs.strip_offset_x;
            uint clip_x = (uint((int(position.x) - _e282)) & 255u);
            int _e290 = _group_0_binding_1_fs.strip_offset_y;
            uint _e298 = _group_0_binding_1_fs.strip_height;
            uint clip_y = ((uint((int(sample_y) - _e290)) & 3u) + (payload * _e298));
            vec4 clip_in_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x, clip_y)), 0);
            float opacity = (float((paint_and_rect_flag & 255u)) * 0.003921569);
            float _e311 = alpha;
            final_color = ((_e311 * opacity) * clip_in_color);
        } else {
            if ((color_source == COLOR_SOURCE_BLEND)) {
                uint _e320 = _group_0_binding_1_fs.height;
                uint _e327 = _group_0_binding_1_fs.ndc_y_negate;
                float sample_y_1 = ((_e327 != 0u) ? (float(_e320) - position.y) : position.y);
                float opacity_1 = (float(((paint_and_rect_flag >> 16u) & 255u)) * 0.003921569);
                uint mix_mode_1 = ((paint_and_rect_flag >> 8u) & 255u);
                uint compose_mode_1 = (paint_and_rect_flag & 255u);
                uint src_slot = (payload & 65535u);
                uint dest_slot = ((payload >> 16u) & 65535u);
                int _e360 = _group_0_binding_1_fs.strip_offset_x;
                uint clip_x_1 = (uint((int(position.x) - _e360)) & 255u);
                int _e368 = _group_0_binding_1_fs.strip_offset_y;
                uint clip_y_in_strip = (uint((int(sample_y_1) - _e368)) & 3u);
                uint _e375 = _group_0_binding_1_fs.strip_height;
                uint src_y = (clip_y_in_strip + (src_slot * _e375));
                vec4 src_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x_1, src_y)), 0);
                uint _e384 = _group_0_binding_1_fs.strip_height;
                uint dest_y = (clip_y_in_strip + (dest_slot * _e384));
                vec4 dest_color = texelFetch(_group_0_binding_2_fs, ivec2(uvec2(clip_x_1, dest_y)), 0);
                float _e392 = alpha;
                vec4 _e394 = blend_mix_compose(dest_color, ((src_color * opacity_1) * _e392), compose_mode_1, mix_mode_1);
                final_color = _e394;
            }
        }
    }
    vec4 _e395 = final_color;
    _fs2p_location0 = _e395;
    return;
}

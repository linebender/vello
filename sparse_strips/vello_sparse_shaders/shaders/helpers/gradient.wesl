// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared encoded gradient accessors and gradient evaluation helpers.

import package::helpers::extend::extend_mode_normalized;
import package::helpers::math::NEARLY_ZERO_TOLERANCE;

// Gradient types.
const GRADIENT_TYPE_LINEAR: u32 = 0u;
const GRADIENT_TYPE_RADIAL: u32 = 1u;
const GRADIENT_TYPE_SWEEP: u32 = 2u;

// Radial gradient types.
const RADIAL_GRADIENT_TYPE_STANDARD: u32 = 0u;
const RADIAL_GRADIENT_TYPE_STRIP: u32 = 1u;
const RADIAL_GRADIENT_TYPE_FOCAL: u32 = 2u;

// Sample from the gradient texture at calculated position.
fn sample_gradient_lut(
    gradient_texture: texture_2d<f32>,
    t_value: f32,
    extend_mode: u32,
    gradient_start: u32,
    texture_width: u32,
) -> vec4<f32> {
    // Apply extend mode to t_value
    let clamped_t = extend_mode_normalized(t_value, extend_mode);
    // Convert t_value to texture coordinate
    let t_offset = u32(clamped_t * f32(texture_width - 1u));
    // Calculate absolute position in flat gradient texture
    let flat_coord = gradient_start + t_offset;
    // Convert flat coordinate to 2D texture coordinate
    let gradient_tex_width = textureDimensions(gradient_texture).x;
    let tex_x = flat_coord % gradient_tex_width;
    let tex_y = flat_coord / gradient_tex_width;
    // Sample from the gradient texture at calculated position
    let gradient_color = textureLoad(gradient_texture, vec2<u32>(tex_x, tex_y), 0);
    return gradient_color;
}

/// Width of the gradient texture.
fn get_gradient_texture_width(texel0: vec4<u32>) -> u32 { return texel0.x & 0x0FFFFFFFu; }

/// The extend mode for the gradient.
fn get_gradient_extend_mode(texel0: vec4<u32>) -> u32 { return (texel0.x >> 30u) & 3u; }

/// Start coordinate in the flat gradient texture.
fn get_gradient_start(texel0: vec4<u32>) -> u32 { return texel0.y; }

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_gradient_transform(texel0: vec4<u32>, texel1: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
        vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y))
    );
}

/// Translation part of the affine transform [tx, ty].
fn get_gradient_translate(texel1: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
}

fn apply_gradient_transform(
    texel0: vec4<u32>,
    texel1: vec4<u32>,
    fragment_pos: vec2<f32>,
) -> vec2<f32> {
    return get_gradient_transform(texel0, texel1) * fragment_pos + get_gradient_translate(texel1);
}

/// Kind of radial gradient (0=Radial, 1=Strip, 2=Focal).
fn get_radial_kind(texel2: vec4<u32>) -> u32 { return texel2.x & 0x3u; }

/// Whether the focal point is swapped for radial gradient (0=false, 1=true).
fn get_radial_f_is_swapped(texel2: vec4<u32>) -> u32 { return (texel2.x >> 2u) & 1u; }

/// Bias value for radial gradient calculation.
fn get_radial_bias(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

/// Scale factor for radial gradient calculation.
fn get_radial_scale(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.z); }

/// Focal point 0 parameter for radial gradient.
fn get_radial_fp0(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.w); }

/// Focal point 1 parameter for radial gradient.
fn get_radial_fp1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.x); }

/// Focal radius 1 parameter for radial gradient.
fn get_radial_fr1(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.y); }

/// Focal X coordinate for radial gradient.
fn get_radial_f_focal_x(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.z); }

/// Scaled radius 0 squared parameter for radial gradient strip.
fn get_radial_scaled_r0_squared(texel3: vec4<u32>) -> f32 { return bitcast<f32>(texel3.w); }

/// Starting angle for sweep gradient (in radians).
fn get_sweep_start_angle(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.x); }

/// Inverse of angle delta for sweep gradient.
fn get_sweep_inv_angle_delta(texel2: vec4<u32>) -> f32 { return bitcast<f32>(texel2.y); }

// Fast polynomial approximation for xy_to_unit_angle from Skia
// Returns angle in [0, 1) range representing [0, 2π)
// See: https://github.com/google/skia/blob/30bba741989865c157c7a997a0caebe94921276b/src/opts/SkRasterPipeline_opts.h#L5859
fn xy_to_unit_angle(x: f32, y: f32) -> f32 {
    let xabs = abs(x);
    let yabs = abs(y);
    let slope = min(xabs, yabs) / max(xabs, yabs);
    let s = slope * slope;
    // Use a 7th degree polynomial to approximate atan.
    // This was generated using sollya.gforge.inria.fr.
    // A float optimized polynomial was generated using the following command.
    // P1 = fpminimax((1/(2*Pi))*atan(x),[|1,3,5,7|],[|24...|],[2^(-40),1],relative);
    var phi = slope * (0.15912117063999176025390625 + s * (-5.185396969318389892578125e-2 + s * (2.476101927459239959716796875e-2 + s * (-7.0547382347285747528076171875e-3))));
    // Map from first octant to full circle using quadrant information
    // Handle [0°, 90°] range
    phi = select(phi, 0.25 - phi, xabs < yabs);
    // Handle [90°, 180°] range
    phi = select(phi, 0.5 - phi, x < 0.0);
    // Handle [180°, 360°] range
    phi = select(phi, 1.0 - phi, y < 0.0);
    // Handle NaN cases (using property that NaN != NaN)
    phi = select(phi, 0.0, phi != phi);
    return phi;
}

// Calculate a radial gradient; matches vello_cpu implementation.
fn calculate_radial_gradient(
    grad_pos: vec2<f32>,
    texel2: vec4<u32>,
    texel3: vec4<u32>,
) -> vec2<f32> {
    let x_pos = grad_pos.x;
    let y_pos = grad_pos.y;
    
    var t_value: f32;
    var is_valid: bool;
    let kind = get_radial_kind(texel2);
    
    switch kind {
        case RADIAL_GRADIENT_TYPE_STANDARD: {
            // Standard radial gradient: bias + scale * sqrt(x^2 + y^2)
            let radius = sqrt(x_pos * x_pos + y_pos * y_pos);
            t_value = get_radial_bias(texel2) + get_radial_scale(texel2) * radius;
            // Radial gradients are always valid
            is_valid = true;
        }
        case RADIAL_GRADIENT_TYPE_STRIP: {
            // Strip gradient: x + sqrt(scaled_r0_squared - y^2)
            let p1 = get_radial_scaled_r0_squared(texel3) - y_pos * y_pos;
            // Invalid if negative under square root
            is_valid = p1 >= 0.0;
            if is_valid {
                t_value = x_pos + sqrt(p1);
            } else {
                // Value doesn't matter when invalid
                t_value = 0.0;
            }
        }
        case RADIAL_GRADIENT_TYPE_FOCAL, default: {
            // Focal gradient implementation
            var t = 0.0;
            let fp0 = get_radial_fp0(texel2);
            let fp1 = get_radial_fp1(texel3);
            let fr1 = get_radial_fr1(texel3);
            let f_focal_x = get_radial_f_focal_x(texel3);
            let is_swapped = get_radial_f_is_swapped(texel2);
            
            // Calculate focal flags directly from field values (matching FocalData implementation)
            let is_focal_on_circle = abs(1.0 - fr1) <= NEARLY_ZERO_TOLERANCE;
            let is_well_behaved = !is_focal_on_circle && fr1 > 1.0;
            let is_natively_focal = abs(f_focal_x) <= NEARLY_ZERO_TOLERANCE;
            
            // Start with valid assumption
            is_valid = true;
            
            if is_focal_on_circle {
                t = x_pos + y_pos * y_pos / x_pos;
                // Check for division by zero and negative t
                is_valid = t >= 0.0 && x_pos != 0.0;
            } else if is_well_behaved {
                t = sqrt(x_pos * x_pos + y_pos * y_pos) - x_pos * fp0;
            } else {
                // For non-well-behaved gradients, check if calculation is valid
                let xx = x_pos * x_pos;
                let yy = y_pos * y_pos;
                let discriminant = xx - yy;
                
                if is_swapped != 0u || (1.0 - f_focal_x < 0.0) {
                    t = -sqrt(discriminant) - x_pos * fp0;
                } else {
                    t = sqrt(discriminant) - x_pos * fp0;
                }
                
                // Invalid if discriminant is negative or t is negative
                is_valid = discriminant >= 0.0 && t >= 0.0;
            }
            
            // Apply additional focal transforms only if still valid
            if is_valid {
                if 1.0 - f_focal_x < 0.0 {
                    t = -t;
                }
                
                if !is_natively_focal {
                    t = t + fp1;
                }
                
                if is_swapped != 0u {
                    t = 1.0 - t;
                }
            }
            
            t_value = t;
        }
    }
    
    return vec2<f32>(t_value, select(0.0, 1.0, is_valid));
}

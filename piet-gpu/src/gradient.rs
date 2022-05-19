// Copyright 2021 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Implementation of gradients.

use std::collections::hash_map::{Entry, HashMap};

use piet::kurbo::Point;
use piet::{Color, FixedLinearGradient, FixedRadialGradient, GradientStop};

/// Radial gradient compatible with COLRv1 spec
#[derive(Debug, Clone)]
pub struct Colrv1RadialGradient {
    /// The center of the iner circle.
    pub center0: Point,
    /// The offset of the origin relative to the center.
    pub center1: Point,
    /// The radius of the inner circle.
    pub radius0: f64,
    /// The radius of the outer circle.
    pub radius1: f64,
    /// The stops.
    pub stops: Vec<GradientStop>,
}

#[derive(Clone)]
pub struct BakedGradient {
    ramp: Vec<u32>,
}

#[derive(Clone)]
pub struct LinearGradient {
    pub(crate) start: [f32; 2],
    pub(crate) end: [f32; 2],
    pub(crate) ramp_id: u32,
}

#[derive(Clone)]
pub struct RadialGradient {
    pub(crate) start: [f32; 2],
    pub(crate) end: [f32; 2],
    pub(crate) r0: f32,
    pub(crate) r1: f32,
    pub(crate) ramp_id: u32,
}

#[derive(Default)]
pub struct RampCache {
    ramps: Vec<GradientRamp>,
    map: HashMap<GradientRamp, usize>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct GradientRamp(Vec<u32>);

pub const N_SAMPLES: usize = 512;
// TODO: make this dynamic
pub const N_GRADIENTS: usize = 256;

#[derive(Clone, Copy)]
struct PremulRgba([f64; 4]);

impl PremulRgba {
    fn from_color(c: &Color) -> PremulRgba {
        let rgba = c.as_rgba();
        let a = rgba.3;
        // TODO: sRGB nonlinearity? This is complicated.
        PremulRgba([rgba.0 * a, rgba.1 * a, rgba.2 * a, a])
    }

    fn to_u32(&self) -> u32 {
        let z = self.0;
        let r = (z[0].max(0.0).min(1.0) * 255.0).round() as u32;
        let g = (z[1].max(0.0).min(1.0) * 255.0).round() as u32;
        let b = (z[2].max(0.0).min(1.0) * 255.0).round() as u32;
        let a = (z[3].max(0.0).min(1.0) * 255.0).round() as u32;
        r | (g << 8) | (b << 16) | (a << 24)
    }

    fn lerp(&self, other: PremulRgba, t: f64) -> PremulRgba {
        fn l(a: f64, b: f64, t: f64) -> f64 {
            a * (1.0 - t) + b * t
        }
        let a = self.0;
        let b = other.0;
        PremulRgba([
            l(a[0], b[0], t),
            l(a[1], b[1], t),
            l(a[2], b[2], t),
            l(a[3], b[3], t),
        ])
    }
}

impl GradientRamp {
    fn from_stops(stops: &[GradientStop]) -> GradientRamp {
        let mut last_u = 0.0;
        let mut last_c = PremulRgba::from_color(&stops[0].color);
        let mut this_u = last_u;
        let mut this_c = last_c;
        let mut j = 0;
        let v = (0..N_SAMPLES)
            .map(|i| {
                let u = (i as f64) / (N_SAMPLES - 1) as f64;
                while u > this_u {
                    last_u = this_u;
                    last_c = this_c;
                    if let Some(s) = stops.get(j + 1) {
                        this_u = s.pos as f64;
                        this_c = PremulRgba::from_color(&s.color);
                        j += 1;
                    } else {
                        break;
                    }
                }
                let du = this_u - last_u;
                let c = if du < 1e-9 {
                    this_c
                } else {
                    last_c.lerp(this_c, (u - last_u) / du)
                };
                c.to_u32()
            })
            .collect();
        GradientRamp(v)
    }

    /// For debugging/development.
    pub(crate) fn dump(&self) {
        for val in &self.0 {
            println!("{:x}", val);
        }
    }
}

impl RampCache {
    /// Add a gradient ramp to the cache.
    ///
    /// Currently there is no eviction, so if the gradient is animating, there may
    /// be resource leaks. In order to support lifetime management, the signature
    /// should probably change so it returns a ref-counted handle, so that eviction
    /// is deferred until the last handle is dropped.
    ///
    /// This function is pretty expensive, but the result is lightweight.
    fn add_ramp(&mut self, ramp: &[GradientStop]) -> usize {
        let ramp = GradientRamp::from_stops(ramp);
        match self.map.entry(ramp) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let idx = self.ramps.len();
                self.ramps.push(v.key().clone());
                v.insert(idx);
                idx
            }
        }
    }

    pub fn add_linear_gradient(&mut self, lin: &FixedLinearGradient) -> LinearGradient {
        let ramp_id = self.add_ramp(&lin.stops);
        LinearGradient {
            ramp_id: ramp_id as u32,
            start: crate::render_ctx::to_f32_2(lin.start),
            end: crate::render_ctx::to_f32_2(lin.end),
        }
    }

    pub fn add_radial_gradient(&mut self, rad: &FixedRadialGradient) -> RadialGradient {
        let ramp_id = self.add_ramp(&rad.stops);
        RadialGradient {
            ramp_id: ramp_id as u32,
            start: crate::render_ctx::to_f32_2(rad.center + rad.origin_offset),
            end: crate::render_ctx::to_f32_2(rad.center),
            r0: 0.0,
            r1: rad.radius as f32,
        }
    }

    pub fn add_radial_gradient_colrv1(&mut self, rad: &Colrv1RadialGradient) -> RadialGradient {
        let ramp_id = self.add_ramp(&rad.stops);
        RadialGradient {
            ramp_id: ramp_id as u32,
            start: crate::render_ctx::to_f32_2(rad.center0),
            end: crate::render_ctx::to_f32_2(rad.center1),
            r0: rad.radius0 as f32,
            r1: rad.radius1 as f32,
        }
    }

    /// Dump the contents of a gradient. This is for debugging.
    #[allow(unused)]
    pub(crate) fn dump_gradient(&self, lin: &LinearGradient) {
        println!("id = {}", lin.ramp_id);
        self.ramps[lin.ramp_id as usize].dump();
    }

    /// Get the ramp data.
    ///
    /// This concatenates all the ramps; we'll want a more sophisticated approach to
    /// incremental update.
    pub fn get_ramp_data(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(N_SAMPLES * self.ramps.len());
        for ramp in &self.ramps {
            result.extend(&ramp.0);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::RampCache;
    use piet::kurbo::Point;
    use piet::{Color, FixedLinearGradient, GradientStop};

    #[test]
    fn simple_ramp() {
        let stops = vec![
            GradientStop {
                color: Color::WHITE,
                pos: 0.0,
            },
            GradientStop {
                color: Color::BLACK,
                pos: 1.0,
            },
        ];
        let mut cache = RampCache::default();
        let lin = FixedLinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(0.0, 1.0),
            stops,
        };
        let our_lin = cache.add_linear_gradient(&lin);
        cache.dump_gradient(&our_lin);
    }
}

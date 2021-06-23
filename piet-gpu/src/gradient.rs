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

use piet::{Color, FixedLinearGradient, GradientStop};

#[derive(Clone)]
pub struct BakedGradient {
    ramp: Vec<u32>,
}

#[derive(Clone)]
pub struct LinearGradient {
    start: [f32; 2],
    end: [f32; 2],
    ramp_id: u32,
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
        Color::rgba(z[0], z[1], z[2], z[3]).as_rgba_u32()
    }

    fn lerp(&self, other: PremulRgba, t: f64) -> PremulRgba {
        fn l(a: f64, b: f64, t: f64) -> f64 {
            a * (1.0 - t) + b * t
        }
        let a = self.0;
        let b = other.0;
        PremulRgba([l(a[0], b[0], t), l(a[1], b[1], t), l(a[2], b[2], t), l(a[2], b[3], t)])
    }
}

impl GradientRamp {
    fn from_stops(stops: &[GradientStop]) -> GradientRamp {
        let mut last_u = 0.0;
        let mut last_c = PremulRgba::from_color(&stops[0].color);
        let mut this_u = last_u;
        let mut this_c = last_c;
        let mut j = 0;
        let v = (0..N_SAMPLES).map(|i| {
            let u = (i as f64) / 255.0;
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
        }).collect();
        GradientRamp(v)
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
}


#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        println!("it works!");
    }
}

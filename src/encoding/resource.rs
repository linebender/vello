// Copyright 2022 Google LLC
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

//! Late bound resource management.

use std::collections::HashMap;
use std::ops::Range;

use peniko::{Color, ColorStop, ColorStops};

const N_SAMPLES: usize = 512;
const RETAINED_COUNT: usize = 64;

/// Token for ensuring that an encoded scene matches the current state
/// of a resource cache.
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Token(u64);

/// Cache for late bound resources.
#[derive(Default)]
pub struct ResourceCache {
    ramps: RampCache,
}

impl ResourceCache {
    /// Creates a new resource cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the ramp data, width and height. Returns `None` if the
    /// given token does not match the current state of the cache.
    pub fn ramps(&self, token: Token) -> Option<(&[u32], u32, u32)> {
        if token.0 == self.ramps.epoch {
            Some((self.ramps.data(), self.ramps.width(), self.ramps.height()))
        } else {
            None
        }
    }

    pub(crate) fn advance(&mut self) -> Token {
        self.ramps.advance();
        Token(self.ramps.epoch)
    }

    pub(crate) fn add_ramp(&mut self, stops: &[ColorStop]) -> u32 {
        self.ramps.add(stops)
    }
}

#[derive(Clone)]
/// Patch for a late bound resource.
pub enum Patch {
    /// Gradient ramp resource.
    Ramp {
        /// Byte offset to the ramp id in the draw data stream.
        offset: usize,
        /// Range of the gradient stops in the resource set.
        stops: Range<usize>,
    },
}

#[derive(Default)]
struct RampCache {
    epoch: u64,
    map: HashMap<ColorStops, (u32, u64)>,
    data: Vec<u32>,
}

impl RampCache {
    pub fn advance(&mut self) {
        self.epoch += 1;
        if self.map.len() > RETAINED_COUNT {
            self.map
                .retain(|_key, value| value.0 < RETAINED_COUNT as u32);
            self.data.truncate(RETAINED_COUNT * N_SAMPLES);
        }
    }

    pub fn add(&mut self, stops: &[ColorStop]) -> u32 {
        if let Some(entry) = self.map.get_mut(stops) {
            entry.1 = self.epoch;
            entry.0
        } else if self.map.len() < RETAINED_COUNT {
            let id = (self.data.len() / N_SAMPLES) as u32;
            self.data.extend(make_ramp(stops));
            self.map.insert(stops.into(), (id, self.epoch));
            id
        } else {
            let mut reuse = None;
            for (stops, (id, epoch)) in &self.map {
                if *epoch + 2 < self.epoch {
                    reuse = Some((stops.to_owned(), *id));
                    break;
                }
            }
            if let Some((old_stops, id)) = reuse {
                self.map.remove(&old_stops);
                let start = id as usize * N_SAMPLES;
                for (dst, src) in self.data[start..start + N_SAMPLES]
                    .iter_mut()
                    .zip(make_ramp(stops))
                {
                    *dst = src;
                }
                self.map.insert(stops.into(), (id, self.epoch));
                id
            } else {
                let id = (self.data.len() / N_SAMPLES) as u32;
                self.data.extend(make_ramp(stops));
                self.map.insert(stops.into(), (id, self.epoch));
                id
            }
        }
    }

    pub fn data(&self) -> &[u32] {
        &self.data
    }

    pub fn width(&self) -> u32 {
        N_SAMPLES as u32
    }

    pub fn height(&self) -> u32 {
        (self.data.len() / N_SAMPLES) as u32
    }
}

fn make_ramp(stops: &[ColorStop]) -> impl Iterator<Item = u32> + '_ {
    let mut last_u = 0.0;
    let mut last_c = ColorF64::from_color(stops[0].color);
    let mut this_u = last_u;
    let mut this_c = last_c;
    let mut j = 0;
    (0..N_SAMPLES).map(move |i| {
        let u = (i as f64) / (N_SAMPLES - 1) as f64;
        while u > this_u {
            last_u = this_u;
            last_c = this_c;
            if let Some(s) = stops.get(j + 1) {
                this_u = s.offset as f64;
                this_c = ColorF64::from_color(s.color);
                j += 1;
            } else {
                break;
            }
        }
        let du = this_u - last_u;
        let c = if du < 1e-9 {
            this_c
        } else {
            last_c.lerp(&this_c, (u - last_u) / du)
        };
        c.as_premul_u32()
    })
}

#[derive(Copy, Clone, Debug)]
struct ColorF64([f64; 4]);

impl ColorF64 {
    fn from_color(color: Color) -> Self {
        Self([
            color.r as f64 / 255.0,
            color.g as f64 / 255.0,
            color.b as f64 / 255.0,
            color.a as f64 / 255.0,
        ])
    }

    fn lerp(&self, other: &Self, a: f64) -> Self {
        fn l(x: f64, y: f64, a: f64) -> f64 {
            x * (1.0 - a) + y * a
        }
        Self([
            l(self.0[0], other.0[0], a),
            l(self.0[1], other.0[1], a),
            l(self.0[2], other.0[2], a),
            l(self.0[3], other.0[3], a),
        ])
    }

    fn as_premul_u32(&self) -> u32 {
        let a = self.0[3].clamp(0.0, 1.0);
        let r = ((self.0[0] * a).clamp(0.0, 1.0) * 255.0) as u32;
        let g = ((self.0[1] * a).clamp(0.0, 1.0) * 255.0) as u32;
        let b = ((self.0[2] * a).clamp(0.0, 1.0) * 255.0) as u32;
        let a = (a * 255.0) as u32;
        r | (g << 8) | (b << 16) | (a << 24)
    }
}

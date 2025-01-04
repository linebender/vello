// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;

use peniko::color::cache_key::CacheKey;
use peniko::color::{HueDirection, Srgb};
use peniko::{ColorStop, ColorStops};

const N_SAMPLES: usize = 512;
const RETAINED_COUNT: usize = 64;

/// Data and dimensions for a set of resolved gradient ramps.
#[derive(Copy, Clone, Debug, Default)]
pub struct Ramps<'a> {
    pub data: &'a [u32],
    pub width: u32,
    pub height: u32,
}

#[derive(Default)]
pub(crate) struct RampCache {
    epoch: u64,
    map: HashMap<CacheKey<ColorStops>, (u32, u64)>,
    data: Vec<u32>,
}

impl RampCache {
    pub(crate) fn maintain(&mut self) {
        self.epoch += 1;
        if self.map.len() > RETAINED_COUNT {
            self.map
                .retain(|_key, value| value.0 < RETAINED_COUNT as u32);
            self.data.truncate(RETAINED_COUNT * N_SAMPLES);
        }
    }

    pub(crate) fn add(&mut self, stops: &[ColorStop]) -> u32 {
        if let Some(entry) = self.map.get_mut(&CacheKey(stops.into())) {
            entry.1 = self.epoch;
            entry.0
        } else if self.map.len() < RETAINED_COUNT {
            let id = (self.data.len() / N_SAMPLES) as u32;
            self.data.extend(make_ramp(stops));
            self.map.insert(CacheKey(stops.into()), (id, self.epoch));
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
                self.map.insert(CacheKey(stops.into()), (id, self.epoch));
                id
            } else {
                let id = (self.data.len() / N_SAMPLES) as u32;
                self.data.extend(make_ramp(stops));
                self.map.insert(CacheKey(stops.into()), (id, self.epoch));
                id
            }
        }
    }

    pub(crate) fn ramps(&self) -> Ramps<'_> {
        Ramps {
            data: &self.data,
            width: N_SAMPLES as u32,
            height: (self.data.len() / N_SAMPLES) as u32,
        }
    }
}

fn make_ramp(stops: &[ColorStop]) -> impl Iterator<Item = u32> + '_ {
    let mut last_u = 0.0;
    let mut last_c = stops[0].color.to_alpha_color::<Srgb>();
    let mut this_u = last_u;
    let mut this_c = last_c;
    let mut j = 0;
    (0..N_SAMPLES).map(move |i| {
        let u = (i as f32) / (N_SAMPLES - 1) as f32;
        while u > this_u {
            last_u = this_u;
            last_c = this_c;
            if let Some(s) = stops.get(j + 1) {
                this_u = s.offset;
                this_c = s.color.to_alpha_color::<Srgb>();
                j += 1;
            } else {
                break;
            }
        }
        let du = this_u - last_u;
        let c = if du < 1e-9 {
            this_c
        } else {
            last_c.lerp(this_c, (u - last_u) / du, HueDirection::default())
        };
        c.premultiply().to_rgba8().to_u32()
    })
}

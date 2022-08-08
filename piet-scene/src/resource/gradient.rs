use crate::brush::{Color, GradientStop, GradientStops};
use std::collections::HashMap;

const N_SAMPLES: usize = 512;
const RETAINED_COUNT: usize = 64;

#[derive(Default)]
pub struct RampCache {
    epoch: u64,
    map: HashMap<GradientStops, (u32, u64)>,
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

    pub fn clear(&mut self) {
        self.epoch = 0;
        self.map.clear();
        self.data.clear();
    }

    pub fn add(&mut self, stops: &[GradientStop]) -> u32 {
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
}

fn make_ramp<'a>(stops: &'a [GradientStop]) -> impl Iterator<Item = u32> + 'a {
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
        c.to_premul_u32()
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

    fn to_premul_u32(&self) -> u32 {
        let a = self.0[3].min(1.0).max(0.0);
        let r = ((self.0[0] * a).min(1.0).max(0.0) * 255.0) as u32;
        let g = ((self.0[1] * a).min(1.0).max(0.0) * 255.0) as u32;
        let b = ((self.0[2] * a).min(1.0).max(0.0) * 255.0) as u32;
        let a = (a * 255.0) as u32;
        b | (g << 8) | (r << 16) | (a << 24)
    }
}

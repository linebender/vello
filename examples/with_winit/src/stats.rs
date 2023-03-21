// Copyright 2023 Google LLC
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

use scenes::SimpleText;
use std::collections::VecDeque;
use vello::{
    kurbo::{Affine, PathEl, Rect},
    peniko::{Brush, Color, Fill, Stroke},
    SceneBuilder,
};

const SLIDING_WINDOW_SIZE: usize = 100;

#[derive(Debug)]
pub struct Snapshot {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub frame_time_min_ms: f64,
    pub frame_time_max_ms: f64,
}

impl Snapshot {
    pub fn draw_layer<'a, T>(
        &self,
        sb: &mut SceneBuilder,
        text: &mut SimpleText,
        viewport_width: f64,
        viewport_height: f64,
        samples: T,
        vsync: bool,
    ) where
        T: Iterator<Item = &'a u64>,
    {
        let width = (viewport_width * 0.4).max(200.).min(600.);
        let height = width * 0.7;
        let x_offset = viewport_width - width;
        let y_offset = viewport_height - height;
        let offset = Affine::translate((x_offset, y_offset));

        // Draw the background
        sb.fill(
            Fill::NonZero,
            offset,
            &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
            None,
            &Rect::new(0., 0., width, height),
        );

        let labels = [
            format!("Frame Time: {:.2} ms", self.frame_time_ms),
            format!("Frame Time (min): {:.2} ms", self.frame_time_min_ms),
            format!("Frame Time (max): {:.2} ms", self.frame_time_max_ms),
            format!("VSync: {}", if vsync { "on" } else { "off" }),
            format!("Resolution: {viewport_width}x{viewport_height}"),
        ];

        // height / 2 is dedicated to the text labels and the rest is filled by the bar graph.
        let text_height = height * 0.5 / (1 + labels.len()) as f64;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                sb,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                &label,
            );
        }
        text.add(
            sb,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((width * 0.67, text_height)),
            &format!("FPS: {:.2}", self.fps),
        );

        // Plot the samples with a bar graph
        use PathEl::*;
        let left_padding = width * 0.05; // Left padding for the frame time marker text.
        let graph_max_height = height * 0.5;
        let graph_max_width = width - 2. * left_margin - left_padding;
        let left_margin_padding = left_margin + left_padding;
        let bar_extent = graph_max_width / (SLIDING_WINDOW_SIZE as f64);
        let bar_width = bar_extent * 0.4;
        let bar = [
            MoveTo((0., graph_max_height).into()),
            LineTo((0., 0.).into()),
            LineTo((bar_width, 0.).into()),
            LineTo((bar_width, graph_max_height).into()),
        ];
        // We determine the scale of the graph based on the maximum sampled frame time unless it's
        // greater than 3x the current average. In that case we cap the max scale at 4/3 * the
        // current average (rounded up to the nearest multiple of 5ms). This allows the scale to
        // adapt to the most recent sample set as relying on the maximum alone can make the
        // displayed samples to look too small in the presence of spikes/fluctuation without
        // manually resetting the max sample.
        let display_max = if self.frame_time_max_ms > 3. * self.frame_time_ms {
            round_up((1.33334 * self.frame_time_ms) as usize, 5) as f64
        } else {
            self.frame_time_max_ms
        };
        for (i, sample) in samples.enumerate() {
            let t = offset * Affine::translate((i as f64 * bar_extent, graph_max_height));
            // The height of each sample is based on its ratio to the maximum observed frame time.
            let sample_ms = ((*sample as f64) * 0.001).min(display_max);
            let h = sample_ms / display_max;
            let s = Affine::scale_non_uniform(1., -h);
            let color = match *sample {
                ..=16_667 => Color::rgb8(100, 143, 255),
                ..=33_334 => Color::rgb8(255, 176, 0),
                _ => Color::rgb8(220, 38, 127),
            };
            sb.fill(
                Fill::NonZero,
                t * Affine::translate((
                    left_margin_padding,
                    (1 + labels.len()) as f64 * text_height,
                )) * s,
                color,
                None,
                &bar,
            );
        }
        // Draw horizontal lines to mark 8.33ms, 16.33ms, and 33.33ms
        let marker = [
            MoveTo((0., graph_max_height).into()),
            LineTo((graph_max_width, graph_max_height).into()),
        ];
        let thresholds = [8.33, 16.66, 33.33];
        let thres_text_height = graph_max_height * 0.05;
        let thres_text_height_2 = thres_text_height * 0.5;
        for t in thresholds.iter().filter(|&&t| t < display_max) {
            let y = t / display_max;
            text.add(
                sb,
                None,
                thres_text_height as f32,
                Some(&Brush::Solid(Color::WHITE)),
                offset
                    * Affine::translate((
                        left_margin,
                        (2. - y) * graph_max_height + thres_text_height_2,
                    )),
                &format!("{}", t),
            );
            sb.stroke(
                &Stroke::new((graph_max_height * 0.01) as f32),
                offset * Affine::translate((left_margin_padding, (1. - y) * graph_max_height)),
                Color::WHITE,
                None,
                &marker,
            );
        }
    }
}

pub struct Sample {
    pub frame_time_us: u64,
}

pub struct Stats {
    count: usize,
    sum: u64,
    min: u64,
    max: u64,
    samples: VecDeque<u64>,
}

impl Stats {
    pub fn new() -> Stats {
        Stats {
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: u64::MIN,
            samples: VecDeque::with_capacity(SLIDING_WINDOW_SIZE),
        }
    }

    pub fn samples(&self) -> impl Iterator<Item = &u64> {
        self.samples.iter()
    }

    pub fn snapshot(&self) -> Snapshot {
        let frame_time_ms = (self.sum as f64 / self.count as f64) * 0.001;
        let fps = 1000. / frame_time_ms;
        Snapshot {
            fps,
            frame_time_ms,
            frame_time_min_ms: self.min as f64 * 0.001,
            frame_time_max_ms: self.max as f64 * 0.001,
        }
    }

    pub fn clear_min_and_max(&mut self) {
        self.min = u64::MAX;
        self.max = u64::MIN;
    }

    pub fn add_sample(&mut self, sample: Sample) {
        let oldest = if self.count < SLIDING_WINDOW_SIZE {
            self.count += 1;
            None
        } else {
            self.samples.pop_front()
        };
        let micros = sample.frame_time_us;
        self.sum += micros;
        self.samples.push_back(micros);
        if let Some(oldest) = oldest {
            self.sum -= oldest;
        }
        self.min = self.min.min(micros);
        self.max = self.max.max(micros);
    }
}

fn round_up(n: usize, f: usize) -> usize {
    n - 1 - (n - 1) % f + f
}

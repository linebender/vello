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
    peniko::{Brush, Color, Fill},
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
    ) where
        T: Iterator<Item = &'a u64>,
    {
        let width = (viewport_width * 0.4).max(200.).min(600.);
        let height = width * 0.6;
        let x_offset = viewport_width - width;
        let y_offset = viewport_height - height;
        let offset = Affine::translate((x_offset, y_offset));
        let text_height = height * 0.1;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        sb.fill(
            Fill::NonZero,
            offset,
            &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
            None,
            &Rect::new(0., 0., width, height),
        );
        text.add(
            sb,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((left_margin, text_height)),
            &format!("Frame Time: {:.2} ms", self.frame_time_ms),
        );
        text.add(
            sb,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((left_margin, 2. * text_height)),
            &format!("Frame Time (min): {:.2} ms", self.frame_time_min_ms),
        );
        text.add(
            sb,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((left_margin, 3. * text_height)),
            &format!("Frame Time (max): {:.2} ms", self.frame_time_max_ms),
        );
        text.add(
            sb,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((left_margin, 4. * text_height)),
            &format!("Resolution: {viewport_width}x{viewport_height}"),
        );
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
        let graph_max_height = height * 0.5;
        let graph_max_width = width - 2. * left_margin;
        let bar_extent = graph_max_width / (SLIDING_WINDOW_SIZE as f64);
        let bar_width = bar_extent * 0.3;
        let bar = [
            MoveTo((0., graph_max_height).into()),
            LineTo((0., 0.).into()),
            LineTo((bar_width, 0.).into()),
            LineTo((bar_width, graph_max_height).into()),
        ];
        for (i, sample) in samples.enumerate() {
            let t = offset * Affine::translate(((i as f64) * bar_extent, graph_max_height));
            // The height of each sample is based on its ratio to the maximum observed frame time.
            // Currently this maximum scale is sticky and a high temporary spike will permanently
            // shrink the draw size of the overall average sample, so scale the size non-linearly to
            // emphasize smaller samples.
            let h = (*sample as f64) * 0.001 / self.frame_time_max_ms;
            let s = Affine::scale_non_uniform(1., -h.sqrt());
            sb.fill(
                Fill::NonZero,
                t * Affine::translate((left_margin, 5. * text_height)) * s,
                Color::rgb8(0, 240, 0),
                None,
                &bar,
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

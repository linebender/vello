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
use std::{collections::VecDeque, time::Instant};
use vello::{
    kurbo::{Affine, Rect},
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
    pub fn draw_layer(
        &self,
        sb: &mut SceneBuilder,
        text: &mut SimpleText,
        viewport_width: f64,
        viewport_height: f64,
    ) {
        let width = (viewport_width * 0.4).max(200.).min(400.);
        let height = width * 0.3;
        let x_offset = viewport_width - width;
        let y_offset = viewport_height - height;
        let offset = Affine::translate((x_offset, y_offset));
        let text_height = height * 0.2;
        let left_margin = width * 0.03;
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
    }
}

pub struct Stats {
    count: usize,
    sum: u64,
    min: u64,
    max: u64,
    samples: VecDeque<u64>,
}

pub struct FrameScope<'a> {
    stats: &'a mut Stats,
    start: Instant,
}

impl<'a> Drop for FrameScope<'a> {
    fn drop(&mut self) {
        self.stats
            .add_sample(self.start.elapsed().as_micros() as u64);
    }
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

    pub fn frame_scope<'a>(&'a mut self) -> FrameScope<'a> {
        FrameScope {
            stats: self,
            start: Instant::now(),
        }
    }

    fn add_sample(&mut self, micros: u64) {
        let oldest = if self.count < SLIDING_WINDOW_SIZE {
            self.count += 1;
            None
        } else {
            self.samples.pop_front()
        };
        self.sum += micros;
        self.samples.push_back(micros);
        if let Some(oldest) = oldest {
            self.sum -= oldest;
        }
        if micros < self.min {
            self.min = micros;
        }
        if micros > self.max {
            self.max = micros;
        }
    }
}

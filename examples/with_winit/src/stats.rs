// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use scenes::SimpleText;
use std::{collections::VecDeque, time::Duration};
use vello::{
    kurbo::{Affine, Line, PathEl, Rect, Stroke},
    peniko::{Brush, Color, Fill},
    AaConfig, BumpAllocators, Scene,
};
use wgpu_profiler::GpuTimerQueryResult;

const SLIDING_WINDOW_SIZE: usize = 100;

#[derive(Debug)]
pub struct Snapshot {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub frame_time_min_ms: f64,
    pub frame_time_max_ms: f64,
}

impl Snapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn draw_layer<'a, T>(
        &self,
        scene: &mut Scene,
        text: &mut SimpleText,
        viewport_width: f64,
        viewport_height: f64,
        samples: T,
        bump: Option<BumpAllocators>,
        vsync: bool,
        aa_config: AaConfig,
    ) where
        T: Iterator<Item = &'a u64>,
    {
        let width = (viewport_width * 0.4).max(200.).min(600.);
        let height = width * 0.7;
        let x_offset = viewport_width - width;
        let y_offset = viewport_height - height;
        let offset = Affine::translate((x_offset, y_offset));

        // Draw the background
        scene.fill(
            Fill::NonZero,
            offset,
            &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
            None,
            &Rect::new(0., 0., width, height),
        );

        let mut labels = vec![
            format!("Frame Time: {:.2} ms", self.frame_time_ms),
            format!("Frame Time (min): {:.2} ms", self.frame_time_min_ms),
            format!("Frame Time (max): {:.2} ms", self.frame_time_max_ms),
            format!("VSync: {}", if vsync { "on" } else { "off" }),
            format!(
                "AA method: {}",
                match aa_config {
                    AaConfig::Area => "Analytic Area",
                    AaConfig::Msaa16 => "16xMSAA",
                    AaConfig::Msaa8 => "8xMSAA",
                }
            ),
            format!("Resolution: {viewport_width}x{viewport_height}"),
        ];
        if let Some(bump) = &bump {
            if bump.failed >= 1 {
                labels.push("Allocation Failed!".into());
            }
            labels.push(format!("binning: {}", bump.binning));
            labels.push(format!("ptcl: {}", bump.ptcl));
            labels.push(format!("tile: {}", bump.tile));
            labels.push(format!("segments: {}", bump.segments));
            labels.push(format!("blend: {}", bump.blend));
        }

        // height / 2 is dedicated to the text labels and the rest is filled by the bar graph.
        let text_height = height * 0.5 / (1 + labels.len()) as f64;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }
        text.add(
            scene,
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
            #[allow(clippy::match_overlapping_arm)]
            let color = match *sample {
                ..=16_667 => Color::rgb8(100, 143, 255),
                ..=33_334 => Color::rgb8(255, 176, 0),
                _ => Color::rgb8(220, 38, 127),
            };
            scene.fill(
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
                scene,
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
            scene.stroke(
                &Stroke::new(graph_max_height * 0.01),
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

const COLORS: &[Color] = &[
    Color::AQUA,
    Color::RED,
    Color::ALICE_BLUE,
    Color::YELLOW,
    Color::GREEN,
    Color::BLUE,
    Color::ORANGE,
    Color::WHITE,
];

pub fn draw_gpu_profiling(
    scene: &mut Scene,
    text: &mut SimpleText,
    viewport_width: f64,
    viewport_height: f64,
    profiles: &[GpuTimerQueryResult],
) {
    if profiles.is_empty() {
        return;
    }
    let width = (viewport_width * 0.3).clamp(150., 450.);
    let height = width * 1.5;
    let y_offset = viewport_height - height;
    let offset = Affine::translate((0., y_offset));

    // Draw the background
    scene.fill(
        Fill::NonZero,
        offset,
        &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
        None,
        &Rect::new(0., 0., width, height),
    );
    // Find the range of the samples, so we can normalise them
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    let mut max_depth = 0;
    let mut depth = 0;
    let mut count = 0;
    traverse_profiling(profiles, &mut |profile, stage| {
        match stage {
            TraversalStage::Enter => {
                count += 1;
                min = min.min(profile.time.start);
                max = max.max(profile.time.end);
                max_depth = max_depth.max(depth);
                // Apply a higher depth to the children
                depth += 1;
            }
            TraversalStage::Leave => depth -= 1,
        }
    });
    let total_time = max - min;
    {
        let labels = [
            format!("GPU Time: {:.2?}", Duration::from_secs_f64(total_time)),
            "Press P to save a trace".to_string(),
        ];

        // height / 5 is dedicated to the text labels and the rest is filled by the frame time.
        let text_height = height * 0.2 / (1 + labels.len()) as f64;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }

        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }
    }
    let timeline_start_y = height * 0.21;
    let timeline_range_y = height * 0.78;
    let timeline_range_end = timeline_start_y + timeline_range_y;

    // Add 6 items worth of margin
    let text_height = timeline_range_y / (6 + count) as f64;
    let left_margin = width * 0.35;
    let mut cur_text_y = timeline_start_y;
    let mut cur_index = 0;
    let mut depth = 0;
    // Leave 1 bar's worth of margin
    let depth_width = width * 0.28 / (max_depth + 1) as f64;
    let depth_size = depth_width * 0.8;
    traverse_profiling(profiles, &mut |profile, stage| {
        if let TraversalStage::Enter = stage {
            let start_normalised =
                ((profile.time.start - min) / total_time) * timeline_range_y + timeline_start_y;
            let end_normalised =
                ((profile.time.end - min) / total_time) * timeline_range_y + timeline_start_y;

            let color = COLORS[cur_index % COLORS.len()];
            let x = width * 0.01 + (depth as f64 * depth_width);
            scene.fill(
                Fill::NonZero,
                offset,
                &Brush::Solid(color),
                None,
                &Rect::new(x, start_normalised, x + depth_size, end_normalised),
            );

            let mut text_start = start_normalised;
            let nested = !profile.nested_queries.is_empty();
            if nested {
                // If we have children, leave some more space for them
                text_start -= text_height * 0.7;
            }
            let this_time = profile.time.end - profile.time.start;
            // Highlight as important if more than 10% of the total time, or more than 1ms
            let slow = this_time * 20. >= total_time || this_time >= 0.001;
            let text_y = text_start
                // Ensure that we don't overlap the previous item
                .max(cur_text_y)
                // Ensure that all remaining items can fit
                .min(timeline_range_end - (count - cur_index) as f64 * text_height);
            let (text_height, text_color) = if slow {
                (text_height, Color::WHITE)
            } else {
                (text_height * 0.6, Color::LIGHT_GRAY)
            };
            let text_size = (text_height * 0.9) as f32;
            // Text is specified by the baseline, but the y positions all refer to the top of the text
            cur_text_y = text_y + text_height;
            let label = format!(
                "{:.2?} - {:.30}",
                Duration::from_secs_f64(this_time),
                profile.label
            );
            scene.fill(
                Fill::NonZero,
                offset,
                &Brush::Solid(color),
                None,
                &Rect::new(
                    width * 0.31,
                    cur_text_y - text_size as f64 * 0.7,
                    width * 0.34,
                    cur_text_y,
                ),
            );
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(text_color)),
                offset * Affine::translate((left_margin, cur_text_y)),
                &label,
            );
            if !nested && slow {
                scene.stroke(
                    &Stroke::new(2.),
                    offset,
                    &Brush::Solid(color),
                    None,
                    &Line::new(
                        (x + depth_size, (end_normalised + start_normalised) / 2.),
                        (width * 0.31, cur_text_y - text_size as f64 * 0.35),
                    ),
                );
            }
            cur_index += 1;
            // Higher depth applies only to the children
            depth += 1;
        } else {
            depth -= 1;
        }
    });
}

enum TraversalStage {
    Enter,
    Leave,
}

fn traverse_profiling(
    profiles: &[GpuTimerQueryResult],
    callback: &mut impl FnMut(&GpuTimerQueryResult, TraversalStage),
) {
    for profile in profiles {
        callback(profile, TraversalStage::Enter);
        traverse_profiling(&profile.nested_queries, &mut *callback);
        callback(profile, TraversalStage::Leave);
    }
}

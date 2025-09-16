// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// There is not much science behind this constant. It was instead determined by doing profiling
/// and finding a constant that seems to strike a reasonable trade-off between not causing too
/// big batch sizes
pub(crate) const COST_THRESHOLD: f32 = 250.0;

use crate::dispatch::multi_threaded::RenderTaskType;
use crate::kurbo::{Affine, PathEl, PathSeg, Point, segments};

/// Try to estimate the cost of the render task.
pub(crate) fn estimate_render_task_cost(task: &RenderTaskType, paths: &[PathEl]) -> f32 {
    const LAYER_COST: f32 = 10.0;

    match task {
        RenderTaskType::FillPath {
            path_range,
            transform,
            ..
        } => {
            let path = &paths[path_range.start as usize..path_range.end as usize];
            estimate_path_cost(segments(path.iter().copied()), *transform, false)
        }
        RenderTaskType::StrokePath {
            path_range,
            transform,
            ..
        } => {
            let path = &paths[path_range.start as usize..path_range.end as usize];
            estimate_path_cost(segments(path.iter().copied()), *transform, true)
        }
        RenderTaskType::PushLayer { clip_path, .. } => {
            LAYER_COST
                + clip_path
                    .as_ref()
                    .map(|(path_range, transform)| {
                        let path = &paths[path_range.start as usize..path_range.end as usize];
                        estimate_path_cost(segments(path.iter().copied()), *transform, false)
                    })
                    .unwrap_or(0.0)
        }
        RenderTaskType::PopLayer => LAYER_COST,
        RenderTaskType::WideCommand { .. } => LAYER_COST,
    }
}

/// Try to estimate an (admittedly somewhat abstract) "path cost".
///
/// The main point here is that when sending paths to a thread to convert them into sparse strip
/// representation, we might want to batch them. This is especially the case for small, line-only
/// geometries, where handling each path separately would lead to a huge overhead.
///
/// Because of this, before rendering a path, we try to estimate a very rough cost based on
/// the following attributes that (usually) have an impact on rendering times:
///
/// - Number of line segments (more line segments -> more work during strip rendering).
/// - Number of curve segments (same as line segments, plus we need to flatten them first).
/// - Path length (if the path is longer, the covered area is _likely_ to also be larger). However,
///   the path length usually grows much faster than the render time, so we only apply a very small
///   fractional value.
/// - Strokes (if we are stroking a path, there is even more overhead for stroke expansion before doing
///   flattening and strip rendering).
pub(crate) fn estimate_path_cost(
    path: impl IntoIterator<Item = PathSeg>,
    transform: Affine,
    is_stroke: bool,
) -> f32 {
    let cost_data = PathCostData::new(path, transform);

    // Once again, those constants were not determined "scientifically" in any way, but
    // are instead based on intuition as well as a number of experiments.
    const CURVE_MULTIPLIER: f32 = 2.5;
    const STROKE_MULTIPLIER: f32 = 1.5;

    let mut cost = cost_data.num_line_segments as f32;
    cost += cost_data.num_curve_segments as f32 * CURVE_MULTIPLIER;
    cost += cost * (cost_data.path_length as f32 / 1024.0);

    cost *= if is_stroke { STROKE_MULTIPLIER } else { 1.0 };
    cost
}

struct PathCostData {
    num_line_segments: u64,
    num_curve_segments: u64,
    path_length: f64,
}

impl PathCostData {
    fn new(path: impl IntoIterator<Item = PathSeg>, transform: Affine) -> Self {
        let mut num_line_segments = 0;
        let mut num_curve_segments = 0;
        let mut path_length = 0.0;

        let mut register_path_length = |mut p0: Point, mut p1: Point| {
            p0 = transform * p0;
            p1 = transform * p1;
            // We don't sqrt here because it's too expensive, we just want a rough estimate.
            let dx = (p1.x - p0.x).abs();
            let dy = (p1.y - p0.y).abs();
            path_length += dx + dy;
        };

        for seg in path.into_iter() {
            match seg {
                PathSeg::Line(l) => {
                    num_line_segments += 1;

                    register_path_length(l.p0, l.p1);
                }
                PathSeg::Quad(q) => {
                    num_curve_segments += 1;

                    register_path_length(q.p0, q.p2);
                }
                PathSeg::Cubic(c) => {
                    num_curve_segments += 1;

                    register_path_length(c.p0, c.p3);
                }
            }
        }

        Self {
            num_line_segments,
            num_curve_segments,
            path_length,
        }
    }
}

// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use {
    bytemuck::{Pod, Zeroable},
    std::{collections::BTreeSet, fmt},
    vello_encoding::LineSoup,
};

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Pod, Zeroable)]
#[repr(C)]
pub struct LineEndpoint {
    pub path_ix: u32,

    // Coordinates in IEEE-754 32-bit float representation
    // We use u32 here because we are comparing bit patterns rather than proximity, to evaluate exact watertightness
    // To accelerate this, we use a BTreeSet, which don't support f32 values directly.
    pub x: u32,
    pub y: u32,
}

impl LineEndpoint {
    pub fn new(line: &LineSoup, start_or_end: bool) -> Self {
        let (x, y) = if start_or_end {
            (line.p0[0], line.p0[1])
        } else {
            (line.p1[0], line.p1[1])
        };
        Self {
            path_ix: line.path_ix,
            x: x.to_bits(),
            y: y.to_bits(),
        }
    }
}

impl fmt::Debug for LineEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point")
            .field("path_ix", &self.path_ix)
            .field("x", &f32::from_bits(self.x))
            .field("y", &f32::from_bits(self.y))
            .finish()
    }
}

pub(crate) fn validate_line_soup(lines: &[LineSoup]) -> Vec<LineEndpoint> {
    let mut points = BTreeSet::new();
    for line in lines {
        let pts = [
            LineEndpoint::new(line, true),
            LineEndpoint::new(line, false),
        ];
        for p in pts {
            if !points.remove(&p) {
                points.insert(p);
            }
        }
    }
    if !points.is_empty() {
        log::warn!("Unpaired points are present: {points:#?}");
    }
    points.into_iter().collect()
}

// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `vello_cpu`'s `push_clip_rect` must render byte-identically to pushing the
//! same rectangle as a clip path, on both the single- and multi-threaded
//! dispatchers, in every routing regime (integer edges, fractional edges, and
//! the rotated fallback to the path pipeline).

use vello_common::kurbo::{Affine, Point, Rect, Shape};
use vello_common::peniko::Color;
use vello_common::pixmap::Pixmap;
use vello_cpu::{RenderContext, RenderSettings, Resources};

const W: u16 = 128;
const H: u16 = 64;

fn render(num_threads: u16, rect_clip: bool, clip: Rect, transform: Affine) -> Pixmap {
    let settings = RenderSettings {
        num_threads,
        ..Default::default()
    };
    let mut ctx = RenderContext::new_with(W, H, settings);
    let mut resources = Resources::new();

    ctx.set_paint(Color::new([0.086, 0.106, 0.133, 1.0]));
    ctx.fill_rect(&Rect::new(0.0, 0.0, f64::from(W), f64::from(H)));

    ctx.set_transform(transform);
    if rect_clip {
        ctx.push_clip_rect(&clip);
    } else {
        ctx.push_clip_path(&clip.to_path(0.1));
    }
    ctx.set_transform(Affine::IDENTITY);

    ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
    ctx.fill_rect(&Rect::new(10.3, 8.7, 40.6, 30.2));
    ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
    ctx.fill_rect(&Rect::new(60.5, 12.25, 95.75, 40.5));

    ctx.pop_clip_path();
    ctx.flush();

    let mut pixmap = Pixmap::new(W, H);
    ctx.render(&mut pixmap, &mut resources);
    pixmap
}

fn assert_parity(clip: Rect, transform: Affine, what: &str) {
    for num_threads in [0, 2] {
        let via_path = render(num_threads, false, clip, transform);
        let via_rect = render(num_threads, true, clip, transform);
        let diffs = via_path
            .data_as_u8_slice()
            .iter()
            .zip(via_rect.data_as_u8_slice())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs, 0,
            "{what} (num_threads={num_threads}): {diffs} differing bytes"
        );
    }
}

#[test]
fn cpu_push_clip_rect_matches_clip_path_integer() {
    assert_parity(
        Rect::new(16.0, 12.0, 112.0, 56.0),
        Affine::IDENTITY,
        "integer rect clip",
    );
}

#[test]
fn cpu_push_clip_rect_matches_clip_path_fractional() {
    assert_parity(
        Rect::new(16.4, 12.6, 111.5, 55.25),
        Affine::IDENTITY,
        "fractional rect clip",
    );
}

#[test]
fn cpu_push_clip_rect_matches_clip_path_rotated() {
    // Rotation forces the fallback to the path pipeline.
    assert_parity(
        Rect::new(30.0, 10.0, 100.0, 50.0),
        Affine::rotate_about(0.3, Point::new(64.0, 32.0)),
        "rotated rect clip",
    );
}

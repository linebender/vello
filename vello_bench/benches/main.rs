// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(dead_code, reason = "Might be unused on platforms not supporting SIMD")]

use vello_bench::registry::Registry;
use vello_bench::{allocator, fine, flatten, glyph, integration, pixmap, sort, strip, tile};

fn benchmarks() -> Registry {
    let mut r = Registry::default();
    pixmap::pixmap(&mut r);
    allocator::allocator(&mut r);
    tile::tile(&mut r);
    strip::render_strips(&mut r);
    strip::render_strips_cull(&mut r);
    strip::render_rect(&mut r);
    flatten::flatten(&mut r);
    flatten::strokes(&mut r);
    glyph::glyph(&mut r);
    fine::fill(&mut r);
    fine::strip(&mut r);
    fine::pack(&mut r);
    fine::gradient(&mut r);
    fine::rounded_blurred_rect(&mut r);
    fine::blend(&mut r);
    fine::image(&mut r);
    sort::sort(&mut r);
    integration::images(&mut r);
    r
}

tango_bench::tango_benchmarks!(benchmarks());

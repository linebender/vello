// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic sanity checks that no panic happen for pixmaps with different sizes.

use crate::util::{get_ctx, render_pixmap};

#[test]
fn empty_1x1() {
    let ctx = get_ctx(1, 1, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_5x1() {
    let ctx = get_ctx(5, 1, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_1x5() {
    let ctx = get_ctx(1, 5, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_3x10() {
    let ctx = get_ctx(3, 10, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_23x45() {
    let ctx = get_ctx(23, 45, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_50x50() {
    let ctx = get_ctx(50, 50, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_463x450() {
    let ctx = get_ctx(463, 450, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_1134x1376() {
    let ctx = get_ctx(1134, 1376, true);
    render_pixmap(&ctx);
}

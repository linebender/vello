// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for rendering after freeing the memory grown for earlier scenes.

use crate::load_image;
use crate::renderer::Renderer;
use crate::util::layout_glyphs_roboto;
use vello_common::color::palette::css::{
    CYAN, DEEP_PINK, GOLD, LAVENDER, MIDNIGHT_BLUE, SLATE_BLUE, TOMATO, WHITE,
};
use vello_common::kurbo::{Affine, Circle, Rect, Shape};
use vello_common::paint::Image;
use vello_common::peniko::{BlendMode, Compose, Gradient, ImageQuality, ImageSampler, Mix};
use vello_dev_macros::vello_test;

fn tile(index: u8) -> Rect {
    let x = 12.0 + 72.0 * f64::from(index);
    Rect::new(x, 18.0, x + 64.0, 82.0)
}

// TODO: Add tests so we can actually measure the memory consumption too.

#[vello_test(transparent, gpu_tolerance = 1, width = 300)]
fn shrink_between_scenes(ctx: &mut impl Renderer) {
    ctx.set_paint(
        Gradient::new_sweep((150.0, 50.0), 0.0, 360.0)
            .with_stops(&[DEEP_PINK, GOLD, DEEP_PINK][..]),
    );
    ctx.fill_rect(&Rect::new(0.0, 0.0, 300.0, 100.0));
    ctx.flush();
    ctx.render();

    ctx.reset_and_shrink();

    let linear = tile(0);
    ctx.set_paint(
        Gradient::new_linear((linear.x0, linear.y0), (linear.x1, linear.y1))
            .with_stops(&[TOMATO, GOLD][..]),
    );
    ctx.fill_rect(&linear);

    let radial = tile(1);
    ctx.set_paint(
        Gradient::new_radial(radial.center(), 46.0)
            .with_stops(&[CYAN, SLATE_BLUE, MIDNIGHT_BLUE][..]),
    );
    ctx.fill_rect(&radial);

    let image = tile(2);
    let source = ctx.get_image_source(load_image!("color_grid_16x16"));
    ctx.set_paint_transform(Affine::translate((image.x0, image.y0)) * Affine::scale(4.0));
    ctx.set_paint(Image {
        image: source,
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..Default::default()
        },
    });
    ctx.fill_rect(&image);
    ctx.set_paint_transform(Affine::IDENTITY);

    let blend = tile(3);
    ctx.set_paint(LAVENDER);
    ctx.fill_rect(&blend);
    ctx.push_clip_layer(&blend.to_path(0.1));
    ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
    let center = blend.center();
    for (color, dx, dy) in [(DEEP_PINK, -8.0, -6.0), (CYAN, 8.0, -6.0), (GOLD, 0.0, 8.0)] {
        ctx.set_paint(color.with_alpha(0.8));
        ctx.fill_path(&Circle::new((center.x + dx, center.y + dy), 16.0).to_path(0.1));
    }
    ctx.pop_layer();
    ctx.pop_layer();
}

fn draw_glyph_card(ctx: &mut impl Renderer, enable_caching: bool, text: &str) {
    let card = Rect::new(12.0, 18.0, 148.0, 82.0);
    ctx.set_paint(
        Gradient::new_linear((card.x0, card.y0), (card.x1, card.y1))
            .with_stops(&[SLATE_BLUE, MIDNIGHT_BLUE][..]),
    );
    ctx.fill_rect(&card);

    let font_size = 48.0;
    let (font, glyphs) = layout_glyphs_roboto(text, font_size);
    ctx.set_paint(WHITE);
    ctx.set_transform(Affine::translate((49.0, 67.0)));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    ctx.set_transform(Affine::IDENTITY);
}

#[vello_test(transparent, width = 160, glyph)]
fn shrink_reuses_gradient_and_glyphs(ctx: &mut impl Renderer, enable_caching: bool) {
    draw_glyph_card(ctx, enable_caching, "AB");
    ctx.flush();
    ctx.render();

    ctx.reset_and_shrink();

    draw_glyph_card(ctx, enable_caching, "AC");
}

#[vello_test(gpu_tolerance = 1, width = 300)]
fn shrink_regrows_layer_textures(ctx: &mut impl Renderer) {
    let canvas = Rect::new(0.0, 0.0, 300.0, 100.0);
    ctx.push_opacity_layer(0.9);
    ctx.push_clip_layer(&canvas.to_path(0.1));
    ctx.set_paint(
        Gradient::new_sweep((150.0, 50.0), 0.0, 360.0)
            .with_stops(&[DEEP_PINK, GOLD, DEEP_PINK][..]),
    );
    ctx.fill_rect(&canvas);
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.flush();
    ctx.render();

    ctx.reset_and_shrink();

    let card = Rect::new(12.0, 18.0, 288.0, 82.0);
    ctx.push_clip_layer(&card.to_path(0.1));
    ctx.push_opacity_layer(0.9);
    ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
    for (color, x) in [(DEEP_PINK, 105.0), (CYAN, 150.0), (GOLD, 195.0)] {
        ctx.set_paint(color);
        ctx.fill_path(&Circle::new((x, 50.0), 28.0).to_path(0.1));
    }
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();
}

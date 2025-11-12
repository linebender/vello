// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests demonstrating the filter effects API usage.

use crate::util::circular_star;
use crate::{renderer::Renderer, util::layout_glyphs_roboto};
use vello_common::color::AlphaColor;
use vello_common::filter_effects::*;
use vello_common::kurbo::Rect;
use vello_cpu::color::palette::css::{REBECCA_PURPLE, WHITE};
use vello_cpu::kurbo::Affine;
use vello_cpu::{
    Mask, Pixmap, RenderContext,
    color::palette::css::{PURPLE, ROYAL_BLUE, SEA_GREEN, TOMATO, VIOLET},
    kurbo::{BezPath, Circle, Point, Shape as _},
    peniko::{BlendMode, Compose, Mix},
};
use vello_dev_macros::vello_test;

/// Test flood filter filling a star shape with solid color using a mask.
///
/// Note: SVG-compliant flood would use `feComposite` with `operator="in"`, which requires
/// implementing the composite primitive and filter subregions.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_flood(ctx: &mut impl Renderer) {
    let filter_flood = Filter::from_primitive(FilterPrimitive::Flood { color: TOMATO });
    let star_path = circular_star(Point::new(50.0, 50.0), 5, 20.0, 40.0);

    // Create a mask from the star shape
    let width = ctx.width();
    let height = ctx.height();
    let mut mask_pixmap = Pixmap::new(width, height);
    let mut mask_ctx = RenderContext::new_with(
        width,
        height,
        vello_cpu::RenderSettings {
            level: vello_cpu::Level::fallback(),
            num_threads: 0,
            render_mode: vello_cpu::RenderMode::default(),
        },
    );
    mask_ctx.set_paint(WHITE);
    mask_ctx.fill_path(&star_path);
    mask_ctx.flush();
    mask_ctx.render_to_pixmap(&mut mask_pixmap);

    let mask = Mask::new_alpha(&mask_pixmap);

    // Apply flood filter with the mask to fill only the star area
    ctx.push_layer(None, None, None, Some(mask), Some(filter_flood));
    // This color will be replaced by the flood
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&star_path);
    ctx.pop_layer();
}

/// Test Gaussian blur with small radius (`std_deviation` = 2.0, no decimation).
/// Uses direct separable convolution at full resolution.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_gaussian_blur_no_decimation(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test Gaussian blur with larger radius (`std_deviation` = 4.0, uses decimation).
/// Uses multi-scale downsampling for performance.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_gaussian_blur_with_decimation(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test drop shadow filter on text glyph.
/// Creates a blurred, offset shadow beneath the original graphic.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_drop_shadow(ctx: &mut impl Renderer) {
    let font_size: f32 = 80_f32;
    let (font, glyphs) = layout_glyphs_roboto("A", font_size);

    let filter = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 16.0,
        dy: 8.0,
        std_deviation: 2.0,
        color: REBECCA_PURPLE,
        edge_mode: EdgeMode::None,
    });
    ctx.push_filter_layer(filter);
    ctx.set_transform(Affine::translate((24.0, f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
    ctx.pop_layer();
}

/// Test `set_filter_effect` and `reset_filter_effect` API.
/// Applies filters to individual draw calls without creating layers.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_set_effect(ctx: &mut impl Renderer) {
    let filter_drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 2.0,
        dy: 2.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 255),
        edge_mode: EdgeMode::None,
    });
    let filter_gaussian_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });

    let width = 32.;
    let overlap = 6.;
    let between = 20.;

    let x = 8.;
    let y = 8.;
    let mut left = x;
    let mut top = y;

    ctx.set_filter_effect(filter_gaussian_blur);
    ctx.set_paint(ROYAL_BLUE);
    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));

    ctx.set_paint(PURPLE);
    left = x + width + between;
    top = y;
    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
    ctx.reset_filter_effect();

    ctx.set_paint(TOMATO);
    left = x + width - overlap;
    top = y + width - overlap;
    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));

    ctx.set_filter_effect(filter_drop_shadow);
    ctx.set_paint(VIOLET);
    left = x;
    top = y + width + between;
    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
    ctx.reset_filter_effect();

    ctx.set_paint(SEA_GREEN);
    left = x + width + between;
    top = y + width + between;
    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
}

/// Test filter interactions with layers, clips, blend modes, and opacity.
/// 9 scenarios testing filters at various depths, with clips, opacity, blend modes, etc.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_varying_depths_clips_and_compositions(ctx: &mut impl Renderer) {
    let filter_drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 2.0,
        dy: 2.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 255),
        edge_mode: EdgeMode::None,
    });
    let filter_gaussian_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });

    let spacing = 32.;
    let width = 10.;
    let overlap = 2.;
    let between = 6.;

    // Test 1
    let mut x = 4.;
    let mut y = 4.;
    let mut left = x;
    let mut top = y;
    {
        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(filter_gaussian_blur.clone());
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                {
                    ctx.push_filter_layer(filter_drop_shadow.clone());
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 2
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(filter_drop_shadow.clone());
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_filter_layer(filter_gaussian_blur.clone());
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(VIOLET);
            left = x;
            top = y + width + between;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(SEA_GREEN);
                left = x + width + between;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 3
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(filter_gaussian_blur.clone());
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 4
    x = 4.;
    y += spacing;
    left = x;
    top = y;
    let mut circle_path = Circle::new((x + 13., y + 13.), 13.).to_path(0.1);
    {
        ctx.push_layer(
            Some(&circle_path),
            None,
            None,
            None,
            Some(filter_drop_shadow.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 5
    x += spacing;
    left = x;
    top = y;
    let mut quad_path = BezPath::new();
    quad_path.move_to((x, y));
    quad_path.line_to((x + 26., y + 5.));
    quad_path.line_to((x + 30., y + 21.));
    quad_path.line_to((x + 5., y + 30.));
    quad_path.close_path();
    {
        ctx.push_layer(
            Some(&quad_path),
            None,
            None,
            None,
            Some(filter_drop_shadow.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 6
    x += spacing;
    left = x;
    top = y;
    circle_path = Circle::new((x + 13., y + 13.), 13.).to_path(0.1);
    {
        ctx.push_layer(
            Some(&circle_path),
            None,
            None,
            None,
            Some(filter_gaussian_blur.clone()),
        );
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 7
    x = 4.;
    y += spacing;
    left = x;
    top = y;
    {
        ctx.push_layer(None, None, Some(0.5), None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            {
                ctx.push_filter_layer(filter_gaussian_blur.clone());
                ctx.set_paint(PURPLE);
                left = x + width + between;
                top = y;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                ctx.pop_layer();
            }
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(TOMATO);
                left = x + width - overlap;
                top = y + width - overlap;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(filter_gaussian_blur.clone());
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.set_paint(SEA_GREEN);
                        left = x + width + between;
                        top = y + width + between;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
        }
        ctx.pop_layer();
    }

    // Test 8
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, None);
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_layer(None, None, None, None, None);
                ctx.set_paint(VIOLET);
                left = x;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_layer(None, None, None, None, None);
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        {
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::DestOut)),
                None,
                None,
                Some(filter_gaussian_blur.clone()),
            );
            ctx.set_paint(TOMATO);
            left = x + width - overlap;
            top = y + width - overlap;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }

    // Test 9
    x += spacing;
    left = x;
    top = y;
    {
        ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 2.0,
            edge_mode: EdgeMode::None,
        }));
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 2.0,
                edge_mode: EdgeMode::None,
            }));
            ctx.set_paint(PURPLE);
            left = x + width + between;
            top = y;
            ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
            {
                ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                    std_deviation: 2.0,
                    edge_mode: EdgeMode::None,
                }));
                ctx.set_paint(VIOLET);
                left = x;
                top = y + width + between;
                ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                {
                    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                        std_deviation: 2.0,
                        edge_mode: EdgeMode::None,
                    }));
                    ctx.set_paint(SEA_GREEN);
                    left = x + width + between;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    {
                        ctx.push_filter_layer(Filter::from_primitive(
                            FilterPrimitive::GaussianBlur {
                                std_deviation: 2.0,
                                edge_mode: EdgeMode::None,
                            },
                        ));
                        ctx.set_paint(TOMATO);
                        left = x + width - overlap;
                        top = y + width - overlap;
                        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }
}

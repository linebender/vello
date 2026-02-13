// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests demonstrating the filter effects API usage.

use crate::util::circular_star;
use crate::{renderer::Renderer, util::layout_glyphs_roboto};
use vello_common::color::AlphaColor;
use vello_common::color::palette::css::{
    BLACK, PURPLE, REBECCA_PURPLE, ROYAL_BLUE, SEA_GREEN, TOMATO, VIOLET,
};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_cpu::color::palette::css::{BLUE, GREEN, RED};
use vello_cpu::kurbo::Dashes;
use vello_dev_macros::vello_test;

// TODO: We are purposefully using multiple of WideTile width/height here, because the implementation
// currently works incorrectly if it's not the case. Once the issue as been fixed, we should update
// this test to use normal dimensions.
#[vello_test(skip_hybrid, skip_multithreaded, width = 256, height = 40)]
fn filter_flood(ctx: &mut impl Renderer) {
    let filter_flood = Filter::from_primitive(FilterPrimitive::Flood { color: TOMATO });

    ctx.push_filter_layer(filter_flood);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 8.0, 256.0, 32.0));
    ctx.pop_layer();
}

/// Test flood filter filling a star shape with solid color using a mask.
///
/// Note: SVG-compliant flood would use `feComposite` with `operator="in"`, which requires
/// implementing the composite primitive and filter subregions.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_flood_star(ctx: &mut impl Renderer) {
    let filter_flood = Filter::from_primitive(FilterPrimitive::Flood { color: TOMATO });
    let star_path = circular_star(Point::new(50.0, 50.0), 5, 20.0, 40.0);

    // Apply flood filter with clip to fill only the star area.
    // We are purposefully first pushing a clip path and then rendering the filter layer instead
    // of combining the both, because doing both at the same time is a special case which we are
    // not trying to test here.
    ctx.push_clip_layer(&star_path);
    ctx.push_filter_layer(filter_flood);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&star_path);
    ctx.pop_layer();
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

/// Test drop shadow on a simple rectangle.
/// Verifies the offset pixel optimization works correctly with different offsets.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_drop_shadow_corners(ctx: &mut impl Renderer) {
    // Layout parameters
    let margin = 8.0;
    let size = 20.0;
    let shadow_offset = 6.0;
    let shadow_blur = 2.0;

    // Calculate positions for 3x3 grid
    let left = margin;
    let center_x = (100.0 - size) / 2.0;
    let right = 100.0 - margin - size;

    let top = margin;
    let center_y = (100.0 - size) / 2.0;
    let bottom = 100.0 - margin - size;

    ctx.set_paint(ROYAL_BLUE);

    // Top-left corner: shadow to upper-left
    let filter_tl = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: -shadow_offset,
        dy: -shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_tl);
    ctx.fill_rect(&Rect::new(left, top, left + size, top + size));

    // Top center: shadow upward (dy only)
    let filter_tc = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 0.0,
        dy: -shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_tc);
    ctx.fill_rect(&Rect::new(center_x, top, center_x + size, top + size));

    // Top-right corner: shadow to upper-right
    let filter_tr = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: shadow_offset,
        dy: -shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_tr);
    ctx.fill_rect(&Rect::new(right, top, right + size, top + size));

    // Left center: shadow leftward (dx only)
    let filter_lc = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: -shadow_offset,
        dy: 0.0,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_lc);
    ctx.fill_rect(&Rect::new(left, center_y, left + size, center_y + size));

    // Center: shadow downward-right
    let filter_c = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: shadow_offset,
        dy: shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_c);
    ctx.fill_rect(&Rect::new(
        center_x,
        center_y,
        center_x + size,
        center_y + size,
    ));

    // Right center: shadow rightward (dx only)
    let filter_rc = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: shadow_offset,
        dy: 0.0,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_rc);
    ctx.fill_rect(&Rect::new(right, center_y, right + size, center_y + size));

    // Bottom-left corner: shadow to lower-left
    let filter_bl = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: -shadow_offset,
        dy: shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_bl);
    ctx.fill_rect(&Rect::new(left, bottom, left + size, bottom + size));

    // Bottom center: shadow downward (dy only)
    let filter_bc = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 0.0,
        dy: shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_bc);
    ctx.fill_rect(&Rect::new(center_x, bottom, center_x + size, bottom + size));

    // Bottom-right corner: shadow to lower-right
    let filter_br = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: shadow_offset,
        dy: shadow_offset,
        std_deviation: shadow_blur,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::Duplicate,
    });
    ctx.set_filter_effect(filter_br);
    ctx.fill_rect(&Rect::new(right, bottom, right + size, bottom + size));

    ctx.reset_filter_effect();
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

    // Test 1: Gaussian blur and drop shadow filters both applied at depth 3 within nested layers.
    // Tests that multiple different filters work correctly when deeply nested in layer hierarchy.
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

    // Test 2: Drop shadow filter at depth 0, gaussian blur at depth 1, followed by nested layers.
    // Tests multiple filters at different depths with mixed layer types.
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

    // Test 3: Gaussian blur filter at depth 0 with deeply nested plain layers inside.
    // Tests that filter applied to outermost layer correctly affects all nested content.
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

    // Test 4: Drop shadow filter with circular clip path.
    // Tests filter interaction with clipping (circular mask).
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

    // Test 5: Drop shadow filter with quadrilateral clip path.
    // Tests filter interaction with clipping (complex polygon mask).
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

    // Test 6: Gaussian blur filter with circular clip path.
    // Tests gaussian blur interaction with clipping (different filter type than Test 4).
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

    // Test 7: Filters nested within layer with opacity (0.5).
    // Tests that filters correctly interact with opacity settings.
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

    // Test 8: Gaussian blur filter with DestOut blend mode.
    // Tests filter interaction with non-standard blend modes.
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

    // Test 9: Five levels of nested gaussian blur filters.
    // Tests deeply nested filter layers and their cumulative effect.
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

/// Test that blur filters correctly expand bounds when the layer is rotated.
///
/// This verifies that the expansion calculation uses `transform_rect_bbox` to account for
/// the full transformation matrix (including rotation and shear), rather than just extracting
/// x/y scales separately. A 45-degree rotation should produce a diamond-shaped blur.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_rotated_blur(ctx: &mut impl Renderer) {
    let filter_gaussian_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    });
    let center = Point::new(50.0, 50.0);
    ctx.set_transform(Affine::rotate_about(std::f64::consts::PI / 4.0, center));

    let width = 24.;
    let overlap = 6.;
    let between = 12.;

    let x = 21.;
    let y = 21.;
    let mut left = x;
    let mut top = y;

    let circle_path = Circle::new(center, 25.).to_path(0.1);
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
}

/// Test that zero blur acts as identity (no-op).
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_gaussian_blur_zero(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 0.0,
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(25.0, 25.0, 75.0, 75.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test drop shadow with sub-pixel offsets.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_drop_shadow_fractional_offset(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 2.5,
        dy: 3.7,
        std_deviation: 1.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(30.0, 30.0, 70.0, 70.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(ROYAL_BLUE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test drop shadow with zero offset (shadow directly behind).
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_drop_shadow_zero_offset(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 0.0,
        dy: 0.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(30.0, 30.0, 70.0, 70.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(ROYAL_BLUE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test offset filter primitive.
///
/// This shifts content within a filter layer and should not clip content to the original bounds.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_offset(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::Offset {
        dx: 18.0,
        dy: -12.0,
    });
    let star_path = circular_star(Point::new(50.0, 50.0), 7, 10.0, 22.0);

    // Draw the unfiltered star as an outline at the original position.
    ctx.set_paint(ROYAL_BLUE);
    ctx.set_stroke(Stroke::new(1.5));
    ctx.stroke_path(&star_path);

    // Draw a marker rect at a known coordinate.
    //
    // This avoids trying to reason about pixel movement from an anti-aliased stroke edge.
    let marker = Rect::new(49.0, 27.0, 53.0, 31.0);
    ctx.set_paint(SEA_GREEN);
    ctx.fill_rect(&marker);

    // Draw the filtered (shifted) star as a fill and stroke, then draw the marker through the
    // filter layer as well.
    ctx.push_filter_layer(filter);
    ctx.set_paint(TOMATO);
    ctx.fill_path(&star_path);
    ctx.set_paint(BLACK);
    ctx.set_stroke(Stroke::new(1.5));
    ctx.stroke_path(&star_path);
    // With (dx, dy) = (18, -12) this should land at (67, 15).
    ctx.set_paint(VIOLET);
    ctx.fill_rect(&marker);
    ctx.pop_layer();
}

/// Test blur with various transforms (translate, rotate, scale, skew).
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_transformed_blur(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 3.0,
        edge_mode: EdgeMode::None,
    });

    ctx.set_transform(
        Affine::translate((55.0, 5.0))
            * Affine::rotate(std::f64::consts::PI / 4.0)
            * Affine::scale(2.0)
            * Affine::skew(0.3, 0.2),
    );
    let rect = Rect::new(0.0, 0.0, 20.0, 30.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test filter layer with no content drawn.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_empty_layers(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    });

    ctx.push_filter_layer(filter.clone());
    ctx.push_filter_layer(filter.clone());
    ctx.push_filter_layer(filter.clone());
    // Draw nothing
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();
}

/// Test nested filter layers (blur inside drop shadow).
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_nested_layers(ctx: &mut impl Renderer) {
    let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });
    let shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 12.0,
        dy: 12.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 180),
        edge_mode: EdgeMode::None,
    });

    ctx.push_filter_layer(shadow);
    ctx.push_filter_layer(blur);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(25.0, 25.0, 75.0, 75.0));
    ctx.pop_layer();
    ctx.pop_layer();
}

/// Test blur with very large `std_deviation`.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_extreme_blur(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 20.0,
        edge_mode: EdgeMode::None,
    });
    let rect = Rect::new(25.0, 25.0, 75.0, 75.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect);
    ctx.pop_layer();
}

/// Test filter on semi-transparent shapes.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_transparent_shapes(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 3.0,
        edge_mode: EdgeMode::None,
    });

    // Fully opaque shape (left)
    let rect1 = Rect::new(10.0, 25.0, 40.0, 75.0).to_path(0.1);
    ctx.push_filter_layer(filter.clone());
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect1);
    ctx.pop_layer();

    // Semi-transparent shape (right)
    let rect2 = Rect::new(60.0, 25.0, 90.0, 75.0).to_path(0.1);
    ctx.push_filter_layer(filter);
    ctx.set_paint(AlphaColor::from_rgba8(150, 100, 200, 128)); // 50% transparent
    ctx.fill_path(&rect2);
    ctx.pop_layer();
}

/// Test filter on stroked paths.
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_stroked_paths(ctx: &mut impl Renderer) {
    use vello_common::kurbo::{Cap, Join, Stroke};

    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });

    let stroke = Stroke {
        width: 4.0,
        join: Join::Round,
        miter_limit: 4.0,
        start_cap: Cap::Round,
        end_cap: Cap::Round,
        dash_pattern: Dashes::default(),
        dash_offset: 0.0,
    };

    let rect = Rect::new(25.0, 25.0, 75.0, 75.0);

    ctx.push_filter_layer(filter);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
    ctx.pop_layer();
}

/// Test filter on shapes at canvas boundaries.
///
/// TODO: This test currently demonstrates a bug where filters render incorrectly
/// when filtered elements are near or extend beyond viewport boundaries.
/// See: <https://github.com/linebender/vello/issues/1304>
#[vello_test(skip_hybrid, skip_multithreaded)]
fn issue_filter_canvas_boundaries(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 5.0,
        edge_mode: EdgeMode::None,
    });

    // Top-left corner
    let rect_tl = Rect::new(-25.0, -25.0, 35.0, 35.0).to_path(0.1);
    ctx.push_filter_layer(filter.clone());
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&rect_tl);
    ctx.pop_layer();

    // Top-right corner
    let rect_tr = Rect::new(65.0, -25.0, 125.0, 35.0).to_path(0.1);
    ctx.push_filter_layer(filter.clone());
    ctx.set_paint(ROYAL_BLUE);
    ctx.fill_path(&rect_tr);
    ctx.pop_layer();

    // Bottom-left corner
    let rect_bl = Rect::new(-25.0, 65.0, 35.0, 125.0).to_path(0.1);
    ctx.push_filter_layer(filter.clone());
    ctx.set_paint(TOMATO);
    ctx.fill_path(&rect_bl);
    ctx.pop_layer();

    // Bottom-right corner
    let rect_br = Rect::new(65.0, 65.0, 125.0, 125.0).to_path(0.1);
    ctx.push_filter_layer(filter);
    ctx.set_paint(VIOLET);
    ctx.fill_path(&rect_br);
    ctx.pop_layer();
}

pub(crate) fn blur_with_edge_mode(ctx: &mut impl Renderer, edge_mode: EdgeMode) {
    let filter = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 6.0,
        edge_mode,
    });

    let step = 256.0 / 3.0;

    ctx.push_filter_layer(filter);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, step, 100.0));
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(step, 0.0, 2.0 * step, 100.0));
    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(2.0 * step, 0.0, 3.0 * step, 100.0));
    ctx.pop_layer();
}

// TODO: Currently, these tests have a width/height that is a multiple of a wide tile,
// because edge modes currently don't handle other widths/heights correctly. Once that is
// fixed, we should change the tests back to 100x100 to exercise that path as well.

#[vello_test(skip_hybrid, skip_multithreaded, width = 256, height = 100)]
fn filter_gaussian_blur_edge_mode_duplicate(ctx: &mut impl Renderer) {
    blur_with_edge_mode(ctx, EdgeMode::Duplicate);
}

#[vello_test(skip_hybrid, skip_multithreaded, width = 256, height = 100)]
fn filter_gaussian_blur_edge_mode_wrap(ctx: &mut impl Renderer) {
    blur_with_edge_mode(ctx, EdgeMode::Wrap);
}

#[vello_test(skip_hybrid, skip_multithreaded, width = 256, height = 100)]
fn filter_gaussian_blur_edge_mode_mirror(ctx: &mut impl Renderer) {
    blur_with_edge_mode(ctx, EdgeMode::Mirror);
}

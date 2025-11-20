// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter example showing deeply nested clipping.
//!
//! This scene is based on the `filter_varying_depths_clips_and_compositions` test.
//! See: `sparse_strips/vello_sparse_tests/tests/filter.rs`

use crate::{ExampleScene, RenderingContext};
use vello_common::color::AlphaColor;
use vello_common::color::palette::css::{PURPLE, ROYAL_BLUE, SEA_GREEN, TOMATO, VIOLET};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Circle, Rect, Shape};
use vello_common::peniko::{BlendMode, Compose, Mix};

/// Filter scene state
#[derive(Debug)]
pub struct FilterScene {}

impl ExampleScene for FilterScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        ctx.set_transform(root_transform);

        let filter_drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
            dx: 20.0,
            dy: 20.0,
            std_deviation: 4.0,
            color: AlphaColor::from_rgba8(255, 255, 255, 128),
            edge_mode: EdgeMode::None,
        });
        let filter_gaussian_blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 10.0,
            edge_mode: EdgeMode::None,
        });

        let spacing = 380.;
        let width = 120.;
        let between = 60.;
        let start_x = 80.;
        let start_y = 80.;

        // Clip path sizes scaled to encompass the full pattern
        let clip_circle_radius = width + between / 2.0;
        let clip_quad_size = 2.0 * width + between;

        let mut x = start_x;
        let mut y = start_y;
        let mut left = x;
        let mut top = y;

        // Test 1: Gaussian blur and drop shadow filters both applied at depth 3 within nested layers.
        // Tests that multiple different filters work correctly when deeply nested in layer hierarchy.
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
        x = start_x;
        y += spacing;
        left = x;
        top = y;
        let mut circle_path = Circle::new(
            (x + clip_circle_radius, y + clip_circle_radius),
            clip_circle_radius,
        )
        .to_path(0.1);
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
        quad_path.line_to((x + clip_quad_size * 0.87, y + clip_quad_size * 0.17));
        quad_path.line_to((x + clip_quad_size, y + clip_quad_size * 0.7));
        quad_path.line_to((x + clip_quad_size * 0.17, y + clip_quad_size));
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
        circle_path = Circle::new(
            (x + clip_circle_radius, y + clip_circle_radius),
            clip_circle_radius,
        )
        .to_path(0.1);
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
        x = start_x;
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
                    left = x + width / 2.0 + between / 2.0;
                    top = y + width / 2.0 + between / 2.0;
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
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
                left = x + width / 2.0 + between / 2.0;
                top = y + width / 2.0 + between / 2.0;
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
                        ctx.push_filter_layer(Filter::from_primitive(
                            FilterPrimitive::GaussianBlur {
                                std_deviation: 2.0,
                                edge_mode: EdgeMode::None,
                            },
                        ));
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
                            left = x + width / 2.0 + between / 2.0;
                            top = y + width / 2.0 + between / 2.0;
                            ctx.fill_rect(&Rect::from_points(
                                (left, top),
                                (left + width, top + width),
                            ));
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
}

impl FilterScene {
    /// Create a new `FilterScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for FilterScene {
    fn default() -> Self {
        Self::new()
    }
}

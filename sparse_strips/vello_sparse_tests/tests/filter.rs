// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests demonstrating the filter effects API usage.

use crate::renderer::Renderer;
use vello_common::color::AlphaColor;
use vello_common::filter_effects::*;
use vello_common::kurbo::Rect;
use vello_cpu::{
    color::palette::css::{PURPLE, ROYAL_BLUE, SEA_GREEN, TOMATO, VIOLET},
    kurbo::{BezPath, Circle, Shape as _},
    peniko::{BlendMode, Compose, Mix},
};
use vello_dev_macros::vello_test;

/// Simple filter function usage
#[vello_test(skip_hybrid, skip_multithreaded)]
fn filter_varying_depths_clips_and_compositions(ctx: &mut impl Renderer) {
    let filter_drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 2.0,
        dy: 2.0,
        std_deviation: 4.0,
        color: AlphaColor::from_rgba8(0, 0, 0, 255),
    });
    let filter_gaussian_blur =
        Filter::from_primitive(FilterPrimitive::GaussianBlur { std_deviation: 2.0 });

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
                    ctx.push_layer(None, None, None, None, Some(filter_gaussian_blur.clone()));
                    ctx.set_paint(VIOLET);
                    left = x;
                    top = y + width + between;
                    ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
                    ctx.pop_layer();
                }
                {
                    ctx.push_layer(None, None, None, None, Some(filter_drop_shadow.clone()));
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
        ctx.push_layer(None, None, None, None, Some(filter_drop_shadow.clone()));
        ctx.set_paint(ROYAL_BLUE);
        ctx.fill_rect(&Rect::from_points((left, top), (left + width, top + width)));
        {
            ctx.push_layer(None, None, None, None, Some(filter_gaussian_blur.clone()));
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
        ctx.push_layer(None, None, None, None, Some(filter_gaussian_blur.clone()));
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
                ctx.push_layer(None, None, None, None, Some(filter_gaussian_blur.clone()));
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
                    ctx.push_layer(None, None, None, None, Some(filter_gaussian_blur.clone()));
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
}

// /// Filter layer usage (affects multiple elements)
// #[vello_test]
// fn filter_layers(ctx: &mut impl Renderer) {
//     // Push an offset layer that affects all subsequent drawing
//     ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
//         dx: 5.0,
//         dy: 5.0,
//     }));

//     // Everything drawn here will be offset
//     ctx.set_paint(PaintType::Solid(RED));
//     ctx.fill_rect(&Rect::new(10.0, 10.0, 100.0, 100.0));

//     ctx.set_paint(PaintType::Solid(GREEN));
//     ctx.fill_rect(&Rect::new(50.0, 50.0, 150.0, 150.0));

//     // Pop the offset layer
//     ctx.pop_layer();

//     // This rectangle won't be offset
//     ctx.set_paint(PaintType::Solid(BLUE));
//     ctx.fill_rect(&Rect::new(200.0, 10.0, 300.0, 100.0));
// }

// /// Filter primitives usage
// #[vello_test]
// fn filter_primitives(ctx: &mut impl Renderer) {
//     // Color matrix transformation (sepia effect)
//     ctx.set_paint(PaintType::Solid(RED));
//     ctx.set_filter_effect(Filter::from_primitive(FilterPrimitive::ColorMatrix {
//         matrix: matrices::SEPIA,
//     }));
//     ctx.fill_rect(&Rect::new(10.0, 10.0, 100.0, 100.0));

//     // Morphological operation (dilate)
//     ctx.set_filter_effect(Filter::from_primitive(FilterPrimitive::Morphology {
//         operator: MorphologyOperator::Dilate,
//         radius: 2.0,
//     }));
//     ctx.fill_rect(&Rect::new(120.0, 10.0, 220.0, 100.0));

//     // Turbulence noise
//     ctx.set_filter_effect(Filter::from_primitive(FilterPrimitive::Turbulence {
//         base_frequency: 0.1,
//         num_octaves: 3,
//         seed: 42,
//         turbulence_type: TurbulenceType::FractalNoise,
//     }));
//     ctx.fill_rect(&Rect::new(240.0, 10.0, 340.0, 100.0));
// }

// /// Complex filter graph (DAG) usage
// #[vello_test]
// fn complex_filter_graph(ctx: &mut impl Renderer) {
//     // Build a complex drop shadow effect using helper function
//     let shadow_graph = build_drop_shadow_graph();

//     // Create filter system and apply the complex shadow filter
//     ctx.push_filter_layer(Filter {
//         graph: shadow_graph,
//         bounds: Some(Rect::new(0.0, 0.0, 800.0, 600.0)),
//     });

//     // Draw rectangle that will have shadow
//     ctx.set_paint(PaintType::Solid(YELLOW));
//     ctx.fill_rect(&Rect::new(50.0, 50.0, 300.0, 100.0));

//     ctx.pop_layer();
// }

// /// Chained filters
// #[vello_test]
// fn chained_filters(ctx: &mut impl Renderer) {
//     // Create a filter chain: blur -> brightness -> contrast using helper function
//     let filter_chain = build_chained_filter_graph();

//     ctx.push_filter_layer(Filter {
//         graph: filter_chain,
//         bounds: Some(Rect::new(0.0, 0.0, 400.0, 300.0)),
//     });

//     // Draw multiple elements that will all be affected by the filter chain
//     ctx.set_paint(PaintType::Solid(RED));
//     ctx.fill_rect(&Rect::new(10.0, 10.0, 100.0, 100.0));

//     ctx.set_paint(PaintType::Solid(GREEN));
//     ctx.fill_rect(&Rect::new(50.0, 50.0, 150.0, 150.0));

//     ctx.set_paint(PaintType::Solid(BLUE));
//     ctx.fill_rect(&Rect::new(90.0, 90.0, 190.0, 190.0));

//     ctx.pop_layer();
// }

// /// Filter bounds optimization
// #[vello_test]
// fn filter_bounds_optimization(ctx: &mut impl Renderer) {
//     // Create a bounded blur filter using helper function
//     ctx.push_filter_layer(Filter {
//         graph: build_blur_graph(10.0),
//         bounds: Some(Rect::new(0.0, 0.0, 200.0, 200.0)),
//     });

//     // Only content within the bounds will be processed by the blur
//     ctx.set_paint(PaintType::Solid(BLUE));
//     // Inside bounds - will be blurred
//     ctx.fill_rect(&Rect::new(10.0, 10.0, 180.0, 180.0));

//     ctx.pop_layer();

//     // This content is outside the filter bounds and won't be affected
//     ctx.set_paint(PaintType::Solid(BLUE));
//     // Outside bounds - no blur
//     ctx.fill_rect(&Rect::new(250.0, 10.0, 400.0, 180.0));
// }

// /// Build a complex drop shadow filter graph
// /// @see <https://drafts.fxtf.org/filter-effects/#dropshadowEquivalent>
// fn build_drop_shadow_graph() -> FilterGraph {
//     let mut graph = FilterGraph::new();

//     // Step 1: Extract alpha channel for shadow
//     let alpha_extract = graph.add(
//         FilterPrimitive::ColorMatrix {
//             matrix: matrices::ALPHA_TO_BLACK,
//         },
//         Some(FilterInputs::single(FilterInput::Source(
//             FilterSource::SourceGraphic,
//         ))),
//     );

//     // Step 2: Offset the shadow
//     let shadow_offset = graph.add(
//         FilterPrimitive::Offset { dx: 4.0, dy: 4.0 },
//         Some(FilterInputs::single(FilterInput::Result(alpha_extract))),
//     );

//     // Step 3: Blur the shadow
//     let shadow_blur = graph.add(
//         FilterPrimitive::GaussianBlur { std_deviation: 3.0 },
//         Some(FilterInputs::single(FilterInput::Result(shadow_offset))),
//     );

//     // Step 4: Create shadow color (no inputs - generates content)
//     let shadow_color = graph.add(FilterPrimitive::Flood { color: YELLOW }, None);

//     // Step 5: Composite shadow color with blurred shape
//     let shadow_composite = graph.add(
//         FilterPrimitive::Composite {
//             operator: CompositeOperator::In,
//         },
//         Some(FilterInputs::dual(
//             FilterInput::Result(shadow_color),
//             FilterInput::Result(shadow_blur),
//         )),
//     );

//     // Step 6: Composite shadow with original
//     let final_composite = graph.add(
//         FilterPrimitive::Composite {
//             operator: CompositeOperator::Over,
//         },
//         Some(FilterInputs::dual(
//             FilterInput::Result(shadow_composite),
//             FilterInput::Source(FilterSource::SourceGraphic),
//         )),
//     );

//     // Set the output
//     graph.set_output(final_composite);

//     graph
// }

// /// Build a chained filter graph: blur -> brightness -> contrast
// fn build_chained_filter_graph() -> FilterGraph {
//     let mut graph = FilterGraph::new();

//     // Step 1: Blur filter
//     let blur_result = graph.add(
//         FilterPrimitive::GaussianBlur { std_deviation: 2.0 },
//         Some(FilterInputs::single(FilterInput::Source(
//             FilterSource::SourceGraphic,
//         ))),
//     );

//     // Step 2: Brightness filter
//     let brightness_result = graph.add(
//         FilterPrimitive::ComponentTransfer {
//             red_function: Some(TransferFunction::Linear {
//                 slope: 1.2,
//                 intercept: 0.0,
//             }),
//             green_function: Some(TransferFunction::Linear {
//                 slope: 1.2,
//                 intercept: 0.0,
//             }),
//             blue_function: Some(TransferFunction::Linear {
//                 slope: 1.2,
//                 intercept: 0.0,
//             }),
//             alpha_function: None,
//         },
//         Some(FilterInputs::single(FilterInput::Result(blur_result))),
//     );

//     // Step 3: Contrast filter
//     let contrast_slope = 1.5;
//     let contrast_intercept = 0.5 * (1.0 - contrast_slope);
//     let contrast_result = graph.add(
//         FilterPrimitive::ComponentTransfer {
//             red_function: Some(TransferFunction::Linear {
//                 slope: contrast_slope,
//                 intercept: contrast_intercept,
//             }),
//             green_function: Some(TransferFunction::Linear {
//                 slope: contrast_slope,
//                 intercept: contrast_intercept,
//             }),
//             blue_function: Some(TransferFunction::Linear {
//                 slope: contrast_slope,
//                 intercept: contrast_intercept,
//             }),
//             alpha_function: None,
//         },
//         Some(FilterInputs::single(FilterInput::Result(brightness_result))),
//     );

//     // Set the final output
//     graph.set_output(contrast_result);

//     graph
// }

// /// Build a simple blur filter graph
// fn build_blur_graph(std_deviation: f32) -> FilterGraph {
//     let mut graph = FilterGraph::new();

//     // Add blur filter with source graphic input
//     let blur_result = graph.add(
//         FilterPrimitive::GaussianBlur { std_deviation },
//         Some(FilterInputs::single(FilterInput::Source(
//             FilterSource::SourceGraphic,
//         ))),
//     );

//     // Set the output
//     graph.set_output(blur_result);

//     graph
// }

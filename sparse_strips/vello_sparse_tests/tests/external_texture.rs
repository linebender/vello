// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// TODO: Increase test coverage to cover things like tinted external textures, etc.

mod tests {
    use std::sync::Arc;

    use vello_common::color;
    use vello_common::color::{AlphaColor, PremulRgba8, Srgb};
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, Circle, Rect, Shape};
    use vello_common::paint::{Image, ImageSource};
    use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
    use vello_common::pixmap::Pixmap;
    use vello_dev_macros::vello_test;
    use vello_hybrid::SampleRect;

    use crate::load_image;
    use crate::renderer::Renderer;

    const SPRITES: [RectU16; 4] = [
        // Checkmark.
        RectU16::new(0, 0, 56, 56),
        // Eyes
        RectU16::new(56, 0, 112, 42),
        // Confetti
        RectU16::new(0, 56, 54, 112),
        // Cowboy Hat Face
        RectU16::new(56, 42, 115, 100),
    ];

    fn solid_pixmap(r: u8, g: u8, b: u8, a: u8) -> Arc<Pixmap> {
        Arc::new(Pixmap::from_parts(
            vec![PremulRgba8::from_u8_array([r, g, b, a])],
            1,
            1,
        ))
    }

    fn texture_rect_at(x: f64, y: f64) -> [SampleRect; 1] {
        [SampleRect {
            source_region: RectU16::new(0, 0, 1, 1),
            transform: Affine::translate((x, y)) * Affine::scale(40.),
        }]
    }

    fn draw_atlas_rect(ctx: &mut impl Renderer, image: ImageSource, rect: Rect) {
        ctx.set_paint_transform(Affine::IDENTITY);
        ctx.set_paint(Image {
            image,
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&rect);
    }

    fn texture_circle(
        ctx: &mut impl Renderer,
        texture_id: vello_hybrid::TextureId,
        circle: Circle,
    ) {
        let clip = circle.to_path(0.1);
        let rect = circle.bounding_box();

        ctx.push_clip_path(&clip);
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Medium,
            [SampleRect {
                source_region: RectU16::new(0, 0, 1, 1),
                transform: Affine::translate((rect.x0, rect.y0))
                    * Affine::scale_non_uniform(rect.width(), rect.height()),
            }],
        );
        ctx.pop_clip_path();
    }

    fn painted_circle(ctx: &mut impl Renderer, color: AlphaColor<Srgb>, circle: Circle) {
        ctx.set_paint(color);
        ctx.fill_path(&circle.to_path(0.1));
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_composite(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Low,
            [SampleRect {
                source_region: SPRITES[0],
                transform: Affine::translate((12., 15.)),
            }],
        );
        ctx.set_paint(color::palette::css::PALE_GOLDENROD.with_alpha(0.7));
        ctx.fill_rect(&Rect::new(10., 7., 65., 55.));
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Low,
            [SampleRect {
                source_region: SPRITES[3],
                transform: Affine::translate((25., 25.)),
            }],
        );
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_opaque_interleaving(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(solid_pixmap(192, 0, 0, 192));

        ctx.set_paint(AlphaColor::from_rgba8(0, 0, 255, 255));
        ctx.fill_rect(&Rect::new(8., 8., 64., 64.));

        ctx.draw_texture_rects(texture_id, ImageQuality::Low, texture_rect_at(28., 28.));

        ctx.set_paint(AlphaColor::from_rgba8(0, 255, 0, 255));
        ctx.fill_rect(&Rect::new(48., 16., 88., 56.));
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_runs_with_opaque_prefix(ctx: &mut impl Renderer) {
        let red_texture = ctx.register_external_texture(solid_pixmap(255, 0, 0, 255));
        let green_texture = ctx.register_external_texture(solid_pixmap(0, 255, 0, 255));

        ctx.set_paint(AlphaColor::from_rgba8(32, 32, 32, 255));
        ctx.fill_rect(&Rect::new(4., 4., 92., 92.));

        ctx.draw_texture_rects(red_texture, ImageQuality::Low, texture_rect_at(8., 8.));

        ctx.set_paint(AlphaColor::from_rgba8(255, 255, 255, 128));
        ctx.fill_rect(&Rect::new(24., 24., 72., 72.));

        ctx.draw_texture_rects(green_texture, ImageQuality::Low, texture_rect_at(28., 28.));
        ctx.draw_texture_rects(red_texture, ImageQuality::Low, texture_rect_at(48., 48.));
    }

    #[vello_test(hybrid_only)]
    fn external_texture_atlas_interleaving(ctx: &mut impl Renderer) {
        let atlas_red = ctx.get_image_source(solid_pixmap(254, 0, 0, 254));
        let external_green = ctx.register_external_texture(solid_pixmap(0, 254, 0, 254));
        let atlas_blue = ctx.get_image_source(solid_pixmap(0, 0, 254, 254));
        let external_yellow = ctx.register_external_texture(solid_pixmap(254, 254, 0, 254));
        let atlas_magenta = ctx.get_image_source(solid_pixmap(254, 0, 254, 254));

        draw_atlas_rect(ctx, atlas_red, Rect::new(10., 10., 34., 34.));
        ctx.draw_texture_rects(
            external_green,
            ImageQuality::Low,
            [SampleRect {
                source_region: RectU16::new(0, 0, 1, 1),
                transform: Affine::translate((66., 10.)) * Affine::scale(24.),
            }],
        );
        draw_atlas_rect(ctx, atlas_blue, Rect::new(38., 38., 62., 62.));
        ctx.draw_texture_rects(
            external_yellow,
            ImageQuality::Low,
            [SampleRect {
                source_region: RectU16::new(0, 0, 1, 1),
                transform: Affine::translate((10., 66.)) * Affine::scale(24.),
            }],
        );
        draw_atlas_rect(ctx, atlas_magenta, Rect::new(66., 66., 90., 90.));
    }

    #[vello_test(hybrid_only)]
    fn external_texture_layer_before_external(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(solid_pixmap(128, 0, 0, 128));

        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(AlphaColor::from_rgba8(0, 255, 0, 255));
        ctx.fill_path(&Rect::new(20., 20., 60., 60.).to_path(0.1));
        ctx.pop_layer();

        ctx.push_layer(None, None, None, None, None);
        ctx.draw_texture_rects(texture_id, ImageQuality::Medium, texture_rect_at(40., 40.));
        ctx.pop_layer();
    }

    #[vello_test(hybrid_only)]
    fn external_texture_external_before_layer(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(solid_pixmap(0, 255, 0, 255));

        ctx.push_layer(None, None, None, None, None);
        ctx.draw_texture_rects(texture_id, ImageQuality::Medium, texture_rect_at(20., 20.));
        ctx.pop_layer();

        ctx.push_layer(None, None, None, None, None);
        ctx.set_paint(AlphaColor::from_rgba8(255, 0, 0, 128));
        ctx.fill_path(&Rect::new(40., 40., 80., 80.).to_path(0.1));
        ctx.pop_layer();
    }

    #[vello_test(hybrid_only)]
    fn external_texture_layer_circle_orders(ctx: &mut impl Renderer) {
        let blue_texture = ctx.register_external_texture(solid_pixmap(0, 0, 255, 255));
        let red_texture = ctx.register_external_texture(solid_pixmap(255, 0, 0, 255));
        let green_texture = ctx.register_external_texture(solid_pixmap(0, 255, 0, 255));

        let blue = AlphaColor::from_rgba8(0, 0, 255, 255);
        let green = AlphaColor::from_rgba8(0, 255, 0, 255);

        #[derive(Clone, Copy)]
        enum CirclePos {
            Blue,
            Red,
            Green,
        }

        #[derive(Clone, Copy)]
        enum CircleOp {
            Texture(CirclePos, vello_hybrid::TextureId),
            Paint(CirclePos, AlphaColor<Srgb>),
        }

        let circle = |(x, y): (f64, f64), pos| match pos {
            CirclePos::Blue => Circle::new((x + 25., y + 17.5), 12.5),
            CirclePos::Red => Circle::new((x + 17.5, y + 32.5), 12.5),
            CirclePos::Green => Circle::new((x + 32.5, y + 32.5), 12.5),
        };

        let mut draw_panel = |origin, ops: &[CircleOp]| {
            ctx.push_layer(None, None, None, None, None);
            for op in ops {
                match *op {
                    CircleOp::Texture(pos, texture_id) => {
                        texture_circle(ctx, texture_id, circle(origin, pos));
                    }
                    CircleOp::Paint(pos, color) => {
                        painted_circle(ctx, color, circle(origin, pos));
                    }
                }
            }
            ctx.pop_layer();
        };

        draw_panel(
            (0., 0.),
            &[
                CircleOp::Texture(CirclePos::Blue, blue_texture),
                CircleOp::Texture(CirclePos::Red, red_texture),
                CircleOp::Texture(CirclePos::Green, green_texture),
            ],
        );
        draw_panel(
            (50., 0.),
            &[
                CircleOp::Texture(CirclePos::Red, red_texture),
                CircleOp::Texture(CirclePos::Green, green_texture),
                CircleOp::Texture(CirclePos::Blue, blue_texture),
            ],
        );
        draw_panel(
            (0., 50.),
            &[
                CircleOp::Texture(CirclePos::Green, green_texture),
                CircleOp::Texture(CirclePos::Red, red_texture),
                CircleOp::Texture(CirclePos::Blue, blue_texture),
            ],
        );
        draw_panel(
            (50., 50.),
            &[
                CircleOp::Paint(CirclePos::Blue, blue),
                CircleOp::Texture(CirclePos::Red, red_texture),
                CircleOp::Paint(CirclePos::Green, green),
            ],
        );
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_skewed(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::High,
            [SampleRect {
                source_region: SPRITES[0],
                transform: Affine::translate((15., 15.)) * Affine::skew(0.2, 0.1),
            }],
        );
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_clipped(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        let clip = Circle::new((48., 48.), 24.).to_path(0.1);

        ctx.push_clip_layer(&clip);
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Medium,
            [
                SampleRect {
                    source_region: SPRITES[1],
                    transform: Affine::translate((18., 18.)),
                },
                SampleRect {
                    source_region: SPRITES[3],
                    transform: Affine::translate((34., 34.)),
                },
            ],
        );
        ctx.pop_layer();
    }

    #[vello_test(width = 96, height = 96, hybrid_only, hybrid_tolerance = 2)]
    fn external_texture_blurred(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        let blur = Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 4.0,
            edge_mode: EdgeMode::None,
        });

        ctx.push_filter_layer(blur);
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Low,
            [SampleRect {
                source_region: SPRITES[2],
                transform: Affine::translate((20., 20.)),
            }],
        );
        ctx.pop_layer();
    }

    #[vello_test(width = 192, height = 132, hybrid_only)]
    fn external_texture_many_sprites(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        let placements = [
            (SPRITES[0], 8., 10.),
            (SPRITES[1], 34., 12.),
            (SPRITES[2], 56., 10.),
            (SPRITES[3], 80., 8.),
            (SPRITES[0], 106., 11.),
            (SPRITES[2], 130., 9.),
            (SPRITES[1], 10., 66.),
            (SPRITES[3], 34., 64.),
            (SPRITES[0], 58., 68.),
            (SPRITES[2], 84., 66.),
            (SPRITES[3], 110., 64.),
            (SPRITES[1], 134., 67.),
        ];

        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Low,
            placements.map(|(source_region, x, y)| SampleRect {
                source_region,
                transform: Affine::translate((x, y)),
            }),
        );
    }

    #[vello_test(width = 96, height = 96, hybrid_only, hybrid_tolerance = 2)]
    fn external_texture_with_scene_transform(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));

        ctx.set_transform(
            Affine::translate((20., 0.))
                * Affine::rotate(0.35)
                * Affine::scale_non_uniform(0.85, 1.1),
        );
        ctx.draw_texture_rects(
            texture_id,
            ImageQuality::Medium,
            [
                SampleRect {
                    source_region: SPRITES[0],
                    transform: Affine::translate((6., 8.)),
                },
                SampleRect {
                    source_region: SPRITES[3],
                    transform: Affine::translate((28., 5.)) * Affine::skew(0.18, -0.08),
                },
                SampleRect {
                    source_region: SPRITES[2],
                    transform: Affine::translate((48., 6.)),
                },
            ],
        );
    }
}

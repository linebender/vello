// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod tests {
    use std::sync::Arc;

    use vello_common::color;
    use vello_common::color::{AlphaColor, PremulRgba8, Srgb};
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, Circle, Rect, Shape};
    use vello_common::paint::{CoverageContrast, Image, ImageSource, Tint, TintMode};
    use vello_common::peniko::{Color, Extend, ImageQuality, ImageSampler};
    use vello_common::pixmap::Pixmap;
    use vello_dev_macros::vello_test;
    use vello_hybrid::{ExternalTextureRect, TextureId};

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

    fn pad_sampler(quality: ImageQuality) -> ImageSampler {
        ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        }
    }

    fn texture_rect(
        texture_id: TextureId,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        may_have_transparency: bool,
    ) -> ExternalTextureRect {
        ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(0, 0, 1, 1),
            dest_rect: Rect::new(x, y, x + width, y + height),
            sampler: pad_sampler(ImageQuality::Low),
            may_have_transparency,
        }
    }

    fn texture_rect_at(texture_id: TextureId, x: f64, y: f64) -> ExternalTextureRect {
        texture_rect(texture_id, x, y, 40., 40., true)
    }

    fn sprite_rect(
        texture_id: TextureId,
        source_region: RectU16,
        x: f64,
        y: f64,
        quality: ImageQuality,
    ) -> ExternalTextureRect {
        ExternalTextureRect {
            texture_id,
            src_rect: source_region,
            dest_rect: Rect::new(
                x,
                y,
                x + f64::from(source_region.width()),
                y + f64::from(source_region.height()),
            ),
            sampler: pad_sampler(quality),
            may_have_transparency: true,
        }
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

    fn texture_circle(ctx: &mut impl Renderer, texture_id: TextureId, circle: Circle) {
        let clip = circle.to_path(0.1);
        let rect = circle.bounding_box();

        ctx.push_clip_path(&clip);
        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(0, 0, 1, 1),
            dest_rect: rect,
            sampler: pad_sampler(ImageQuality::Medium),
            may_have_transparency: true,
        });
        ctx.pop_clip_path();
    }

    fn painted_circle(ctx: &mut impl Renderer, color: AlphaColor<Srgb>, circle: Circle) {
        ctx.set_paint(color);
        ctx.fill_path(&circle.to_path(0.1));
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_composite(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        ctx.set_paint_transform(Affine::translate((12., 15.)));
        ctx.draw_texture_rect(sprite_rect(
            texture_id,
            SPRITES[0],
            12.,
            15.,
            ImageQuality::Low,
        ));
        ctx.set_paint(color::palette::css::PALE_GOLDENROD.with_alpha(0.7));
        ctx.fill_rect(&Rect::new(10., 7., 65., 55.));
        ctx.set_paint_transform(Affine::translate((25., 25.)));
        ctx.draw_texture_rect(sprite_rect(
            texture_id,
            SPRITES[3],
            25.,
            25.,
            ImageQuality::Low,
        ));
    }

    #[vello_test(hybrid_only)]
    fn external_texture_repeat(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("color_grid_16x16"));

        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(0, 0, 16, 16),
            dest_rect: Rect::new(5., 5., 95., 95.),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Medium,
                alpha: 1.0,
            },
            may_have_transparency: false,
        });
    }

    #[vello_test(hybrid_only)]
    fn external_texture_with_paint_transform(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("color_grid_16x16"));

        ctx.set_paint_transform(Affine::rotate(0.35) * Affine::scale(2.));
        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(2, 2, 14, 14),
            dest_rect: Rect::new(5., 5., 95., 95.),
            sampler: ImageSampler {
                x_extend: Extend::Reflect,
                y_extend: Extend::Reflect,
                quality: ImageQuality::Medium,
                alpha: 1.0,
            },
            may_have_transparency: false,
        });
    }

    #[vello_test(hybrid_only)]
    fn external_texture_with_scene_transform_2(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("color_grid_16x16"));

        ctx.set_transform(Affine::translate((5., 5.)) * Affine::scale(5.625));
        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(0, 0, 16, 16),
            dest_rect: Rect::new(0., 0., 16., 16.),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Medium,
                alpha: 1.0,
            },
            may_have_transparency: false,
        });
    }

    #[vello_test(hybrid_only)]
    fn external_texture_with_cropped_source(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("color_grid_16x16"));

        ctx.set_paint_transform(Affine::translate((14.0, 14.0)) * Affine::scale(6.0));
        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(2, 2, 14, 14),
            dest_rect: Rect::new(5., 5., 95., 95.),
            sampler: ImageSampler {
                x_extend: Extend::Reflect,
                y_extend: Extend::Reflect,
                quality: ImageQuality::Medium,
                alpha: 1.0,
            },
            may_have_transparency: false,
        });
    }

    #[vello_test(width = 96, height = 96, hybrid_only, hybrid_no_depth)]
    fn external_texture_opaque_interleaving(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(solid_pixmap(192, 0, 0, 192));

        ctx.set_paint(AlphaColor::from_rgba8(0, 0, 255, 255));
        ctx.fill_rect(&Rect::new(8., 8., 64., 64.));

        ctx.draw_texture_rect(texture_rect_at(texture_id, 28., 28.));

        ctx.set_paint(AlphaColor::from_rgba8(0, 255, 0, 255));
        ctx.fill_rect(&Rect::new(48., 16., 88., 56.));
    }

    #[vello_test(width = 96, height = 96, hybrid_only, hybrid_no_depth)]
    fn external_texture_runs_with_opaque_prefix(ctx: &mut impl Renderer) {
        let red_texture = ctx.register_external_texture(solid_pixmap(255, 0, 0, 255));
        let green_texture = ctx.register_external_texture(solid_pixmap(0, 255, 0, 255));

        ctx.set_paint(AlphaColor::from_rgba8(32, 32, 32, 255));
        ctx.fill_rect(&Rect::new(4., 4., 92., 92.));

        ctx.draw_texture_rect(texture_rect_at(red_texture, 8., 8.));

        ctx.set_paint(AlphaColor::from_rgba8(255, 255, 255, 128));
        ctx.fill_rect(&Rect::new(24., 24., 72., 72.));

        ctx.draw_texture_rect(texture_rect_at(green_texture, 28., 28.));
        ctx.draw_texture_rect(texture_rect_at(red_texture, 48., 48.));
    }

    #[vello_test(hybrid_only, hybrid_no_depth)]
    fn external_texture_root_painter_order(ctx: &mut impl Renderer) {
        let coral = ctx.register_external_texture(solid_pixmap(225, 87, 89, 255));
        let teal = ctx.register_external_texture(solid_pixmap(42, 157, 143, 255));
        let navy = ctx.register_external_texture(solid_pixmap(38, 70, 83, 255));
        let gold = ctx.register_external_texture(solid_pixmap(153, 102, 61, 160));
        let tint_source = ctx.register_external_texture(solid_pixmap(255, 255, 255, 255));

        ctx.draw_texture_rect(texture_rect(coral, 6., 6., 88., 88., false));
        ctx.draw_texture_rect(texture_rect(gold, 11.5, 33.5, 77., 33., true));
        ctx.draw_texture_rect(texture_rect(gold, 33.5, 11.5, 33., 77., true));
        ctx.draw_texture_rect(texture_rect(teal, 21.5, 21.5, 57., 57., false));
        ctx.set_tint(Some(Tint {
            color: Color::from_rgba8(128, 102, 204, 160),
            mode: TintMode::Multiply,
            contrast: CoverageContrast::NONE,
        }));
        ctx.draw_texture_rect(texture_rect(tint_source, 16., 38., 68., 24., false));
        ctx.draw_texture_rect(texture_rect(tint_source, 38., 16., 24., 68., false));
        ctx.set_tint(None);
        ctx.draw_texture_rect(texture_rect(navy, 33.5, 33.5, 33., 33., false));
        ctx.draw_texture_rect(texture_rect(gold, 28., 44., 44., 12., true));
        ctx.draw_texture_rect(texture_rect(gold, 44., 28., 12., 44., true));
        ctx.draw_texture_rect(texture_rect(coral, 44., 44., 12., 12., false));
    }

    #[vello_test(hybrid_only)]
    fn external_texture_atlas_interleaving(ctx: &mut impl Renderer) {
        let atlas_red = ctx.get_image_source(solid_pixmap(254, 0, 0, 254));
        let external_green = ctx.register_external_texture(solid_pixmap(0, 254, 0, 254));
        let atlas_blue = ctx.get_image_source(solid_pixmap(0, 0, 254, 254));
        let external_yellow = ctx.register_external_texture(solid_pixmap(254, 254, 0, 254));
        let atlas_magenta = ctx.get_image_source(solid_pixmap(254, 0, 254, 254));

        draw_atlas_rect(ctx, atlas_red, Rect::new(10., 10., 34., 34.));
        ctx.draw_texture_rect(texture_rect(external_green, 66., 10., 24., 24., true));
        draw_atlas_rect(ctx, atlas_blue, Rect::new(38., 38., 62., 62.));
        ctx.draw_texture_rect(texture_rect(external_yellow, 10., 66., 24., 24., true));
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
        ctx.draw_texture_rect(texture_rect_at(texture_id, 40., 40.));
        ctx.pop_layer();
    }

    #[vello_test(hybrid_only)]
    fn external_texture_external_before_layer(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(solid_pixmap(0, 255, 0, 255));

        ctx.push_layer(None, None, None, None, None);
        ctx.draw_texture_rect(texture_rect_at(texture_id, 20., 20.));
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
            Texture(CirclePos, TextureId),
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
        let source_region = SPRITES[0];
        ctx.set_transform(Affine::translate((15., 15.)) * Affine::skew(0.2, 0.1));
        ctx.set_paint_transform(Affine::IDENTITY);
        ctx.draw_texture_rect(ExternalTextureRect {
            texture_id,
            src_rect: source_region,
            dest_rect: Rect::new(
                0.,
                0.,
                f64::from(source_region.width()),
                f64::from(source_region.height()),
            ),
            sampler: pad_sampler(ImageQuality::High),
            may_have_transparency: true,
        });
    }

    #[vello_test(width = 96, height = 96, hybrid_only)]
    fn external_texture_clipped(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));
        let clip = Circle::new((48., 48.), 24.).to_path(0.1);

        ctx.push_clip_layer(&clip);
        ctx.set_paint_transform(Affine::translate((18., 18.)));
        ctx.draw_texture_rect(sprite_rect(
            texture_id,
            SPRITES[1],
            18.,
            18.,
            ImageQuality::Medium,
        ));
        ctx.set_paint_transform(Affine::translate((34., 34.)));
        ctx.draw_texture_rect(sprite_rect(
            texture_id,
            SPRITES[3],
            34.,
            34.,
            ImageQuality::Medium,
        ));
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
        ctx.set_paint_transform(Affine::translate((20., 20.)));
        ctx.draw_texture_rect(sprite_rect(
            texture_id,
            SPRITES[2],
            20.,
            20.,
            ImageQuality::Low,
        ));
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

        for (source_region, x, y) in placements {
            ctx.set_paint_transform(Affine::translate((x, y)));
            ctx.draw_texture_rect(sprite_rect(
                texture_id,
                source_region,
                x,
                y,
                ImageQuality::Low,
            ));
        }
    }

    #[vello_test(width = 96, height = 96, hybrid_only, hybrid_tolerance = 2)]
    fn external_texture_with_scene_transform(ctx: &mut impl Renderer) {
        let texture_id = ctx.register_external_texture(load_image!("glyphs_colr_noto"));

        let scene_transform = Affine::translate((20., 0.))
            * Affine::rotate(0.35)
            * Affine::scale_non_uniform(0.85, 1.1);

        for (source_region, local_transform) in [
            (SPRITES[0], Affine::translate((6., 8.))),
            (
                SPRITES[3],
                Affine::translate((28., 5.)) * Affine::skew(0.18, -0.08),
            ),
            (SPRITES[2], Affine::translate((48., 6.))),
        ] {
            ctx.set_transform(scene_transform * local_transform);
            ctx.set_paint_transform(Affine::IDENTITY);
            ctx.draw_texture_rect(ExternalTextureRect {
                texture_id,
                src_rect: source_region,
                dest_rect: Rect::new(
                    0.,
                    0.,
                    f64::from(source_region.width()),
                    f64::from(source_region.height()),
                ),
                sampler: pad_sampler(ImageQuality::Medium),
                may_have_transparency: true,
            });
        }
    }
}

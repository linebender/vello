use crate::util::{check_ref, get_ctx, layout_glyphs};
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::Affine;

#[test]
fn filled_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "filled_glyphs");
}

#[test]
fn stroked_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .stroke_glyphs(glyphs.into_iter());

    check_ref(&ctx, "stroked_glyphs");
}

#[test]
fn skewed_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.0))
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "skewed_glyphs");
}

#[test]
fn scaled_glyphs() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "scaled_glyphs");
}

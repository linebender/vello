pub mod pico_svg;

use kurbo::Affine;
use pico_svg::Item;
use vello_hybrid::RenderContext;

// Define a render function that works with our pico_svg::Item type
pub fn render_svg(ctx: &mut RenderContext, scale: f64, items: &[Item]) {
    ctx.set_transform(Affine::scale(scale));

    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color.into());
                ctx.fill_path(&fill_item.path.path);
            }
            Item::Stroke(stroke_item) => {
                let style = kurbo::Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color.into());
                ctx.stroke_path(&stroke_item.path.path);
            }
            Item::Group(group_item) => {
                // TODO: apply transform from group
                render_svg(ctx, scale, &group_item.children);
            }
        }
    }
}

// Remove duplicate re-export
// pub use vello_hybrid::utils::render_svg;

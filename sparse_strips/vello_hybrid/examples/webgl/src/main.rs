// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        use vello_common::kurbo::{Affine, Stroke};
        use vello_common::pico_svg::Item;
        use vello_common::pico_svg::PicoSvg;
        use vello_hybrid::Scene;
        use webgl::render_scene;

        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug).unwrap();

        let window = web_sys::window().unwrap();
        let dpr = window.device_pixel_ratio();

        let ghosttiger = include_str!("../../../../../examples/assets/Ghostscript_Tiger.svg");
        let svg = PicoSvg::load(ghosttiger, 6.0).expect("error parsing SVG");

        let width = window.inner_width().unwrap().as_f64().unwrap() as u16 * dpr as u16;
        let height = window.inner_height().unwrap().as_f64().unwrap() as u16 * dpr as u16;

        let mut scene = vello_hybrid::Scene::new(width, height);

        fn render_svg(ctx: &mut Scene, scale: f64, items: &[Item]) {
            fn render_svg_inner(ctx: &mut Scene, items: &[Item], transform: Affine) {
                ctx.set_transform(transform);
                for item in items {
                    match item {
                        Item::Fill(fill_item) => {
                            ctx.set_paint(fill_item.color.into());
                            ctx.fill_path(&fill_item.path);
                        }
                        Item::Stroke(stroke_item) => {
                            let style = Stroke::new(stroke_item.width);
                            ctx.set_stroke(style);
                            ctx.set_paint(stroke_item.color.into());
                            ctx.stroke_path(&stroke_item.path);
                        }
                        Item::Group(group_item) => {
                            render_svg_inner(
                                ctx,
                                &group_item.children,
                                transform * group_item.affine,
                            );
                            ctx.set_transform(transform);
                        }
                    }
                }
            }

            render_svg_inner(ctx, items, Affine::scale(scale));
        }

        render_svg(&mut scene, dpr, &svg.items);

        wasm_bindgen_futures::spawn_local(async move {
            render_scene(scene, width, height).await;
        });
    }
}

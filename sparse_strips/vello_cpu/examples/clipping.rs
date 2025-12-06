// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Applying clip paths using Vello CPU.

use vello_cpu::Pixmap;
use vello_cpu::RenderContext;
use vello_cpu::color::palette::css::{BLUE, RED, WHITE};
use vello_cpu::kurbo::{Circle, Rect, Shape};

fn main() {
    // Clip-paths are a fundamental operation in 2D rendering and allow you
    // to constrain the visible areas of all subsequently drawn paths to the
    // shape of another area. Vello CPU has full support for them and actually
    // provides 2 different ways of using them. Below, we will explore them
    // and explain their difference and when to use which one.

    let mut ctx = RenderContext::new(200, 200);

    // Two example clip shapes. They have a small overlap in the center that forms
    // an ellipse.
    let clip_1 = Circle::new((75.0, 75.0), 50.0).to_path(0.1);
    let clip_2 = Circle::new((125.0, 75.0), 50.0).to_path(0.1);

    // Method 1: Non-isolated clipping using the `push_clip_path` and
    // `pop_clip_path` methods:
    {
        // Let's first create a white background.
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        // As mentioned, clip paths contain the area that will be affected by
        // our drawing operations. By default, drawing operations will be
        // visible on the whole width/height of the render context.

        // After this first `push_clip_path`, all drawing operations will be
        // constrained to the area of the first circle.
        ctx.push_clip_path(&clip_1.to_path(0.1));
        // Clip paths can be nested/stacked to arbitrary depths. By
        // nesting clip paths, the drawing area will be further reduced to
        // the _intersection_ of all clip paths that are currently in-place.
        // Thus, after this second `push_clip_path` call, only the pixels
        // that lie in the intersection of both circles will be painted.
        ctx.push_clip_path(&clip_2.to_path(0.1));
        ctx.set_paint(RED);
        // Even though the rectangle covers the whole viewport, only the parts
        // that lie in the intersection of both circles will be painted.
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));
        // By popping a clip path, the top clip-path on the element (in this case
        // `clip_2`) will be removed. Thus, only `clip_1` remains in-place.
        ctx.pop_clip_path();
        ctx.set_paint(BLUE.with_alpha(0.2));
        // This rectangle will only be constrained by the area of `clip_1`.
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        // This is optional. You don't strictly have to pop all clip paths
        // currently in-place before rasterizing to the pixmap.
        ctx.pop_clip_path();

        ctx.flush();

        save_pixmap(&ctx, "example_clipping1");
    }

    // Method 2: Isolated clipping using the `push_clip_layer` and `pop_layer`
    // methods:
    {
        // Overall, this method works exactly the same as the previous
        // one, just that the method calls are different. Instead of
        // `push_clip_path`, we have `push_clip_layer`, and instead of
        // `pop_clip_path`, we have `pop_clip_layer`.

        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        ctx.push_clip_layer(&clip_1.to_path(0.1));
        ctx.push_clip_layer(&clip_2.to_path(0.1));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));
        ctx.pop_layer();
        ctx.set_paint(BLUE.with_alpha(0.2));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        // Unlike the first method, THIS PART IS NOT OPTIONAL! Before
        // rasterizing, you need to make sure that all previously pushed layers
        // have been popped. Otherwise, the renderer will panic.
        ctx.pop_layer();

        ctx.flush();
        save_pixmap(&ctx, "example_clipping2");
    }

    // If you inspect the above results, you will see that they visually yield
    // the same result. So what is their difference and when should you use
    // which one? The answer should become clearer when explaining how they
    // differ conceptually.
    // When creating a clip path using `push_clip_path`, every subsequent drawing
    // operation will conceptually be stencil-masked through the intersection
    // of all currently active clip paths before being drawn onto the screen.
    // On the other hand, doing `push_clip_layer` will actually push a whole
    // new isolated layer, and once you call
    // `pop_layer`, the layer _as a whole_ will be clipped to the bounds of the
    // paths and composited back into the previous layer.
    //
    // Which one of these two methods you should use depend on the imaging model
    // you are trying to reflect. For example, in SVG, each group with a clip-path
    // automatically requires creating a new isolated layer. In this case, the
    // isolated clipping method fits the imaging model better. On the other hand,
    // in PDF for example, clip paths and layer isolation are two completely
    // separate concepts. Therefore, it makes much more sense to use the
    // `push_clip_path` method, since you don't want to introduce an isolated
    // layer each time a new clip path is added.
    //
    // Finally, it is also worth mentioning that according to your experiments,
    // non-isolated clipping is usually faster than isolated clipping, especially
    // on the CPU. Therefore, if you are still in doubt, it is recommended
    // that you simply use the non-isolated method. If necessary, you can easily
    // just mix the two different methods as well.
    //
    // Another small note: Clip paths can actually be emulated using alpha
    // masks (see the masking example), so strictly speaking you don't need
    // to use the specialized clipping methods to create clip paths. However,
    // the clipping methods are _much faster_ than masking, and you should
    // therefore always prefer using those over masking.
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_to_pixmap(&mut pixmap);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}

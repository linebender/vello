// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Using masks with Vello CPU.

use vello_cpu::color::palette::css::{BLUE, RED, WHITE};
use vello_cpu::kurbo::Rect;
use vello_cpu::{Mask, Pixmap, RenderContext};

const SIZE: u16 = 200;

fn main() {
    // Vello CPU supports applying luminance and alpha masks to your drawings.

    // First, we need to create our actual mask. There are multiple ways how
    // you can get to it, in our case we are going to draw our own custom mask.
    let mask = {
        // In this case, we are drawing the mask ourselves. Note that the
        // dimensions of the final mask need to match the dimensions of our
        // original render context!
        let mut mask_ctx = RenderContext::new(SIZE, SIZE);
        let mut pixmap = Pixmap::new(SIZE, SIZE);

        mask_ctx.set_paint(RED);
        mask_ctx.fill_rect(&Rect::new(30.0, 30.0, 170.0, 170.0));
        mask_ctx.flush();
        mask_ctx.render_to_pixmap(&mut pixmap);

        Mask::new_luminance(&pixmap)
    };

    // Create the main render context.
    let mut ctx = RenderContext::new(SIZE, SIZE);

    // Similarly to clip paths (see the clipping example), there are two
    // different ways of applying them:
    // The first method is by creating a new isolated layer where the mask
    // will be applied once the whole layer has been drawn and is composited
    // into the backdrop. The second method is by setting the mask in the
    // render context, in which case the mask will be applied to each shape
    // directly before being drawn. Which method you should use once again
    // depends on the imaging model you are reflecting.

    // Method 1: Non-isolated masking via `set_mask`.
    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64));

        // Once the mask is set, the mask will be applied to every path we
        // are drawing individually before compositing it into the background.
        ctx.set_mask(mask.clone());
        // We first apply the mask to the blue rectangle and then composite it.
        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(20.0, 20.0, 130.0, 130.0));
        // Now, we yet again first apply the mask to the red rectangle only and
        // then composite the result.
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(70.0, 70.0, 180.0, 180.0));
        // Use this method if you want to reset the mask currently in place.
        ctx.reset_mask();

        ctx.flush();
        save_pixmap(&ctx, "example_masking1");
    }

    ctx.reset();
    // Method 2: Isolated masking via `push_mask_layer`.
    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64));

        // Using this method, we first push a new isolated layer. Apart from that,
        // nothing happens so far.
        ctx.push_mask_layer(mask);
        // Here, the blue rectangle will be drawn first. Then, the red one is drawn
        // and subsequently composited on top of the blue one,
        // without any special handling.
        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(20.0, 20.0, 130.0, 130.0));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(70.0, 70.0, 180.0, 180.0));
        // Now, the whole layer is taken, the mask is applied to all of it
        // and then composited into the background.
        ctx.pop_layer();

        ctx.flush();
        save_pixmap(&ctx, "example_masking2");
    }

    // As can be seen, the visual result of the two methods can be different!
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_to_pixmap(&mut pixmap);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}

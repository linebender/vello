// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This example demonstrates the most basic usage of Vello CPU, including
//! fundamental features in 2D rendering like filling, stroking and applying
//! transforms.

use vello_cpu::color::palette::css::YELLOW;
use vello_cpu::kurbo::Affine;
use vello_cpu::{
    Level, Pixmap, RenderContext, RenderMode, RenderSettings,
    color::palette::css::{BLUE, GREEN, RED},
    kurbo::{Circle, Rect, Shape},
};

fn main() {
    // Vello CPU is a CPU-based 2D renderer. It takes drawing commands like
    // "fill a rectangle" or "stroke a triangle" as input and rasterizes
    // them into a bitmap with premultiplied RGBA pixels, which can then
    // be further processed (for example by converting to PNG or displaying
    // the result in a window).

    // We first need to define some basic render settings.
    let settings = RenderSettings {
        // The `level` field indicates what SIMD level should be used. In
        // the vast majority of cases, you should just use `Level::new` so that
        // Vello CPU automatically uses an appropriate level that is
        // available on the host system.
        //
        // There are very few reasons to override that value. One instance where
        // it might be useful to override is for example when using Vello CPU
        // to create reference images for test suites. In that case, you could
        // pass `Level::fallback`, which indicates to Vello CPU that no
        // platform-specific SIMD intrinsics should be used. This might be useful
        // because it reduces the possibility of slight pixel differences when
        // running on different platforms.
        level: Level::new(),
        // The number of additional threads that should be used for rendering.
        // This setting only has an effect when the `multi-threading` feature
        // is enabled. More threads _usually_ translates to better performance,
        // but also higher CPU usage. Multi-threading can be very effective
        // in situations where you need to continually redraw scenes (for example
        // when rendering GUIs), but is usually less useful for one-time
        // rendering operations. It should also be noted that the current
        // implementation of multi-threading is still not fully optimized and
        // might not work as well in certain workloads, so it's worth trying
        // yourself whether it actually leads to any speedups before using it.
        //
        // If you enabled the multi-threading feature and leave this as 0,
        // it has the exact same effect as rendering in single-threaded mode.
        // According to our experiments, 2-4 threads give the best results,
        // using 4+ threads might result in diminishing results, depending on
        // the workload.
        num_threads: 0,
        // Define whether the renderer should prioritize speed or quality
        // during rendering. Currently, the only difference is that
        // `OptimizeSpeed` will use u8/u16 for the rasterization
        // while `OptimizeQuality` uses f32. The former is much faster, but
        // has the disadvantage that the overall color accuracy might be slightly
        // worse due to quantization. Unless you really care about that, it is
        // highly recommended to use the `OptimizeSpeed` rendering mode.
        render_mode: RenderMode::OptimizeSpeed,
    };

    // Vello CPU embraces a slightly different paradigm than a lot of other 2D
    // renderers. Many 2D renderers use _immediate mode rendering_, where
    // you create a single `Pixmap` and then dispatch rendering commands into
    // it. In Vello CPU, there are two components instead.
    //
    // The first component is the `RenderContext`, which can be thought of as
    // a reusable buffer for dispatching rendering commands. The `reusable`
    // part is important: In situations where you are executing multiple
    // rendering passes at the same resolution, you should reset the context
    // and reuse it instead of always creating a new one, as will be shown below.
    // This is important because it allows Vello CPU to reuse existing memory
    // allocations, leading to better performance.
    //
    // The second component is then the `Pixmap`, which simply acts as a storage
    // for the raw RGBA pixels from the render context.

    // Let's start by creating a new render context with a certain
    // width and height in pixels, as well as our render settings.
    let mut ctx = RenderContext::new_with(100, 100, settings);

    // Vello CPU uses a Postscript-like API, where you can use methods like
    // `set_paint` or `set_stroke` to update an internal state, and then
    // dispatch commands to fill or stroke paths, using the current state.

    // Using the `set_paint` method, we can change what color our shapes should
    // be drawn with. You can use any RGBA color for that. Apart from that, you
    // can also paint your shapes using gradients or patterns (see the `paints`
    // example). In this case, we are setting the color to blue.
    ctx.set_paint(BLUE);
    // Now, we can dispatch a commands, for example to fill a rectangle with
    // the given dimensions.
    ctx.fill_path(&Rect::new(25.0, 25.0, 75.0, 75.0).to_path(0.1));

    // We can then update the color to a red with 50% opacity...
    ctx.set_paint(RED.with_alpha(0.5));
    // ...and draw a different rectangle that overlaps the previous one. You
    // can also use the `fill_rect` convenience method for that.
    ctx.fill_rect(&Rect::new(50.0, 50.0, 85.0, 85.0));

    ctx.set_paint(GREEN);
    // As mentioned, stroking is also supported. You can update the stroke
    // properties using the `set_stroke` method.
    ctx.stroke_path(&Circle::new((50.0, 50.0), 30.0).to_path(0.1));

    // Let's say that we have drawn everything we wanted to now. Next, it is
    // recommended that you call the `flush` method. In theory, this call
    // is only necessary when using multi-threaded rendering, to signal that
    // no more operations will be dispatched and the render context should be
    // synchronized. If you forget to call this during multi-threaded rendering,
    // the application will panic. In single-threaded rendering, nothing happens.
    //
    // However, it is highly recommended that you always call this.
    // This way, downstream consumers of your crate that might have externally
    // enabled Vello CPU's multi-threading feature won't run into panics when
    // running your code.
    ctx.flush();

    // Now the second step is to copy the results of the render context into the
    // pixmap. We do this by creating a new pixmap (or reusing an existing one).
    // Please note that the pixmap and the render context need to have the same
    // dimensions! Otherwise, the renderer will panic.
    let mut pixmap_1 = Pixmap::new(100, 100);
    // Now, simply extract the results from the render context into the
    // pixmap.
    ctx.render_to_pixmap(&mut pixmap_1);

    // Now you can do whatever you want with the pixmap, which provides raw
    // access to the premultiplied RGBA pixels of the image. If you have enabled
    // the `png` feature, you can convert it into a PNG image very easily and
    // then save it to disk.
    let png_1 = pixmap_1.into_png().unwrap();
    std::fs::write("example_basic1.png", png_1).unwrap();

    // If you have another scene you want to draw at the same resolution,
    // you can simply reuse the existing render context instead of creating
    // a new one.
    ctx.reset();

    // Vello CPU supports arbitrary affine transformations.
    ctx.set_transform(Affine::scale(3.0));
    ctx.set_paint(YELLOW);
    // The rectangle will now actually have the dimensions 60x60 since we
    // applied an affine transform that scales everything by 3x to the context.
    ctx.fill_rect(&Rect::new(0.0, 0.0, 20.0, 20.0));
    ctx.flush();

    // Once again, we render the results into a pixmap again. If you can,
    // you can just reuse existing pixmaps (assuming they have the correct
    // dimension), since all previous pixels in the pixmap will be
    // discarded. In our case, we need to create a new one since our call
    // to `into_png` consumed the pixmap.
    let mut pixmap_2 = Pixmap::new(100, 100);
    ctx.render_to_pixmap(&mut pixmap_2);
    let png_2 = pixmap_2.into_png().unwrap();
    std::fs::write("example_basic2.png", png_2).unwrap();
}

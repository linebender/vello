//! Visualize the intermediate stages of `vello_common` in an SVG.

use std::collections::HashSet;
use std::path::PathBuf;
use clap::Parser;
use svg::{Document, Node};
use svg::node::element::{Circle, Path, Rectangle};
use svg::node::element::path::Data;
use vello_common::flatten;
use vello_common::flatten::Line;
use vello_common::kurbo::{Affine, BezPath};
use vello_common::tile::{Tiles, TILE_HEIGHT, TILE_WIDTH};

fn main() {
    let args = Args::parse();

    let mut document = Document::new().set("viewBox", (-10, -10, args.width + 20, args.height + 20));

    let mut line_buf = vec![];
    let mut tiles = Tiles::new();

    flatten::fill(&args.path, Affine::IDENTITY, &mut line_buf);
    tiles.make_tiles(&line_buf);
    tiles.sort_tiles();

    draw_grid(&mut document, args.width, args.height);
    draw_line_segments(&mut document, &line_buf);
    draw_tile_areas(&mut document, &tiles);
    draw_tile_intersections(&mut document, &tiles);
    // draw_strips(&mut document, ctx.strip_buf(), ctx.alphas());
    // draw_wide_tiles(&mut document, ctx.wide_tiles(), ctx.alphas());

    svg::save(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug.svg"), &document).unwrap();
}
//
// fn draw_wide_tiles(document: &mut Document, wide_tiles: &[WideTile], alphas: &[u32]) {
//     for (t_i, tile) in wide_tiles.iter().enumerate() {
//         for cmd in &tile.cmds {
//             match cmd {
//                 Cmd::Fill(f) => {
//                     for i in 0..f.width {
//                         let Paint::Solid(c) = f.paint else { continue };
//                         let color = c.to_rgba8();
//
//                         for h in 0..STRIP_HEIGHT {
//                             let rect = Rectangle::new()
//                                 .set("x", f.x + i)
//                                 .set("y", t_i * STRIP_HEIGHT + h)
//                                 .set("width", 1)
//                                 .set("height", 1)
//                                 .set(
//                                     "fill",
//                                     format!("rgb({}, {}, {})", color.r, color.g, color.b),
//                                 )
//                                 .set("fill-opacity", color.a as f32 / 255.0);
//
//                             document.append(rect);
//                         }
//                     }
//                 }
//                 Cmd::Strip(s) => {
//                     for i in 0..s.width {
//                         let alpha = alphas[s.alpha_ix + i as usize];
//                         let entries = alpha.to_le_bytes();
//                         let Paint::Solid(c) = s.paint else { continue };
//                         let color = c.to_rgba8();
//
//                         for h in 0..STRIP_HEIGHT {
//                             let rect = Rectangle::new()
//                                 .set("x", s.x + i)
//                                 .set("y", t_i * STRIP_HEIGHT + h)
//                                 .set("width", 1)
//                                 .set("height", 1)
//                                 .set(
//                                     "fill",
//                                     format!("rgb({}, {}, {})", color.r, color.g, color.b),
//                                 )
//                                 .set(
//                                     "fill-opacity",
//                                     (entries[h] as f32 / 255.0) * (color.a as f32 / 255.0),
//                                 );
//
//                             document.append(rect);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
//

//
// fn draw_strips(document: &mut Document, strips: &[Strip], alphas: &[u32]) {
//     for i in 0..strips.len() {
//         let strip = &strips[i];
//         let x = strip.x();
//         let y = strip.strip_y();
//
//         let end = strips
//             .get(i + 1)
//             .map(|s| s.col)
//             .unwrap_or(alphas.len() as u32);
//
//         let width = end - strip.col;
//
//         let color = if strip.winding != 0 {
//             "red"
//         } else {
//             "limegreen"
//         };
//
//         let rect = Rectangle::new()
//             .set("x", x)
//             .set("y", y * STRIP_HEIGHT as u32)
//             .set("width", width)
//             .set("height", STRIP_HEIGHT)
//             .set("stroke", color)
//             .set("fill", color)
//             .set("fill-opacity", 0.4)
//             .set("stroke-opacity", 0.6)
//             .set("stroke-width", 0.2);
//
//         document.append(rect);
//     }
//
//     for i in 0..strips.len() {
//         let strip = &strips[i];
//         // Draw the points
//         let x = strip.x();
//         let y = strip.strip_y();
//
//         let end = strips
//             .get(i + 1)
//             .map(|s| s.col)
//             .unwrap_or(alphas.len() as u32);
//
//         let width = end - strip.col;
//
//         let color = if strip.winding != 0 {
//             "red"
//         } else {
//             "limegreen"
//         };
//
//         for i in 0..width {
//             let alpha = alphas[(i + strip.col) as usize];
//             let entries = alpha.to_le_bytes();
//
//             for h in 0..STRIP_HEIGHT {
//                 let rect = Rectangle::new()
//                     .set("x", x + i as i32)
//                     .set("y", y * STRIP_HEIGHT as u32 + h as u32)
//                     .set("width", 1)
//                     .set("height", 1)
//                     .set("fill", color)
//                     .set("fill-opacity", entries[h] as f32 / 255.0);
//
//                 document.append(rect);
//             }
//         }
//     }
// }
//

fn draw_grid(document: &mut Document, width: u16, height: u16) {
    let border_data = Data::new()
        .move_to((0, 0))
        .line_to((width, 0))
        .line_to((width, height))
        .line_to((0, height))
        .close();

    let border = Path::new()
        .set("stroke-width", 0.2)
        .set("fill", "none")
        .set("vectorEffect", "non-scaling-stroke")
        .set("stroke", "black")
        .set("d", border_data);

    let grid_line = |data: Data| {
        Path::new()
            .set("stroke", "grey")
            .set("stroke-opacity", 0.3)
            .set("stroke-width", 0.1)
            .set("vectorEffect", "non-scaling-stroke")
            .set("d", data)
    };

    for i in 1..height {
        let data = Data::new().move_to((0, i)).line_to((width, i));

        document.append(grid_line(data));
    }

    for i in 1..width {
        let data = Data::new().move_to((i, 0)).line_to((i, height));

        document.append(grid_line(data));
    }

    document.append(border);
}

fn draw_line_segments(document: &mut Document, line_buf: &[Line]) {
    let mut data = Data::new();

    let mut last = None;

    for line in line_buf {
        let first = (line.p0.x, line.p0.y);
        let second = (line.p1.x, line.p1.y);

        if Some(first) != last {
            data = data.move_to(first)
        }

        data = data.line_to(second);

        last = Some(second);
    }

    let border = Path::new()
        .set("stroke-width", 0.1)
        .set("stroke", "green")
        .set("fill", "yellow")
        .set("fill-opacity", 0.1)
        .set("d", data);

    document.append(border);
}

fn draw_tile_areas(document: &mut Document, tiles: &Tiles) {
    let mut seen = HashSet::new();

    for i in 0..tiles.len() {
        let tile = tiles.get(i);
        // Draw the points
        let x = tile.x * TILE_WIDTH as i32;
        let y = tile.y * TILE_HEIGHT as u16;

        if seen.contains(&(x, y)) {
            continue;
        }

        let color = "darkblue";

        let rect = Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", TILE_WIDTH)
            .set("height", TILE_HEIGHT)
            .set("fill", color)
            .set("stroke", color)
            .set("stroke-opacity", 1.0)
            .set("stroke-width", 0.2)
            .set("fill-opacity", 0.1);

        document.append(rect);

        seen.insert((x, y));
    }
}

fn draw_tile_intersections(document: &mut Document, tiles: &Tiles) {
    for i in 0..tiles.len() {
        let tile = tiles.get(i);

        let x = tile.x * TILE_WIDTH as i32;
        let y = tile.y * TILE_HEIGHT as u16;

        let p0 = tile.p0;
        let p1 = tile.p1;

        // Add a tiny offset so start and end point don't overlap completely.
        for p in [(p0, -0.05, "green"), (p1, 0.05, "purple")] {
            let circle = Circle::new()
                .set("cx", x as f32 + p.0.x + p.1)
                .set("cy", y as f32 + p.0.y)
                .set("r", 0.25)
                .set("fill", p.2)
                .set("fill-opacity", 0.5);

            document.append(circle);
        }
    }
}


#[derive(Parser, Debug)]
struct Args {
    /// The width of the viewport.
    #[arg(long, default_value_t = 50, value_parser = parse_dim)]
    pub width: u16,
    /// The height of the viewport.
    #[arg(long, default_value_t = 50, value_parser = parse_dim)]
    pub height: u16,
    /// The SVG path that should be drawn.
    #[arg(short, long, value_parser = parse_path)]
    pub path: BezPath,
    /// Whether the path should be filled (if false, it will be stroked).
    #[arg(short, long, default_value_t = true)]
    pub fill: bool,
    /// The stroke width for stroking operations.
    #[arg(short, long, default_value_t = 1.0)]
    pub stroke_width: f32,
}

fn parse_dim(val: &str) -> Result<u16, String> {
    let parsed = val.parse::<u16>().map_err(|_| "Width/Height must be a positive integer")?;
    if parsed > 500 {
        Err("The width/height cannot be larger than 500 (otherwise the SVG will be very slow).".to_string())
    } else {
        Ok(parsed)
    }
}

fn parse_path(val: &str) -> Result<BezPath, String> {
    BezPath::from_svg(val)
        .map_err(|_| "Failed to parse the SVG path".to_string())
}
// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Visualize the intermediate stages of `vello_common` in an SVG.

#![allow(
    clippy::cast_possible_truncation,
    reason = "this is only a debug tool, so we can ignore them"
)]

use clap::Parser;
use std::collections::HashSet;
use std::path;
use svg::node::element::path::Data;
use svg::node::element::{Line as SvgLine, Path, Rectangle};
use svg::{Document, Node};
use vello_common::coarse::{Cmd, MODE_CPU, Wide, WideTile};
use vello_common::color::palette::css::BLACK;
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, Line};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Stroke, StrokeCtx};
use vello_common::peniko::Fill;
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};
use vello_common::{flatten, strip};

fn main() {
    let args = Args::parse();

    let mut document =
        Document::new().set("viewBox", (-10, -10, args.width + 20, args.height + 20));

    let mut line_buf = vec![];
    let mut tiles = Tiles::new(Level::new());
    let mut strip_buf = vec![];
    let mut alpha_buf = vec![];
    let mut wide = Wide::<MODE_CPU>::new(args.width, args.height);

    let stages = &args.stages;

    // Not super efficient doing it this way, but it doesn't really matter.

    if stages.iter().any(|s| s.requires_flatten()) {
        if !args.stroke {
            flatten::fill(
                Level::new(),
                &args.path,
                Affine::IDENTITY,
                &mut line_buf,
                &mut FlattenCtx::default(),
            );
        } else {
            let stroke = Stroke {
                width: args.stroke_width as f64,
                join: Join::Bevel,
                start_cap: Cap::Butt,
                end_cap: Cap::Butt,
                ..Default::default()
            };
            flatten::stroke(
                Level::new(),
                &args.path,
                &stroke,
                Affine::IDENTITY,
                &mut line_buf,
                &mut FlattenCtx::default(),
                &mut StrokeCtx::default(),
            );
        }
    }

    if stages.iter().any(|s| s.requires_tiling()) {
        tiles.make_tiles(&line_buf, args.width, args.height);
        tiles.sort_tiles();
    }

    if stages.iter().any(|s| s.requires_strips()) {
        strip::render(
            Level::new(),
            &tiles,
            &mut strip_buf,
            &mut alpha_buf,
            args.fill_rule,
            None,
            &line_buf,
        );
    }

    if stages.iter().any(|s| s.requires_wide_tiles()) {
        wide.generate(&strip_buf, BLACK.into(), 0, None);
    }

    draw_grid(&mut document, args.width, args.height);

    if stages.contains(&Stage::LineSegments) {
        draw_line_segments(&mut document, &line_buf);
    }

    if stages.contains(&Stage::TileAreas) {
        draw_tile_areas(&mut document, &tiles);
    }

    if stages.contains(&Stage::StripAreas) {
        draw_strip_areas(&mut document, &strip_buf, &alpha_buf);
    }

    if stages.contains(&Stage::Strips) {
        draw_strips(&mut document, &strip_buf, &alpha_buf);
    }

    if stages.contains(&Stage::WideTiles) {
        draw_wide_tiles(&mut document, wide.tiles(), &alpha_buf);
    }

    let path = path::absolute("debug.svg").unwrap();
    eprintln!("Saved debug output to '{}'", path.display());
    svg::save(path, &document).unwrap();
}

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
    let svg_line = SvgLine::new()
        .set("stroke-width", 0.1)
        .set("fill", "none")
        .set("fill-opacity", 0.1);

    for line in line_buf {
        let dy = line.p1.y - line.p0.y;
        let color = if dy < 0. {
            // Lines oriented upwards add to winding.
            "green"
        } else if dy > 0. {
            // Lines oriented upwards subtract from winding.
            "red"
        } else {
            // Horizontal lines don't impact winding.
            "grey"
        };

        let svg_line = svg_line
            .clone()
            .set("x1", line.p0.x)
            .set("y1", line.p0.y)
            .set("x2", line.p1.x)
            .set("y2", line.p1.y)
            .set("stroke", color);
        document.append(svg_line);
    }
}

fn draw_tile_areas(document: &mut Document, tiles: &Tiles) {
    let mut seen = HashSet::new();

    for i in 0..tiles.len() {
        let tile = tiles.get(i);
        let x = tile.x * Tile::WIDTH;
        let y = tile.y * Tile::HEIGHT;

        if seen.contains(&(x, y)) {
            continue;
        }

        let color = "darkblue";

        let rect = Rectangle::new()
            .set("x", x)
            .set("y", y)
            .set("width", Tile::WIDTH)
            .set("height", Tile::HEIGHT)
            .set("fill", color)
            .set("stroke", color)
            .set("stroke-opacity", 1.0)
            .set("stroke-width", 0.2)
            .set("fill-opacity", 0.1);

        document.append(rect);

        seen.insert((x, y));
    }
}

fn draw_strip_areas(document: &mut Document, strips: &[Strip], alphas: &[u8]) {
    for i in 0..strips.len() {
        let strip = &strips[i];
        let x = strip.x;
        let y = strip.strip_y();

        let end = strips
            .get(i + 1)
            .map(|s| s.alpha_idx() / u32::from(Tile::HEIGHT))
            .unwrap_or(alphas.len() as u32);

        let width = end - strip.alpha_idx() / u32::from(Tile::HEIGHT);

        // TODO: Account for even-odd?
        let color = if strip.fill_gap() { "red" } else { "limegreen" };

        let rect = Rectangle::new()
            .set("x", x)
            .set("y", y * Tile::HEIGHT)
            .set("width", width)
            .set("height", Tile::HEIGHT)
            .set("stroke", color)
            .set("fill", color)
            .set("fill-opacity", 0.4)
            .set("stroke-opacity", 0.6)
            .set("stroke-width", 0.2);

        document.append(rect);
    }
}

fn draw_strips(document: &mut Document, strips: &[Strip], alphas: &[u8]) {
    for s in 0..strips.len() {
        let strip = &strips[s];

        let end = strips
            .get(s + 1)
            .map(|st| st.alpha_idx() / u32::from(Tile::HEIGHT))
            .unwrap_or(alphas.len() as u32);

        let width = u16::try_from(end - strip.alpha_idx() / u32::from(Tile::HEIGHT)).unwrap();

        // TODO: Account for even-odd?
        let color = if strip.fill_gap() { "red" } else { "limegreen" };

        for x in 0..width {
            for y in 0..Tile::HEIGHT {
                let alpha = alphas[strip.alpha_idx() as usize
                    + usize::from(x) * usize::from(Tile::HEIGHT)
                    + usize::from(y)];
                let rect = Rectangle::new()
                    .set("x", strip.x + x)
                    .set("y", strip.y + y)
                    .set("width", 1)
                    .set("height", 1)
                    .set("fill", color)
                    .set("fill-opacity", alpha as f32 / 255.0);

                document.append(rect);
            }
        }
    }
}

fn draw_wide_tiles(document: &mut Document, wide_tiles: &[WideTile], alphas: &[u8]) {
    // TODO: account for multiple wide tiles per row.
    for (tile_idx, tile) in wide_tiles.iter().enumerate() {
        for cmd in &tile.cmds {
            match cmd {
                Cmd::Fill(f) => {
                    for x in 0..f.width {
                        for y in 0..Tile::HEIGHT {
                            let rect = Rectangle::new()
                                .set("x", f.x + x)
                                .set("y", tile_idx * usize::from(Tile::HEIGHT) + usize::from(y))
                                .set("width", 1)
                                .set("height", 1)
                                .set("fill", "blue");

                            document.append(rect);
                        }
                    }
                }
                Cmd::AlphaFill(s) => {
                    for x in 0..s.width {
                        for y in 0..Tile::HEIGHT {
                            let alpha = alphas[s.alpha_idx
                                + usize::from(x) * usize::from(Tile::HEIGHT)
                                + usize::from(y)];

                            let rect = Rectangle::new()
                                .set("x", s.x + x)
                                .set("y", tile_idx * usize::from(Tile::HEIGHT) + usize::from(y))
                                .set("width", 1)
                                .set("height", 1)
                                .set("fill", "yellow")
                                .set("fill-opacity", alpha as f32 / 255.0);

                            document.append(rect);
                        }
                    }
                }
                _ => {
                    unimplemented!("unsupported command: {:?}", cmd);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Draw the flattened lines of the path.
    LineSegments,
    /// Draw the tile areas covered by the path.
    TileAreas,
    /// Draw the stripped areas.
    StripAreas,
    /// Draw the strips with their alpha masks.
    Strips,
    /// Draw the wide tiles.
    WideTiles,
}

impl Stage {
    fn requires_flatten(&self) -> bool {
        matches!(self, Self::LineSegments) || self.requires_tiling()
    }
    fn requires_tiling(&self) -> bool {
        matches!(self, Self::TileAreas) || self.requires_strips()
    }

    fn requires_strips(&self) -> bool {
        matches!(self, Self::StripAreas)
            || matches!(self, Self::Strips)
            || self.requires_wide_tiles()
    }

    fn requires_wide_tiles(&self) -> bool {
        matches!(self, Self::WideTiles)
    }
}

impl std::str::FromStr for Stage {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.to_lowercase().as_str() {
            "ls" | "line_segments" => Ok(Self::LineSegments),
            "ta" | "tile_areas" => Ok(Self::TileAreas),
            "sa" | "strip_areas" => Ok(Self::StripAreas),
            "s" | "strips" => Ok(Self::Strips),
            "wt" | "wide_tiles" => Ok(Self::WideTiles),
            _ => Err(format!(
                "invalid stage: {input}. Expected one of `line_segments`, `tile_areas`, `tile_intersections`, `strip_areas`, `strips`, or `wide_tiles`, or their acronym"
            )),
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
    /// Whether the path should be stroked (if false, it will be filled).
    #[arg(short, long, default_value_t = false)]
    pub stroke: bool,
    /// The stroke width for stroking operations.
    #[arg(long, default_value_t = 1.0)]
    pub stroke_width: f32,
    /// The fill rule used for fill operations.
    #[arg(long, default_value = "nonzero", value_parser = parse_fill_rule)]
    pub fill_rule: Fill,
    /// The stages of the pipeline that should be included in the SVG.
    #[arg(long, num_args = 1.., value_delimiter = ',', default_value = "ls,ta,ti,sa,s", value_parser = parse_stages)]
    pub stages: Vec<Stage>,
}

fn parse_stages(val: &str) -> Result<Stage, String> {
    val.parse::<Stage>()
}

fn parse_dim(val: &str) -> Result<u16, String> {
    let parsed = val
        .parse::<u16>()
        .map_err(|_| "width/height must be a positive integer")?;
    if parsed > 500 {
        Err(
            "the width/height cannot be larger than 500 (otherwise the SVG will be very slow)."
                .to_string(),
        )
    } else {
        Ok(parsed)
    }
}

fn parse_path(val: &str) -> Result<BezPath, String> {
    BezPath::from_svg(val).map_err(|_| "failed to parse the SVG path".to_string())
}

fn parse_fill_rule(val: &str) -> Result<Fill, String> {
    match val {
        "nonzero" => Ok(Fill::NonZero),
        "evenodd" => Ok(Fill::EvenOdd),
        _ => Err(format!(
            "unsupported fill rule: {val}. Expected one of `nonzero` or `evenodd`"
        )),
    }
}

//! Reading a dumped SVG file.

use crate::DATA_PATH;
use vello_common::flatten::Line;
use vello_common::kurbo::{Affine, BezPath, Stroke};
use vello_common::peniko::Fill;
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};
use vello_common::{flatten, strip};

pub struct PathContainer {
    fills: Vec<BezPath>,
    strokes: Vec<BezPath>,
}

impl PathContainer {
    pub fn from_data_file(name: &str) -> Self {
        let path = DATA_PATH.join(format!("{}.txt", name));
        let raw_data = std::fs::read(path).unwrap();

        let data = std::str::from_utf8(&raw_data).unwrap();
        let mut splitted = data.split("STROKES");
        let fill_str = splitted.next().unwrap();
        let stroke_str = splitted.next().unwrap();

        let fills = fill_str
            .split("\n")
            .map(|s| BezPath::from_svg(s).unwrap())
            .collect::<Vec<_>>();
        let strokes = stroke_str
            .split("\n")
            .map(|s| BezPath::from_svg(s).unwrap())
            .collect::<Vec<_>>();

        Self { fills, strokes }
    }

    /// Get the raw paths used for filling operations.
    pub fn fills(&self) -> &[BezPath] {
        &self.fills
    }

    /// Get the raw paths used for stroking operations.
    pub fn strokes(&self) -> &[BezPath] {
        &self.strokes
    }

    /// Get the raw flattened lines of both fills and strokes.
    ///
    /// A stroke width of 2 is assumed.
    pub fn lines(&self) -> Vec<Line> {
        let mut line_buf = vec![];

        for path in &self.fills {
            flatten::fill(path, Affine::default(), &mut line_buf);
        }

        let stroke = Stroke {
            // Obviously not all strokes have that width, but it should be good enough
            // for benchmarking.
            width: 2.0,
            ..Default::default()
        };

        for path in &self.strokes {
            flatten::stroke(path, &stroke, Affine::default(), &mut line_buf)
        }

        line_buf
    }

    /// Get the unsorted tiles.
    pub fn unsorted_tiles(&self, tile_width: u16, tile_height: u16) -> Tiles {
        let mut tiles = Tiles::new();
        let lines = self.lines();
        tiles.make_tiles(&lines, tile_width, tile_height);

        tiles
    }

    /// Get the sorted tiles.
    pub fn sorted_tiles(&self, tile_width: u16, tile_height: u16) -> Tiles {
        let mut tiles = self.unsorted_tiles(tile_width, tile_height);
        tiles.sort_tiles();

        tiles
    }

    /// Get the alpha buffer and rendered strips.
    pub fn strips(&self, tile_width: u16, tile_height: u16) -> (Vec<u8>, Vec<Strip>) {
        let mut strip_buf = vec![];
        let mut alpha_buf = vec![];
        let lines = self.lines();
        let tiles = self.sorted_tiles(tile_width, tile_height);

        strip::render(
            &tiles,
            &mut strip_buf,
            &mut alpha_buf,
            Fill::NonZero,
            &lines,
        );

        (alpha_buf, strip_buf)
    }
}

// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    n_drawobj: u32,
    n_path: u32,

    // offsets within scene buffer (in u32 units)
    // Note: this is a difference from piet-gpu, which is in bytes
    pathtag_base: u32,
    pathdata_base: u32,

    drawtag_base: u32,
    drawdata_base: u32,

    transform_base: u32,
    linewidth_base: u32,
}

// Geometry of tiles and bins

let TILE_WIDTH = 16u;
let TILE_HEIGHT = 16u;
// Number of tiles per bin
let N_TILE_X = 16u;
let N_TILE_Y = 16u;
//let N_TILE = N_TILE_X * N_TILE_Y;
let N_TILE = 256u;

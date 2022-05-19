// Copyright 2022 The piet-gpu authors.
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

/// Defines the color mixing function for a blend operation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum Mix {
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    Hue = 12,
    Saturation = 13,
    Color = 14,
    Luminosity = 15,
    // Clip is the same as normal, but doesn't always push a blend group.
    Clip = 128,
}

/// Defines the layer composition function for a blend operation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum Compose {
    Clear = 0,
    Copy = 1,
    Dest = 2,
    SrcOver = 3,
    DestOver = 4,
    SrcIn = 5,
    DestIn = 6,
    SrcOut = 7,
    DestOut = 8,
    SrcAtop = 9,
    DestAtop = 10,
    Xor = 11,
    Plus = 12,
    PlusLighter = 13,
}

/// Blend mode consisting of mixing and composition functions.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Blend {
    pub mix: Mix,
    pub compose: Compose,
}

impl Blend {
    pub fn new(mix: Mix, compose: Compose) -> Self {
        Self { mix, compose }
    }

    pub fn pack(&self) -> u32 {
        (self.mix as u32) << 8 | self.compose as u32
    }
}

impl Default for Blend {
    fn default() -> Self {
        Self {
            mix: Mix::Clip,
            compose: Compose::SrcOver,
        }
    }
}

impl From<Mix> for Blend {
    fn from(mix: Mix) -> Self {
        Self {
            mix,
            compose: Compose::SrcOver,
        }
    }
}

impl From<Compose> for Blend {
    fn from(compose: Compose) -> Self {
        Self {
            mix: Mix::Normal,
            compose,
        }
    }
}

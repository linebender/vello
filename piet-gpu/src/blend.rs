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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum BlendMode {
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
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub enum CompositionMode {
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
    PlusDarker = 13,
    PlusLighter = 14,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Blend {
    pub mode: BlendMode,
    pub composition_mode: CompositionMode,
}

impl Blend {
    pub fn new(mode: BlendMode, composition_mode: CompositionMode) -> Self {
        Self { mode, composition_mode }
    }

    pub(crate) fn pack(&self) -> u32 {
        (self.mode as u32) << 8 | self.composition_mode as u32
    }
}

impl Default for Blend {
    fn default() -> Self {
        Self {
            mode: BlendMode::Normal,
            composition_mode: CompositionMode::SrcOver,
        }
    }
}

impl From<BlendMode> for Blend {
    fn from(mode: BlendMode) -> Self {
        Self {
            mode,
            composition_mode: CompositionMode::SrcOver,
        }
    }
}

impl From<CompositionMode> for Blend {
    fn from(mode: CompositionMode) -> Self {
        Self {
            mode: BlendMode::Normal,
            composition_mode: mode,
        }
    }
}

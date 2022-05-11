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

use std::result::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Format {
    A8,
    Rgba8,
}

impl Format {
    pub fn data_size(self, width: u32, height: u32) -> Option<usize> {
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|size| {
                size.checked_mul(match self {
                    Self::A8 => 1,
                    Self::Rgba8 => 4,
                })
            })
    }
}

#[derive(Clone, Debug)]
pub struct Image(Arc<Inner>);

#[derive(Clone, Debug)]
struct Inner {
    id: u64,
    format: Format,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Image {
    pub fn new(
        format: Format,
        width: u32,
        height: u32,
        mut data: Vec<u8>,
    ) -> Result<Self, DataSizeError> {
        let data_size = format.data_size(width, height).ok_or(DataSizeError)?;
        if data.len() < data_size {
            return Err(DataSizeError);
        }
        data.truncate(data_size);
        static ID: AtomicU64 = AtomicU64::new(1);
        Ok(Self(Arc::new(Inner {
            id: ID.fetch_add(1, Ordering::Relaxed),
            format,
            width,
            height,
            data,
        })))
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn format(&self) -> Format {
        self.0.format
    }

    pub fn width(&self) -> u32 {
        self.0.width
    }

    pub fn height(&self) -> u32 {
        self.0.height
    }

    pub fn data(&self) -> &[u8] {
        &self.0.data
    }
}

#[derive(Clone, Debug)]
pub struct DataSizeError;

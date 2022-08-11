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

/// Image data resource.
#[derive(Clone, Debug)]
pub struct Image(Arc<Inner>);

#[derive(Clone, Debug)]
struct Inner {
    id: u64,
    width: u32,
    height: u32,
    data: Arc<[u8]>,
}

impl Image {
    pub fn new(
        width: u32,
        height: u32,
        data: impl Into<Arc<[u8]>>,
    ) -> Result<Self, ImageDataSizeError> {
        let data_size = width
            .checked_mul(height)
            .and_then(|x| x.checked_mul(4))
            .ok_or(ImageDataSizeError)? as usize;
        let data = data.into();
        if data.len() < data_size {
            return Err(ImageDataSizeError);
        }
        static ID: AtomicU64 = AtomicU64::new(1);
        Ok(Self(Arc::new(Inner {
            id: ID.fetch_add(1, Ordering::Relaxed),
            width,
            height,
            data,
        })))
    }

    pub fn id(&self) -> u64 {
        self.0.id
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

/// Error returned when image data size is not sufficient for the specified
/// dimensions.
#[derive(Clone, Debug)]
pub struct ImageDataSizeError;

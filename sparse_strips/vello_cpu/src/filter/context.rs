// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use vello_common::pixmap::Pixmap;

#[derive(Debug, Default)]
pub(crate) struct FilterContext {
    /// The rendered pixmaps for each filter layer.
    layers: Vec<Option<Pixmap>>,
    scratch: ScratchBuffer,
}

impl FilterContext {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            scratch: ScratchBuffer::new(),
        }
    }

    pub(crate) fn scratch(&mut self) -> &mut ScratchBuffer {
        &mut self.scratch
    }

    pub(crate) fn set_layer(&mut self, layer_id: usize, pixmap: Pixmap) {
        if layer_id >= self.layers.len() {
            self.layers.resize_with(layer_id + 1, || None);
        }
        self.layers[layer_id] = Some(pixmap);
    }

    pub(crate) fn layer(&self, layer_id: usize) -> Option<&Pixmap> {
        self.layers.get(layer_id).and_then(Option::as_ref)
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScratchBuffer {
    scratch_buffer: Option<Pixmap>,
}

impl ScratchBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_scratch_buffer(&mut self, width: u16, height: u16) -> &mut Pixmap {
        match &mut self.scratch_buffer {
            None => {
                self.scratch_buffer = Some(Pixmap::new(width, height));
            }
            Some(buf) if buf.width() < width || buf.height() < height => {
                buf.resize(width, height);
            }
            Some(_) => {}
        }

        self.scratch_buffer.as_mut().unwrap()
    }
}

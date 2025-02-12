// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub(crate) struct Buffer<T> {
    run: bool,
    // State used by the CPU pass
    cpu_write_count: u16,
    cpu_read_count: u16,
    remaining_writes_cpu: u16,
    remaining_reads_cpu: u16,
    cpu_content: Vec<T>,

    /// The buffer used to stage content into the GPU.
    /// The buffer is mapped for writing.
    ///
    /// Will be `None` if the buffer is never calculated on the CPU.
    staging_buffer: Option<wgpu::Buffer>,
    /// Whether the content has been written into the staging buffer.
    /// Only needed if `cpu_write_count` > 0
    staging_written: bool,

    gpu_buffer: wgpu::Buffer,
    // Whether content has been copied from.
    gpu_written: bool,
}

impl<T> Buffer<T> {
    pub(crate) fn read(&mut self) -> &[T] {
        if self.run {
            self.remaining_reads_cpu -= 1;
            &self.cpu_content
        } else {
            self.cpu_read_count += 1;
            &[]
        }
    }
    pub(crate) fn write(&mut self) -> &mut [T] {
        if self.run {
            self.remaining_writes_cpu -= 1;
            // If the buffer is being written to, but will never be used again on the CPU,
            // it must be needed on the GPU (otherwise, why would we write to it?).
            // Therefore, we can write directly into the staging buffer.
            // Technically, this breaks down if we run a GPU pipeline only partially (e.g. for debugging)
            // but that case is rare enough that we don't worry about it.
            if self.remaining_reads_cpu == 0 && self.remaining_writes_cpu == 0 {
                // self.staging_written = true;
                // return self
                //     .staging_buffer
                //     .slice(..)
                //     .get_mapped_range_mut()
                //     .deref_mut();
            }
            &mut self.cpu_content
        } else {
            self.cpu_write_count += 1;
            &mut []
        }
    }
    pub(crate) fn read_write(&mut self) -> &mut [T] {
        if self.run {
            self.remaining_reads_cpu -= 1;
            self.remaining_writes_cpu -= 1;
            &mut self.cpu_content
        } else {
            self.cpu_write_count += 1;
            self.cpu_read_count += 1;
            &mut []
        }
    }
}

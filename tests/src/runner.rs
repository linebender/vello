// Copyright 2021 The piet-gpu authors.
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

//! Test runner intended to make it easy to write tests.

use std::ops::RangeBounds;

use bytemuck::Pod;
use piet_gpu_hal::{
    BackendType, BufReadGuard, BufWriteGuard, Buffer, BufferUsage, CmdBuf, ComputePass,
    ComputePassDescriptor, Instance, InstanceFlags, QueryPool, Session,
};

pub struct Runner {
    #[allow(unused)]
    instance: Instance,
    pub session: Session,
    cmd_buf_pool: Vec<CmdBuf>,
}

/// A wrapper around command buffers.
pub struct Commands {
    pub cmd_buf: CmdBuf,
    query_pool: QueryPool,
}

/// Buffer for both uploading and downloading
pub struct BufStage {
    pub stage_buf: Buffer,
    pub dev_buf: Buffer,
}

impl Runner {
    pub unsafe fn new(flags: InstanceFlags) -> Runner {
        let (instance, _) = Instance::new(None, flags).unwrap();
        let device = instance.device(None).unwrap();
        let session = Session::new(device);
        let cmd_buf_pool = Vec::new();
        Runner {
            instance,
            session,
            cmd_buf_pool,
        }
    }

    pub unsafe fn commands(&mut self) -> Commands {
        let mut cmd_buf = self
            .cmd_buf_pool
            .pop()
            .unwrap_or_else(|| self.session.cmd_buf().unwrap());
        cmd_buf.begin();
        // TODO: also pool these. But we might sort by size, as
        // we might not always be doing two.
        let query_pool = self.session.create_query_pool(2).unwrap();
        cmd_buf.reset_query_pool(&query_pool);
        Commands {
            cmd_buf,
            query_pool,
        }
    }

    pub unsafe fn submit(&mut self, commands: Commands) -> f64 {
        let mut cmd_buf = commands.cmd_buf;
        let query_pool = commands.query_pool;
        cmd_buf.finish_timestamps(&query_pool);
        cmd_buf.host_barrier();
        cmd_buf.finish();
        let submitted = self.session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
        self.cmd_buf_pool.extend(submitted.wait().unwrap());
        let timestamps = self.session.fetch_query_pool(&query_pool).unwrap();
        timestamps.get(0).copied().unwrap_or_default()
    }

    #[allow(unused)]
    pub fn buf_up(&self, size: u64) -> BufStage {
        let stage_buf = self
            .session
            .create_buffer(size, BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC)
            .unwrap();
        let dev_buf = self
            .session
            .create_buffer(size, BufferUsage::COPY_DST | BufferUsage::STORAGE)
            .unwrap();
        BufStage { stage_buf, dev_buf }
    }

    /// Create a buffer for download (readback).
    ///
    /// The `usage` parameter need not include COPY_SRC and STORAGE.
    pub fn buf_down(&self, size: u64, usage: BufferUsage) -> BufStage {
        let stage_buf = self
            .session
            .create_buffer(size, BufferUsage::MAP_READ | BufferUsage::COPY_DST)
            .unwrap();
        let dev_buf = self
            .session
            .create_buffer(size, usage | BufferUsage::COPY_SRC | BufferUsage::STORAGE)
            .unwrap();
        BufStage { stage_buf, dev_buf }
    }

    pub fn backend_type(&self) -> BackendType {
        self.session.backend_type()
    }
}

impl Commands {
    /// Start a compute pass with timer queries.
    pub unsafe fn compute_pass(&mut self, start_query: u32, end_query: u32) -> ComputePass {
        self.cmd_buf
            .begin_compute_pass(&ComputePassDescriptor::timer(
                &self.query_pool,
                start_query,
                end_query,
            ))
    }

    pub unsafe fn upload(&mut self, buf: &BufStage) {
        self.cmd_buf.copy_buffer(&buf.stage_buf, &buf.dev_buf);
    }

    pub unsafe fn download(&mut self, buf: &BufStage) {
        self.cmd_buf.copy_buffer(&buf.dev_buf, &buf.stage_buf);
    }
}

impl BufStage {
    pub unsafe fn read(&self, dst: &mut Vec<impl Pod>) {
        self.stage_buf.read(dst).unwrap()
    }

    pub unsafe fn map_read<'a>(&'a self, range: impl RangeBounds<usize>) -> BufReadGuard<'a> {
        self.stage_buf.map_read(range).unwrap()
    }

    pub unsafe fn map_write<'a>(&'a mut self, range: impl RangeBounds<usize>) -> BufWriteGuard {
        self.stage_buf.map_write(range).unwrap()
    }
}

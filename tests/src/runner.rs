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

use piet_gpu_hal::{
    BackendType, Buffer, BufferUsage, CmdBuf, Instance, InstanceFlags, PlainData, QueryPool,
    Session,
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

/// Buffer for uploading data to GPU.
#[allow(unused)]
pub struct BufUp {
    pub stage_buf: Buffer,
    pub dev_buf: Buffer,
}

/// Buffer for downloading data from GPU.
pub struct BufDown {
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
        timestamps[0]
    }

    #[allow(unused)]
    pub fn buf_up(&self, size: u64) -> BufUp {
        let stage_buf = self
            .session
            .create_buffer(size, BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC)
            .unwrap();
        let dev_buf = self
            .session
            .create_buffer(size, BufferUsage::COPY_DST | BufferUsage::STORAGE)
            .unwrap();
        BufUp { stage_buf, dev_buf }
    }

    pub fn buf_down(&self, size: u64) -> BufDown {
        let stage_buf = self
            .session
            .create_buffer(size, BufferUsage::MAP_READ | BufferUsage::COPY_DST)
            .unwrap();
        // Note: the COPY_DST isn't needed in all use cases, but I don't think
        // making this tighter would help.
        let dev_buf = self
            .session
            .create_buffer(
                size,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST | BufferUsage::STORAGE,
            )
            .unwrap();
        BufDown { stage_buf, dev_buf }
    }

    pub fn backend_type(&self) -> BackendType {
        self.session.backend_type()
    }
}

impl Commands {
    pub unsafe fn write_timestamp(&mut self, query: u32) {
        self.cmd_buf.write_timestamp(&self.query_pool, query);
    }

    #[allow(unused)]
    pub unsafe fn upload(&mut self, buf: &BufUp) {
        self.cmd_buf.copy_buffer(&buf.stage_buf, &buf.dev_buf);
    }

    pub unsafe fn download(&mut self, buf: &BufDown) {
        self.cmd_buf.copy_buffer(&buf.dev_buf, &buf.stage_buf);
    }
}

impl BufDown {
    pub unsafe fn read(&self, dst: &mut Vec<impl PlainData>) {
        self.stage_buf.read(dst).unwrap()
    }
}

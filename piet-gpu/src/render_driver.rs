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

use bytemuck::Pod;
use piet_gpu_hal::{CmdBuf, Error, Image, QueryPool, Semaphore, Session, SubmittedCmdBuf};

use crate::{EncodedSceneRef, MemoryHeader, PietGpuRenderContext, Renderer, SceneStats};

/// Additional logic for sequencing rendering operations, specifically
/// for handling failure and reallocation.
///
/// It may be this shouldn't be a separate object from Renderer.
pub struct RenderDriver {
    frames: Vec<RenderFrame>,
    renderer: Renderer,
    buf_ix: usize,
    /// The index of a pending fine rasterization submission.
    pending: Option<usize>,
}

pub struct TargetState<'a> {
    pub cmd_buf: &'a mut CmdBuf,
    pub image: &'a Image,
}

#[derive(Default, Debug)]
pub struct TimingStats {
    coarse: Vec<f64>,
    fine: Vec<f64>,
}

struct RenderFrame {
    cmd_buf: CmdBufState,
    coarse_query_pool: QueryPool,
    fine_query_pool: QueryPool,
    timing_stats: TimingStats,
}

enum CmdBufState {
    Start,
    Submitted(SubmittedCmdBuf),
    Ready(CmdBuf),
}

impl RenderDriver {
    /// Create new render driver.
    ///
    /// Should probably be fallible.
    ///
    /// We can get n from the renderer as well.
    pub fn new(session: &Session, n: usize, renderer: Renderer) -> RenderDriver {
        let frames = (0..n)
            .map(|_| {
                // Maybe should allocate here so it doesn't happen on first frame?
                let cmd_buf = CmdBufState::default();
                let coarse_query_pool =
                    session.create_query_pool(Renderer::COARSE_QUERY_POOL_SIZE)?;
                let fine_query_pool = session.create_query_pool(Renderer::FINE_QUERY_POOL_SIZE)?;
                Ok(RenderFrame {
                    cmd_buf,
                    coarse_query_pool,
                    fine_query_pool,
                    timing_stats: TimingStats::default(),
                })
            })
            .collect::<Result<_, Error>>()
            .unwrap();
        RenderDriver {
            frames,
            renderer,
            buf_ix: 0,
            pending: None,
        }
    }

    pub fn upload_render_ctx(
        &mut self,
        session: &Session,
        render_ctx: &mut PietGpuRenderContext,
    ) -> Result<(), Error> {
        let stats = render_ctx.stats();
        self.ensure_scene_buffers(session, &stats)?;
        self.renderer.upload_render_ctx(render_ctx, self.buf_ix)
    }

    pub fn upload_scene<T: Copy + Pod>(
        &mut self,
        session: &Session,
        scene: &EncodedSceneRef<T>,
    ) -> Result<(), Error> {
        let stats = scene.stats();
        self.ensure_scene_buffers(session, &stats)?;
        self.renderer.upload_scene(scene, self.buf_ix)
    }

    fn ensure_scene_buffers(&mut self, session: &Session, stats: &SceneStats) -> Result<(), Error> {
        let scene_size = stats.scene_size();
        unsafe {
            self.renderer
                .realloc_scene_if_needed(session, scene_size as u64, self.buf_ix)?;
        }
        let memory_size = self.renderer.memory_size(&stats);
        // TODO: better estimate of additional memory needed
        // Note: if we were to cover the worst-case binning output, we could make the
        // binning stage infallible and cut checking logic. It also may not be a bad
        // estimate for the rest.
        let estimated_needed = memory_size as u64 + (1 << 20);
        if estimated_needed > self.renderer.memory_buf_size() {
            if let Some(pending) = self.pending.take() {
                // There might be a fine rasterization task that binds the memory buffer
                // still in flight.
                self.frames[pending].cmd_buf.wait();
            }
            unsafe {
                self.renderer.realloc_memory(session, estimated_needed)?;
            }
        }
        Ok(())
    }

    /// Run one try of the coarse rendering pipeline.
    pub(crate) fn try_run_coarse(&mut self, session: &Session) -> Result<MemoryHeader, Error> {
        let frame = &mut self.frames[self.buf_ix];
        let cmd_buf = frame.cmd_buf.cmd_buf(session)?;
        unsafe {
            cmd_buf.begin();
            // TODO: probably want to return query results as well
            self.renderer
                .record_coarse(cmd_buf, &frame.coarse_query_pool, self.buf_ix);
            self.renderer.record_readback(cmd_buf);
            let cmd_buf = frame.cmd_buf.cmd_buf(session)?;
            cmd_buf.finish_timestamps(&frame.coarse_query_pool);
            cmd_buf.host_barrier();
            cmd_buf.finish();
            frame.cmd_buf.submit(session, &[], &[])?;
            frame.cmd_buf.wait();
            frame.timing_stats.coarse = session.fetch_query_pool(&frame.coarse_query_pool)?;
            let mut result = Vec::new();
            // TODO: consider read method for single POD value
            self.renderer.memory_buf_readback.read(&mut result)?;
            Ok(result[0])
        }
    }

    /// Run the coarse render pipeline, ensuring enough memory for intermediate buffers.
    pub fn run_coarse(&mut self, session: &Session) -> Result<(), Error> {
        loop {
            let mem_header = self.try_run_coarse(session)?;
            //println!("{:?}", mem_header);
            if mem_header.mem_error == 0 {
                let blend_needed = mem_header.blend_offset as u64;
                if blend_needed > self.renderer.blend_size() {
                    unsafe {
                        self.renderer.realloc_blend(session, blend_needed)?;
                    }
                }
                return Ok(());
            }
            // Not enough memory, reallocate and retry.
            // TODO: be smarter (multiplier for early stages)
            let mem_size = mem_header.mem_offset + 4096;
            // Safety rationalization: no command buffers containing the buffer are
            // in flight.
            unsafe {
                self.renderer.realloc_memory(session, mem_size.into())?;
                self.renderer.upload_config(self.buf_ix)?;
            }
        }
    }

    /// Record the fine rasterizer, leaving the command buffer open.
    pub fn record_fine(&mut self, session: &Session) -> Result<TargetState, Error> {
        let frame = &mut self.frames[self.buf_ix];
        let cmd_buf = frame.cmd_buf.cmd_buf(session)?;
        unsafe {
            self.renderer
                .record_fine(cmd_buf, &frame.fine_query_pool, 0);
        }
        let image = &self.renderer.image_dev;
        Ok(TargetState { cmd_buf, image })
    }

    /// Submit the current command buffer.
    pub fn submit(
        &mut self,
        session: &Session,
        wait_semaphores: &[&Semaphore],
        signal_semaphores: &[&Semaphore],
    ) -> Result<(), Error> {
        let frame = &mut self.frames[self.buf_ix];
        let cmd_buf = frame.cmd_buf.cmd_buf(session)?;
        unsafe {
            cmd_buf.finish_timestamps(&frame.fine_query_pool);
            cmd_buf.host_barrier();
            cmd_buf.finish();
            frame
                .cmd_buf
                .submit(session, wait_semaphores, signal_semaphores)?
        }
        self.pending = Some(self.buf_ix);
        Ok(())
    }

    unsafe fn wait_frame(&mut self, session: &Session, buf_ix: usize) {
        let frame = &mut self.frames[buf_ix];
        frame.cmd_buf.wait();
        if let Ok(stats) = session.fetch_query_pool(&frame.fine_query_pool) {
            frame.timing_stats.fine = stats;
        }
        if self.pending == Some(buf_ix) {
            self.pending = None;
        }
    }

    pub unsafe fn wait(&mut self, session: &Session) {
        self.wait_frame(session, self.buf_ix);
    }

    /// Move to the next buffer.
    pub fn next_buffer(&mut self) {
        self.buf_ix = (self.buf_ix + 1) % self.frames.len()
    }

    pub unsafe fn get_timing_stats(&mut self, session: &Session, buf_ix: usize) -> &TimingStats {
        self.wait_frame(session, buf_ix);
        &self.frames[buf_ix].timing_stats
    }

    pub fn wait_all(&mut self, session: &Session) {
        for buf_ix in 0..self.frames.len() {
            unsafe {
                self.wait_frame(session, buf_ix);
            }
        }
    }
}

impl Default for CmdBufState {
    fn default() -> Self {
        CmdBufState::Start
    }
}

impl CmdBufState {
    /// Get a command buffer suitable for recording.
    ///
    /// If the command buffer is submitted, wait.
    fn cmd_buf(&mut self, session: &Session) -> Result<&mut CmdBuf, Error> {
        if let CmdBufState::Ready(cmd_buf) = self {
            return Ok(cmd_buf);
        }
        if let CmdBufState::Submitted(submitted) = std::mem::take(self) {
            if let Ok(Some(cmd_buf)) = submitted.wait() {
                *self = CmdBufState::Ready(cmd_buf);
            }
        }
        if matches!(self, CmdBufState::Start) {
            *self = CmdBufState::Ready(session.cmd_buf()?);
        }
        if let CmdBufState::Ready(cmd_buf) = self {
            Ok(cmd_buf)
        } else {
            unreachable!()
        }
    }

    unsafe fn submit(
        &mut self,
        session: &Session,
        wait_semaphores: &[&Semaphore],
        signal_semaphores: &[&Semaphore],
    ) -> Result<(), Error> {
        if let CmdBufState::Ready(cmd_buf) = std::mem::take(self) {
            let submitted = session.run_cmd_buf(cmd_buf, wait_semaphores, signal_semaphores)?;
            *self = CmdBufState::Submitted(submitted);
            Ok(())
        } else {
            Err("Tried to submit CmdBufState not in ready state".into())
        }
    }

    fn wait(&mut self) {
        if matches!(self, CmdBufState::Submitted(_)) {
            if let CmdBufState::Submitted(submitted) = std::mem::take(self) {
                if let Ok(Some(cmd_buf)) = submitted.wait() {
                    *self = CmdBufState::Ready(cmd_buf);
                }
            }
        }
    }
}

impl TimingStats {
    pub fn print_summary(&self) {
        let ts = &self.coarse;
        println!("Element time: {:.3}ms", ts[0] * 1e3);
        println!("Clip + bin + tile time: {:.3}ms", (ts[2] - ts[1]) * 1e3);
        println!("Coarse path time: {:.3}ms", (ts[4] - ts[2]) * 1e3);
        println!("Backdrop time: {:.3}ms", (ts[6] - ts[5]) * 1e3);
        println!("Coarse raster kernel time: {:.3}ms", (ts[8] - ts[7]) * 1e3);
        println!("Fine kernel time: {:.3}ms", self.fine[0] * 1e3);
    }

    pub fn short_summary(&self) -> String {
        let ts = &self.coarse;
        let el = ts[0] * 1e3;
        let cl = (ts[2] - ts[1]) * 1e3;
        let cp = (ts[4] - ts[3]) * 1e3;
        let bd = (ts[6] - ts[5]) * 1e3;
        let cr = (ts[8] - ts[7]) * 1e3;
        let fr = self.fine[0] * 1e3;
        let total = el + cl + cp + bd + cr + fr;
        format!(
            "{:.3}ms :: el:{:.3}ms|cl:{:.3}ms|cp:{:.3}ms|bd:{:.3}ms|cr:{:.3}ms|fr:{:.3}ms",
            total, el, cl, cp, bd, cr, fr
        )
    }
}

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

//! The path stage (includes substages).

use piet_gpu_hal::{
    include_shader, BindType, Buffer, BufferUsage, ComputePass, DescriptorSet, Pipeline, Session,
};

pub struct PathCode {
    reduce_pipeline: Pipeline,
    tag_root_pipeline: Pipeline,
    clear_pipeline: Pipeline,
    pathseg_pipeline: Pipeline,
}

pub struct PathStage {
    tag_root_buf: Buffer,
    tag_root_ds: DescriptorSet,
}

pub struct PathBinding {
    reduce_ds: DescriptorSet,
    clear_ds: DescriptorSet,
    path_ds: DescriptorSet,
}

const REDUCE_WG: u32 = 128;
const REDUCE_N_ROWS: u32 = 2;
const REDUCE_PART_SIZE: u32 = REDUCE_WG * REDUCE_N_ROWS;

const ROOT_WG: u32 = 256;
const ROOT_N_ROWS: u32 = 8;
const ROOT_PART_SIZE: u32 = ROOT_WG * ROOT_N_ROWS;

const SCAN_WG: u32 = 256;
const SCAN_N_ROWS: u32 = 4;
const SCAN_PART_SIZE: u32 = SCAN_WG * SCAN_N_ROWS;

pub const PATHSEG_PART_SIZE: u32 = SCAN_PART_SIZE;

const CLEAR_WG: u32 = 256;

impl PathCode {
    pub unsafe fn new(session: &Session) -> PathCode {
        let reduce_code = include_shader!(session, "../../shader/gen/pathtag_reduce");
        let reduce_pipeline = session
            .create_compute_pipeline(
                reduce_code,
                &[
                    BindType::Buffer,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::Buffer,
                ],
            )
            .unwrap();
        let tag_root_code = include_shader!(session, "../../shader/gen/pathtag_root");
        let tag_root_pipeline = session
            .create_compute_pipeline(tag_root_code, &[BindType::Buffer])
            .unwrap();
        let clear_code = include_shader!(session, "../../shader/gen/bbox_clear");
        let clear_pipeline = session
            .create_compute_pipeline(clear_code, &[BindType::Buffer, BindType::BufReadOnly])
            .unwrap();
        let pathseg_code = include_shader!(session, "../../shader/gen/pathseg");
        let pathseg_pipeline = session
            .create_compute_pipeline(
                pathseg_code,
                &[
                    BindType::Buffer,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                    BindType::BufReadOnly,
                ],
            )
            .unwrap();
        PathCode {
            reduce_pipeline,
            tag_root_pipeline,
            clear_pipeline,
            pathseg_pipeline,
        }
    }
}

impl PathStage {
    pub unsafe fn new(session: &Session, code: &PathCode) -> PathStage {
        let tag_root_buf_size = (ROOT_PART_SIZE * 20) as u64;
        let tag_root_buf = session
            .create_buffer(tag_root_buf_size, BufferUsage::STORAGE)
            .unwrap();
        let tag_root_ds = session
            .create_simple_descriptor_set(&code.tag_root_pipeline, &[&tag_root_buf])
            .unwrap();
        PathStage {
            tag_root_buf,
            tag_root_ds,
        }
    }

    pub unsafe fn bind(
        &self,
        session: &Session,
        code: &PathCode,
        config_buf: &Buffer,
        scene_buf: &Buffer,
        memory_buf: &Buffer,
    ) -> PathBinding {
        let reduce_ds = session
            .create_simple_descriptor_set(
                &code.reduce_pipeline,
                &[memory_buf, config_buf, scene_buf, &self.tag_root_buf],
            )
            .unwrap();
        let clear_ds = session
            .create_simple_descriptor_set(&code.clear_pipeline, &[memory_buf, config_buf])
            .unwrap();
        let path_ds = session
            .create_simple_descriptor_set(
                &code.pathseg_pipeline,
                &[memory_buf, config_buf, scene_buf, &self.tag_root_buf],
            )
            .unwrap();
        PathBinding {
            reduce_ds,
            clear_ds,
            path_ds,
        }
    }

    /// Record the path stage.
    ///
    /// Note: no barrier is needed for transform output, we have a barrier before
    /// those are consumed. Result is written without barrier.
    pub unsafe fn record(
        &self,
        pass: &mut ComputePass,
        code: &PathCode,
        binding: &PathBinding,
        n_paths: u32,
        n_tags: u32,
    ) {
        if n_tags > ROOT_PART_SIZE * SCAN_PART_SIZE {
            println!(
                "number of pathsegs exceeded {} > {}",
                n_tags,
                ROOT_PART_SIZE * SCAN_PART_SIZE
            );
        }

        // Number of tags consumed in a tag reduce workgroup
        let reduce_part_tags = REDUCE_PART_SIZE * 4;
        let n_wg_tag_reduce = (n_tags + reduce_part_tags - 1) / reduce_part_tags;
        if n_wg_tag_reduce > 1 {
            pass.dispatch(
                &code.reduce_pipeline,
                &binding.reduce_ds,
                (n_wg_tag_reduce, 1, 1),
                (REDUCE_WG, 1, 1),
            );
            // I think we can skip root if n_wg_tag_reduce == 2
            pass.memory_barrier();
            pass.dispatch(
                &code.tag_root_pipeline,
                &self.tag_root_ds,
                (1, 1, 1),
                (ROOT_WG, 1, 1),
            );
            // No barrier needed here; clear doesn't depend on path tags
        }
        let n_wg_clear = (n_paths + CLEAR_WG - 1) / CLEAR_WG;
        pass.dispatch(
            &code.clear_pipeline,
            &binding.clear_ds,
            (n_wg_clear, 1, 1),
            (CLEAR_WG, 1, 1),
        );
        pass.memory_barrier();
        let n_wg_pathseg = (n_tags + SCAN_PART_SIZE - 1) / SCAN_PART_SIZE;
        pass.dispatch(
            &code.pathseg_pipeline,
            &binding.path_ds,
            (n_wg_pathseg, 1, 1),
            (SCAN_WG, 1, 1),
        );
    }
}

impl PathBinding {
    pub unsafe fn rebind_memory(&mut self, session: &Session, memory: &Buffer) {
        session.update_buffer_descriptor(&mut self.reduce_ds, 0, memory);
        session.update_buffer_descriptor(&mut self.clear_ds, 0, memory);
        session.update_buffer_descriptor(&mut self.path_ds, 0, memory);
    }

    pub unsafe fn rebind_scene(&mut self, session: &Session, scene: &Buffer) {
        session.update_buffer_descriptor(&mut self.reduce_ds, 2, scene);
        session.update_buffer_descriptor(&mut self.path_ds, 2, scene);
    }
}

pub struct PathEncoder<'a> {
    tag_stream: &'a mut Vec<u8>,
    // If we're never going to use the i16 encoding, it might be
    // slightly faster to store this as Vec<u32>, we'd get aligned
    // stores on ARM etc.
    pathseg_stream: &'a mut Vec<u8>,
    first_pt: [f32; 2],
    state: State,
    n_pathseg: u32,
}

#[derive(PartialEq)]
enum State {
    Start,
    MoveTo,
    NonemptySubpath,
}

impl<'a> PathEncoder<'a> {
    pub fn new(tags: &'a mut Vec<u8>, pathsegs: &'a mut Vec<u8>) -> PathEncoder<'a> {
        PathEncoder {
            tag_stream: tags,
            pathseg_stream: pathsegs,
            first_pt: [0.0, 0.0],
            state: State::Start,
            n_pathseg: 0,
        }
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.first_pt = buf;
        if self.state == State::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if self.state == State::NonemptySubpath {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.pathseg_stream.extend_from_slice(bytes);
        self.state = State::MoveTo;
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.state == State::Start {
            // should warn or error
            return;
        }
        let buf = [x, y];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(9);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        if self.state == State::Start {
            return;
        }
        let buf = [x1, y1, x2, y2];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(10);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn cubic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        if self.state == State::Start {
            return;
        }
        let buf = [x1, y1, x2, y2, x3, y3];
        let bytes = bytemuck::bytes_of(&buf);
        self.pathseg_stream.extend_from_slice(bytes);
        self.tag_stream.push(11);
        self.state = State::NonemptySubpath;
        self.n_pathseg += 1;
    }

    pub fn close_path(&mut self) {
        match self.state {
            State::Start => return,
            State::MoveTo => {
                let new_len = self.pathseg_stream.len() - 8;
                self.pathseg_stream.truncate(new_len);
                self.state = State::Start;
                return;
            }
            State::NonemptySubpath => (),
        }
        let len = self.pathseg_stream.len();
        if len < 8 {
            // can't happen
            return;
        }
        let first_bytes = bytemuck::bytes_of(&self.first_pt);
        if &self.pathseg_stream[len - 8..len] != first_bytes {
            self.pathseg_stream.extend_from_slice(first_bytes);
            self.tag_stream.push(13);
            self.n_pathseg += 1;
        } else {
            if let Some(tag) = self.tag_stream.last_mut() {
                *tag |= 4;
            }
        }
        self.state = State::Start;
    }

    fn finish(&mut self) {
        if self.state == State::MoveTo {
            let new_len = self.pathseg_stream.len() - 8;
            self.pathseg_stream.truncate(new_len);
        }
        if let Some(tag) = self.tag_stream.last_mut() {
            *tag |= 4;
        }
    }

    /// Finish encoding a path.
    ///
    /// Encode this after encoding path segments.
    pub fn path(&mut self) {
        self.finish();
        // maybe don't encode if path is empty? might throw off sync though
        self.tag_stream.push(0x10);
    }

    /// Get the number of path segments.
    ///
    /// This is the number of path segments that will be written by the
    /// path stage; use this for allocating the output buffer.
    ///
    /// Also note: it takes `self` for lifetime reasons.
    pub fn n_pathseg(self) -> u32 {
        self.n_pathseg
    }
}

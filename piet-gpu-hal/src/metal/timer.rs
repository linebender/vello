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

//! Support for timer queries.
//!
//! Likely some of this should be upstreamed into metal-rs.

use std::{ffi::CStr, ptr::null_mut};

use cocoa_foundation::{
    base::id,
    foundation::{NSRange, NSUInteger},
};
use metal::{DeviceRef, MTLStorageMode};
use objc::{class, msg_send, sel, sel_impl};

pub struct CounterSampleBuffer {
    id: id,
    count: u64,
}

pub struct CounterSet {
    id: id,
}

#[derive(Default)]
pub struct TimeCalibration {
    pub cpu_start_ts: u64,
    pub gpu_start_ts: u64,
    pub cpu_end_ts: u64,
    pub gpu_end_ts: u64,
}

impl Drop for CounterSampleBuffer {
    fn drop(&mut self) {
        unsafe { msg_send![self.id, release] }
    }
}

impl Clone for CounterSampleBuffer {
    fn clone(&self) -> CounterSampleBuffer {
        unsafe {
            CounterSampleBuffer {
                id: msg_send![self.id, retain],
                count: self.count,
            }
        }
    }
}

impl CounterSampleBuffer {
    pub fn id(&self) -> id {
        self.id
    }
}

impl Drop for CounterSet {
    fn drop(&mut self) {
        unsafe { msg_send![self.id, release] }
    }
}

impl CounterSet {
    pub fn get_timer_counter_set(device: &DeviceRef) -> Option<CounterSet> {
        unsafe {
            // TODO: version check
            let sets: id = msg_send!(device, counterSets);
            let count: NSUInteger = msg_send![sets, count];
            for i in 0..count {
                let set: id = msg_send![sets, objectAtIndex: i];
                let name: id = msg_send![set, name];
                let name_cstr = CStr::from_ptr(msg_send![name, UTF8String]);
                if name_cstr.to_bytes() == b"timestamp" {
                    return Some(CounterSet { id: set });
                }
            }
            None
        }
    }
}

// copied from metal-rs; should be in common utilities maybe?
fn nsstring_as_str(nsstr: &objc::runtime::Object) -> &str {
    let bytes = unsafe {
        let bytes: *const std::os::raw::c_char = msg_send![nsstr, UTF8String];
        bytes as *const u8
    };
    let len: NSUInteger = unsafe { msg_send![nsstr, length] };
    unsafe {
        let bytes = std::slice::from_raw_parts(bytes, len as usize);
        std::str::from_utf8(bytes).unwrap()
    }
}

impl CounterSampleBuffer {
    pub fn new(
        device: &DeviceRef,
        count: u64,
        counter_set: &CounterSet,
    ) -> Option<CounterSampleBuffer> {
        unsafe {
            let desc_cls = class!(MTLCounterSampleBufferDescriptor);
            let descriptor: id = msg_send![desc_cls, alloc];
            let _: id = msg_send![descriptor, init];
            let count = count as NSUInteger;
            let () = msg_send![descriptor, setSampleCount: count];
            let () = msg_send![descriptor, setCounterSet: counter_set.id];
            let () = msg_send![
                descriptor,
                setStorageMode: MTLStorageMode::Shared as NSUInteger
            ];
            let mut error: id = null_mut();
            let buf: id = msg_send![device, newCounterSampleBufferWithDescriptor: descriptor error: &mut error];
            let () = msg_send![descriptor, release];
            if !error.is_null() {
                let description = msg_send![error, localizedDescription];
                println!(
                    "error allocating sample buffer, code = {}",
                    nsstring_as_str(description)
                );
                let () = msg_send![error, release];
                return None;
            }
            Some(CounterSampleBuffer { id: buf, count })
        }
    }

    // Read the timestamps.
    //
    // Safety: the lifetime of the returned slice is wrong, it's actually autoreleased.
    pub unsafe fn resolve(&self) -> &[u64] {
        let range = NSRange::new(0, self.count);
        let data: id = msg_send![self.id, resolveCounterRange: range];
        if data.is_null() {
            &[]
        } else {
            let bytes: *const u64 = msg_send![data, bytes];
            std::slice::from_raw_parts(bytes, self.count as usize)
        }
    }
}

impl TimeCalibration {
    /// Convert GPU timestamp into CPU time base.
    ///
    /// See https://developer.apple.com/documentation/metal/performance_tuning/correlating_cpu_and_gpu_timestamps
    pub fn correlate(&self, raw_ts: u64) -> f64 {
        let delta_cpu = self.cpu_end_ts - self.cpu_start_ts;
        let delta_gpu = self.gpu_end_ts - self.gpu_start_ts;
        let adj_ts = if delta_gpu > 0 {
            let scale = delta_cpu as f64 / delta_gpu as f64;
            self.cpu_start_ts as f64 + (raw_ts as f64 - self.gpu_start_ts as f64) * scale
        } else {
            // Default is ns on Apple Silicon; on other hardware this will be wrong
            raw_ts as f64
        };
        adj_ts * 1e-9
    }
}

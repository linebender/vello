// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Performance measurement functionality for GPU operations.

use std::sync::Mutex;
use wgpu::{Buffer, BufferUsages, CommandEncoder, Device, QuerySet, QueryType, Queue};

/// Handles performance measurement using GPU timestamp queries
pub struct PerfMeasurement {
    pub timestamp_query_set: QuerySet,
    pub timestamp_buffer: Buffer,
    pub readback_buffer: Mutex<Option<Buffer>>,
    pub duration_measurements: Mutex<Vec<f64>>,
    pub max_measurements: usize,
}

impl PerfMeasurement {
    pub fn new(device: &Device) -> Self {
        let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp QuerySet"),
            ty: QueryType::Timestamp,
            // Start and end timestamps
            count: 2,
        });

        let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Buffer"),
            // 2 timestamps * 8 bytes
            size: 16,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            timestamp_query_set,
            timestamp_buffer,
            readback_buffer: Mutex::new(None),
            duration_measurements: Mutex::new(Vec::new()),
            max_measurements: 100,
        }
    }

    /// Write timestamp in a command encoder
    pub fn write_timestamp(&self, encoder: &mut CommandEncoder, query_index: u32) {
        encoder.write_timestamp(&self.timestamp_query_set, query_index);
    }

    /// Resolve timestamp queries to buffer
    pub fn resolve_timestamp_queries(&self, encoder: &mut CommandEncoder, device: &Device) {
        encoder.resolve_query_set(&self.timestamp_query_set, 0..2, &self.timestamp_buffer, 0);

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Readback Buffer"),
            // 2 timestamps * 8 bytes
            size: 16,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&self.timestamp_buffer, 0, &readback_buffer, 0, 16);

        if let Ok(mut opt_buffer) = self.readback_buffer.lock() {
            *opt_buffer = Some(readback_buffer);
        }
    }

    /// Map and read the timestamp buffer
    pub fn map_and_read_timestamp_buffer(&self, device: &Device, queue: &Queue) {
        if let Ok(mut opt_buffer) = self.readback_buffer.lock() {
            if let Some(buffer) = opt_buffer.take() {
                // Map the buffer and read the timestamps
                let slice = buffer.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();

                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });

                // Wait for the mapping to complete
                device.poll(wgpu::Maintain::Wait);

                if let Ok(Ok(())) = receiver.recv() {
                    let data = slice.get_mapped_range();
                    let timestamps: &[u64] = bytemuck::cast_slice(&data);

                    if timestamps.len() >= 2 {
                        let start = timestamps[0];
                        let end = timestamps[1];

                        if end > start {
                            let period_ns = queue.get_timestamp_period() as f64;
                            let duration_ns = (end - start) as f64 * period_ns;
                            let duration_ms = duration_ns / 1_000_000.0;

                            if let Ok(mut measurements) = self.duration_measurements.lock() {
                                measurements.push(duration_ms);
                                if measurements.len() >= self.max_measurements {
                                    let count = measurements.len();
                                    let sum: f64 = measurements.iter().sum();
                                    let avg = sum / count as f64;

                                    let min =
                                        measurements.iter().copied().fold(f64::INFINITY, f64::min);
                                    let max = measurements
                                        .iter()
                                        .copied()
                                        .fold(f64::NEG_INFINITY, f64::max);

                                    println!("===== GPU TIMESTAMP MEASUREMENT STATISTICS =====");
                                    println!("Samples: {}", count);
                                    println!("Avg    : {:.3} ms", avg);
                                    println!("Min    : {:.3} ms", min);
                                    println!("Max    : {:.3} ms", max);
                                    println!("================================================");

                                    measurements.clear();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

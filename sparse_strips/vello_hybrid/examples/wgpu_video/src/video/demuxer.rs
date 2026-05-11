// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `AVAssetReader`-based demuxer pulling compressed video packets from an MP4 / MOV.
//!
//! Trimmed port of `lightspeed_av/backend/apple/avasset_reader.rs`:
//! - **No audio**: only video tracks are enumerated and read.
//! - **No seek**: read-once-forward.
//! - **No marker-buffer skipping fast path**: re-pulls until a buffer with a data buffer
//!   appears (same correctness, less code).
//! - **No round-robin** across tracks: this example consumes a single video track.
//!
//! The original handles a number of edge cases we deliberately ignore here so the FFI
//! surface stays as small as possible while still using the typed `objc2-*` bindings.

use std::path::Path;
use std::ptr::NonNull;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_av_foundation::{
    AVAsset, AVAssetReader, AVAssetReaderOutput, AVAssetReaderTrackOutput, AVAssetTrack,
    AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_foundation::{CFRetained, Type};
use objc2_core_media::{CMFormatDescription, CMSampleBuffer, CMTimeFlags};
use objc2_foundation::{NSArray, NSString, NSURL};

use super::error::AvError;
use super::frame::FourCC;
use super::packet::{Packet, TrackInfo};

pub(crate) struct Demuxer {
    /// Owning reference to the reader so its outputs stay alive for the demuxer's
    /// lifetime.
    _reader: Retained<AVAssetReader>,
    track: TrackState,
    track_info: TrackInfo,
}

// SAFETY: `Demuxer` is sole-owner-driven (`&mut`-only API). The objc2 `Retained`
// fields hold +1 retains on Objective-C objects whose retain/release is atomic.
// `AVAssetReader`'s contract is "one owner drives `copyNextSampleBuffer` per output",
// which our `&mut`-only API enforces.
unsafe impl Send for Demuxer {}

struct TrackState {
    output: Retained<AVAssetReaderTrackOutput>,
    finished: bool,
}

impl Demuxer {
    /// Open an MP4 / MOV file and prepare to read packets from its first video track.
    pub(crate) fn open(path: &Path) -> Result<Self, AvError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| AvError::Source(format!("non-UTF8 path: {}", path.display())))?;
        let ns_path = NSString::from_str(path_str);
        // `+[NSURL fileURLWithPath:]` — typed binding.
        let url: Retained<NSURL> = NSURL::fileURLWithPath(&ns_path);

        // SAFETY: typed binding from objc2-av-foundation; `options` is nil.
        let asset: Retained<AVURLAsset> =
            unsafe { AVURLAsset::URLAssetWithURL_options(&url, None) };

        let descriptor = first_video_track(&asset)?;
        let asset_super: &AVAsset = &asset;

        // SAFETY: typed binding; `asset` is valid. `init…_error` lifts NSError
        // out-parameter into a Result.
        let reader: Retained<AVAssetReader> =
            unsafe { AVAssetReader::initWithAsset_error(AVAssetReader::alloc(), asset_super) }
                .map_err(|err| {
                    let msg = err.localizedDescription().to_string();
                    log::warn!("AVAssetReader rejected container: {msg}");
                    AvError::UnsupportedContainer("container not opened by AVAsset")
                })?;

        // SAFETY: typed binding. `outputSettings: nil` keeps the source codec untouched
        // (no decode-time CSC, no scaling).
        let output: Retained<AVAssetReaderTrackOutput> = unsafe {
            AVAssetReaderTrackOutput::initWithTrack_outputSettings(
                AVAssetReaderTrackOutput::alloc(),
                &descriptor.track,
                None,
            )
        };

        // SAFETY: typed bindings on a fresh reader and output.
        if !unsafe { reader.canAddOutput(&output) } {
            return Err(AvError::backend(
                format!(
                    "AVAssetReader rejected output for track {}",
                    descriptor.info.track_id
                ),
                0,
            ));
        }
        // SAFETY: `canAddOutput` just confirmed it's acceptable.
        unsafe { reader.addOutput(&output) };

        // SAFETY: typed accessor.
        if !unsafe { reader.startReading() } {
            return Err(AvError::backend(
                "AVAssetReader startReading returned NO",
                0,
            ));
        }

        Ok(Self {
            _reader: reader,
            track: TrackState {
                output,
                finished: false,
            },
            track_info: descriptor.info,
        })
    }

    pub(crate) fn track_info(&self) -> &TrackInfo {
        &self.track_info
    }

    /// Pull the next packet from the video track, or `Ok(None)` at end-of-stream.
    ///
    /// Skips AVFoundation's "marker" `CMSampleBuffer`s (no data buffer) silently — same
    /// behaviour as the source pipeline.
    pub(crate) fn read_packet(&mut self) -> Result<Option<Packet>, AvError> {
        if self.track.finished {
            return Ok(None);
        }
        loop {
            // `copyNextSampleBuffer` is defined on the parent class
            // `AVAssetReaderOutput`; explicitly deref-cast to call it.
            let parent: &AVAssetReaderOutput = &self.track.output;
            // SAFETY: typed binding on a valid output.
            let sample_buffer: Option<Retained<CMSampleBuffer>> =
                unsafe { parent.copyNextSampleBuffer() };
            let Some(sample_buffer) = sample_buffer else {
                self.track.finished = true;
                return Ok(None);
            };
            let sample_buffer: CFRetained<CMSampleBuffer> = retained_to_cf_retained(sample_buffer);

            // SAFETY: typed accessor on a valid CMSampleBuffer.
            if unsafe { sample_buffer.data_buffer() }.is_none() {
                continue;
            }

            // SAFETY: typed accessor.
            let pts = unsafe { sample_buffer.presentation_time_stamp() };
            let pts_ns = if pts.flags.contains(CMTimeFlags::Valid) && pts.timescale > 0 {
                let value = pts.value as i128;
                let scale = pts.timescale as i128;
                ((value * 1_000_000_000) / scale) as i64
            } else {
                0
            };

            return Ok(Some(Packet {
                track_id: self.track_info.track_id,
                pts_ns,
                sample_buffer,
            }));
        }
    }
}

struct VideoTrackDescriptor {
    track: Retained<AVAssetTrack>,
    info: TrackInfo,
}

/// Find the first video track on `asset` and return its underlying `AVAssetTrack`
/// alongside the cross-platform `TrackInfo` shape.
fn first_video_track(asset: &AVURLAsset) -> Result<VideoTrackDescriptor, AvError> {
    let asset_super: &AVAsset = asset;
    // SAFETY: typed accessor; returns a retained NSArray of AVAssetTracks.
    let tracks_array: Retained<NSArray<AVAssetTrack>> = unsafe { asset_super.tracks() };
    let count = tracks_array.len();

    // SAFETY: framework-owned static CFString constant.
    let media_type_video: &NSString =
        unsafe { AVMediaTypeVideo }.expect("AVMediaTypeVideo unavailable");

    for i in 0..count {
        let track: Retained<AVAssetTrack> = tracks_array.objectAtIndex(i);
        // SAFETY: typed accessor.
        let media_type_str: Retained<NSString> = unsafe { track.mediaType() };
        if &*media_type_str != media_type_video {
            continue;
        }

        // SAFETY: typed accessor.
        let format_descriptions: Retained<NSArray> = unsafe { track.formatDescriptions() };
        let format_desc: Option<CFRetained<CMFormatDescription>> =
            format_descriptions.firstObject().map(|obj| {
                let raw_ptr: *const AnyObject = &*obj;
                let cm_ptr = raw_ptr.cast::<CMFormatDescription>().cast_mut();
                // SAFETY: cm_ptr is non-null (came from a valid Retained) and points to
                // a CMFormatDescription kept alive by the array.
                unsafe { CFRetained::retain(NonNull::new_unchecked(cm_ptr)) }
            });

        let codec = if let Some(fd) = format_desc.as_deref() {
            // SAFETY: typed accessor on a valid CMFormatDescription.
            FourCC(unsafe { fd.media_sub_type() })
        } else {
            FourCC(0)
        };

        return Ok(VideoTrackDescriptor {
            track,
            info: TrackInfo {
                track_id: i as u32,
                codec,
            },
        });
    }

    Err(AvError::TrackNotFound(0))
}

/// Convert an objc2 `Retained<T>` into an objc2-core-foundation `CFRetained<T>`.
///
/// `CMSampleBuffer` is a CFType, but `AVAssetReaderTrackOutput::copyNextSampleBuffer`
/// hands it back in objc2's `Retained`. We unwrap and rewrap without changing the
/// retain count.
fn retained_to_cf_retained<T: Type + objc2::Message>(retained: Retained<T>) -> CFRetained<T> {
    let raw = Retained::into_raw(retained);
    // SAFETY: `raw` is a non-null retained pointer; we transfer its retain into the
    // returned `CFRetained` without rebalancing.
    unsafe { CFRetained::from_raw(NonNull::new_unchecked(raw)) }
}

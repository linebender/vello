// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demuxer built on AVAssetReader that reads compressed video packets from MP4 / MOV files.
//!
//! Kept minimal on purpose: it reads only the first video track, once from start to end,
//! with no audio and no seeking.

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
use objc2_core_media::{CMFormatDescription, CMSampleBuffer};
use objc2_foundation::{NSArray, NSString, NSURL};

use super::error::AvError;
use super::frame::FourCC;

/// The video track the demuxer reads.
pub(crate) struct TrackInfo {
    pub(crate) codec: FourCC,
    /// Codec configuration (such as H.264 SPS / PPS) the decoder session is created from.
    pub(crate) format_description: CFRetained<CMFormatDescription>,
}

pub(crate) struct Demuxer {
    /// Kept alive because `output` reads through it.
    _reader: Retained<AVAssetReader>,
    output: Retained<AVAssetReaderTrackOutput>,
    finished: bool,
}

impl Demuxer {
    /// Opens an MP4 / MOV file for reading packets from its first video track.
    pub(crate) fn open(path: &Path) -> Result<(Self, TrackInfo), AvError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| AvError::Source(format!("non-UTF8 path: {}", path.display())))?;
        let url: Retained<NSURL> = NSURL::fileURLWithPath(&NSString::from_str(path_str));

        // SAFETY: `url` is valid, and `options` may be nil.
        let asset: Retained<AVURLAsset> =
            unsafe { AVURLAsset::URLAssetWithURL_options(&url, None) };
        let (track, track_info) = first_video_track(&asset)?;

        // SAFETY: `asset` is valid; failures come back as `Err`.
        let reader: Retained<AVAssetReader> =
            unsafe { AVAssetReader::initWithAsset_error(AVAssetReader::alloc(), &asset) }.map_err(
                |err| {
                    let msg = err.localizedDescription().to_string();
                    log::warn!("AVAssetReader rejected container: {msg}");
                    AvError::UnsupportedContainer("container not opened by AVAsset")
                },
            )?;

        // SAFETY: the track is valid. Nil output settings pass samples through still
        // compressed.
        let output: Retained<AVAssetReaderTrackOutput> = unsafe {
            AVAssetReaderTrackOutput::initWithTrack_outputSettings(
                AVAssetReaderTrackOutput::alloc(),
                &track,
                None,
            )
        };

        // SAFETY: `reader` and `output` are valid.
        if !unsafe { reader.canAddOutput(&output) } {
            return Err(AvError::backend(
                "AVAssetReader rejected the video track output",
                0,
            ));
        }
        // SAFETY: `canAddOutput` just accepted this output.
        unsafe { reader.addOutput(&output) };

        // SAFETY: `reader` is valid and has its output attached.
        if !unsafe { reader.startReading() } {
            return Err(AvError::backend(
                "AVAssetReader startReading returned NO",
                0,
            ));
        }

        let demuxer = Self {
            _reader: reader,
            output,
            finished: false,
        };
        Ok((demuxer, track_info))
    }

    /// Returns the next compressed sample, or `None` at the end of the track. Skips
    /// AVFoundation's marker sample buffers, which carry no data.
    pub(crate) fn read_packet(&mut self) -> Option<CFRetained<CMSampleBuffer>> {
        // `copyNextSampleBuffer` is defined on the superclass.
        let output: &AVAssetReaderOutput = &self.output;
        while !self.finished {
            // SAFETY: `output` is valid and only read from this thread.
            let Some(sample_buffer) = (unsafe { output.copyNextSampleBuffer() }) else {
                self.finished = true;
                break;
            };
            let sample_buffer = retained_to_cf_retained(sample_buffer);
            // SAFETY: `sample_buffer` is valid.
            if unsafe { sample_buffer.data_buffer() }.is_some() {
                return Some(sample_buffer);
            }
        }
        None
    }
}

/// Finds the first video track in `asset`.
fn first_video_track(asset: &AVAsset) -> Result<(Retained<AVAssetTrack>, TrackInfo), AvError> {
    // SAFETY: `asset` is valid.
    let tracks: Retained<NSArray<AVAssetTrack>> = unsafe { asset.tracks() };
    // SAFETY: a framework constant that lives for the whole process.
    let media_type_video: &NSString =
        unsafe { AVMediaTypeVideo }.expect("AVMediaTypeVideo unavailable");

    for i in 0..tracks.len() {
        let track = tracks.objectAtIndex(i);
        // SAFETY: `track` is valid.
        if &*unsafe { track.mediaType() } != media_type_video {
            continue;
        }

        // SAFETY: `track` is valid.
        let format_descriptions: Retained<NSArray> = unsafe { track.formatDescriptions() };
        let Some(first) = format_descriptions.firstObject() else {
            return Err(AvError::Source(
                "video track has no format description".into(),
            ));
        };
        let raw: *const AnyObject = &*first;
        // SAFETY: a video track's format descriptions are CMFormatDescriptions, and
        // `first` keeps this one alive until we retain it.
        let format_description: CFRetained<CMFormatDescription> = unsafe {
            CFRetained::retain(NonNull::new_unchecked(
                raw.cast::<CMFormatDescription>().cast_mut(),
            ))
        };
        // SAFETY: `format_description` is valid.
        let codec = FourCC(unsafe { format_description.media_sub_type() });

        return Ok((
            track,
            TrackInfo {
                codec,
                format_description,
            },
        ));
    }

    Err(AvError::NoVideoTrack)
}

/// Moves a `Retained<T>` into a `CFRetained<T>` without changing the retain count.
/// `copyNextSampleBuffer` returns its `CMSampleBuffer`, a CF type, as `Retained`.
fn retained_to_cf_retained<T: Type + objc2::Message>(retained: Retained<T>) -> CFRetained<T> {
    let raw = Retained::into_raw(retained);
    // SAFETY: `raw` is non-null, and its retain moves into the `CFRetained`.
    unsafe { CFRetained::from_raw(NonNull::new_unchecked(raw)) }
}

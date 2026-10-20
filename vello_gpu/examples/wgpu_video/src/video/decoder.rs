// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Video decoder built on VTDecompressionSession.
//!
//! VT is asked for video-range NV12 in IOSurface-backed, Metal-compatible pixel buffers,
//! so it does no color conversion; Vello samples the planes directly.
//!
//! Kept minimal on purpose:
//! - Decoding is synchronous, on the caller's thread.
//! - Frames come out in decode order; `VideoFileSource` reorders them for display.
//! - No format-change handling: each file is assumed to have one codec and one
//!   resolution.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_media::{CMSampleBuffer, CMTime, CMTimeFlags};
use objc2_core_video::{
    CVImageBuffer, CVPixelBuffer, kCVImageBufferColorPrimariesKey,
    kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrixKey,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession,
};

use super::demuxer::TrackInfo;
use super::error::AvError;
use super::frame::{
    ColorMatrix, ColorPrimaries, ColorRange, ColorSpace, FourCC, TransferFunction, video_codec,
};

pub(crate) struct VideoDecoder {
    session: CFRetained<VTDecompressionSession>,
    /// VT's callback receives this as its refcon. An `Arc` rather than a `Box`, because
    /// moving a `Box` would invalidate the raw pointer VT holds.
    state: Arc<CallbackState>,
}

/// A frame VT has finished decoding, before its planes are imported.
pub(crate) struct DecodedPixelBuffer {
    pub(crate) pixel_buffer: CFRetained<CVPixelBuffer>,
    pub(crate) pts_ns: i64,
    pub(crate) color_space: ColorSpace,
}

impl VideoDecoder {
    /// Creates a session for `track`, failing if VT has no decoder for its format.
    pub(crate) fn new(track: &TrackInfo) -> Result<Self, AvError> {
        if !is_supported_codec(track.codec) {
            return Err(AvError::UnsupportedCodec(track.codec));
        }

        let state = Arc::new(CallbackState {
            queue: Mutex::new(VecDeque::new()),
        });
        let dest_attrs = build_destination_attrs();
        let dest_attrs_erased: &CFDictionary = (*dest_attrs).as_ref();
        let callback_record = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(decompression_output_callback),
            decompressionOutputRefCon: Arc::as_ptr(&state).cast_mut().cast(),
        };

        let mut session_ptr: *mut VTDecompressionSession = ptr::null_mut();
        // SAFETY: all references outlive the call. The refcon points at `state`, which
        // the returned decoder owns and only frees after `Drop` invalidates the session.
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                &track.format_description,
                None,
                Some(dest_attrs_erased),
                &callback_record,
                NonNull::new_unchecked(&mut session_ptr),
            )
        };
        if status != 0 || session_ptr.is_null() {
            return Err(AvError::backend(
                format!("VTDecompressionSessionCreate failed for {}", track.codec),
                status,
            ));
        }
        // SAFETY: on success, `create` stores a retained session in `session_ptr`.
        let session = unsafe { CFRetained::from_raw(NonNull::new_unchecked(session_ptr)) };
        Ok(Self { session, state })
    }

    /// Decodes one compressed sample. The frame, if any, is picked up by `recv_frame`.
    pub(crate) fn send_packet(&mut self, sample_buffer: &CMSampleBuffer) -> Result<(), AvError> {
        let mut info_flags = VTDecodeInfoFlags::empty();
        // SAFETY: `session` and `sample_buffer` are valid, and `info_flags` is writable.
        let status = unsafe {
            self.session.decode_frame(
                sample_buffer,
                VTDecodeFrameFlags::empty(),
                ptr::null_mut(),
                &mut info_flags,
            )
        };
        if status != 0 {
            return Err(AvError::backend(
                "VTDecompressionSessionDecodeFrame failed",
                status,
            ));
        }
        Ok(())
    }

    /// Waits for every frame VT is still decoding, so `recv_frame` can drain them.
    /// Call once the demuxer runs out of packets.
    pub(crate) fn flush(&mut self) {
        // SAFETY: `session` is valid.
        let status = unsafe { self.session.wait_for_asynchronous_frames() };
        if status != 0 {
            log::warn!("VTDecompressionSessionWaitForAsynchronousFrames failed: {status}");
        }
    }

    /// Returns the next frame VT has finished decoding, if any. Frames come in decode
    /// order, so B-frames arrive out of display order.
    pub(crate) fn recv_frame(&mut self) -> Option<DecodedPixelBuffer> {
        self.state.lock().pop_front()
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        // SAFETY: `invalidate` is the documented teardown, and returns only once no
        // callback is running, so `state` can be freed afterwards.
        unsafe { self.session.invalidate() };
    }
}

fn is_supported_codec(codec: FourCC) -> bool {
    matches!(
        codec,
        video_codec::H264
            | video_codec::HEVC
            | video_codec::HEVC_HEV1
            | video_codec::PRORES_4444
            | video_codec::PRORES_422
            | video_codec::AV1
            | video_codec::MJPEG
    )
}

/// Shared between the decoder and VT's callback, which runs on VT's own queue.
struct CallbackState {
    queue: Mutex<VecDeque<DecodedPixelBuffer>>,
}

impl CallbackState {
    /// A panicking callback can't leave the queue half-updated, so poisoning is ignored.
    fn lock(&self) -> MutexGuard<'_, VecDeque<DecodedPixelBuffer>> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

// SAFETY: `CallbackState` is reached from VT's queue as well as the decoder's thread.
// The queue is behind a `Mutex`, and each `CVPixelBuffer` in it is atomically
// reference-counted and not written after decoding. objc2 only leaves `CVPixelBuffer`
// non-`Send` because Apple's headers lack the annotation.
unsafe impl Send for CallbackState {}
unsafe impl Sync for CallbackState {}

/// Output pixel-buffer attributes: NV12, IOSurface-backed and Metal-compatible.
fn build_destination_attrs() -> CFRetained<CFDictionary<CFString, CFType>> {
    // SAFETY: these keys are framework constants that live for the whole process.
    let (key_format, key_metal_compat, key_iosurface_props): (&CFString, &CFString, &CFString) = unsafe {
        (
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferMetalCompatibilityKey,
            kCVPixelBufferIOSurfacePropertiesKey,
        )
    };

    let format_value = CFNumber::new_i32(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange as i32);
    let metal_compat_value = CFBoolean::new(true);
    let empty_iosurface_props = CFDictionary::<CFString, CFType>::empty();

    let keys: [&CFString; 3] = [key_format, key_metal_compat, key_iosurface_props];
    let values: [&CFType; 3] = [
        format_value.as_ref(),
        metal_compat_value.as_ref(),
        empty_iosurface_props.as_ref(),
    ];
    CFDictionary::from_slices(&keys, &values)
}

/// Converts a `CMTime` to nanoseconds, or 0 if it is invalid.
fn cmtime_to_ns(time: CMTime) -> i64 {
    if !time.flags.contains(CMTimeFlags::Valid) || time.timescale <= 0 {
        return 0;
    }
    let ns = i128::from(time.value) * 1_000_000_000 / i128::from(time.timescale);
    i64::try_from(ns).unwrap_or(i64::MAX)
}

/// Called by VT for each decoded frame; queues the frame for `recv_frame`.
unsafe extern "C-unwind" fn decompression_output_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: i32,
    _info_flags: VTDecodeInfoFlags,
    image_buffer: *mut CVImageBuffer,
    presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    if status != 0 {
        log::warn!("VT decode callback delivered status={status} (frame dropped)");
        return;
    }
    // A null buffer with a success status means VT dropped the frame on purpose.
    let Some(image_buffer) = NonNull::new(image_buffer) else {
        return;
    };

    // SAFETY: `refcon` points at the decoder's `CallbackState`, which outlives the
    // session.
    let state = unsafe { &*refcon.cast::<CallbackState>() };

    // SAFETY: `image_buffer` is a valid image buffer, and the destination attributes
    // make it a pixel buffer. VT only lends it to the callback, so retain it.
    let pixel_buffer: CFRetained<CVPixelBuffer> =
        unsafe { CFRetained::retain(image_buffer.cast::<CVPixelBuffer>()) };
    let color_space = read_color_space(&pixel_buffer);

    state.lock().push_back(DecodedPixelBuffer {
        pixel_buffer,
        pts_ns: cmtime_to_ns(presentation_time_stamp),
        color_space,
    });
}

/// Reads the color metadata VT attached to `pixel_buffer`. Missing or unrecognized
/// values stay `Unspecified`.
fn read_color_space(pixel_buffer: &CVPixelBuffer) -> ColorSpace {
    let mut cs = ColorSpace::default();

    let pf = objc2_core_video::CVPixelBufferGetPixelFormatType(pixel_buffer);
    cs.range = if pf == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };

    let image_buffer: &CVImageBuffer = pixel_buffer;
    let attachment = |key: &CFString| -> Option<String> {
        // SAFETY: `key` is a framework constant, and the returned value is retained for us.
        let raw = unsafe { image_buffer.attachment(key, ptr::null_mut()) }?;
        Some(raw.downcast_ref::<CFString>()?.to_string())
    };

    // SAFETY: these keys are framework constants that live for the whole process.
    let (matrix_key, primaries_key, transfer_key) = unsafe {
        (
            kCVImageBufferYCbCrMatrixKey,
            kCVImageBufferColorPrimariesKey,
            kCVImageBufferTransferFunctionKey,
        )
    };

    if let Some(s) = attachment(matrix_key) {
        cs.matrix = match s.as_str() {
            "ITU_R_601_4" => ColorMatrix::Bt601,
            "ITU_R_709_2" => ColorMatrix::Bt709,
            "ITU_R_2020" => ColorMatrix::Bt2020Ncl,
            _ => ColorMatrix::Unspecified,
        };
    }

    if let Some(s) = attachment(primaries_key) {
        cs.primaries = match s.as_str() {
            "ITU_R_709_2" => ColorPrimaries::Bt709,
            "SMPTE_C" => ColorPrimaries::Bt601_525,
            "EBU_3213" => ColorPrimaries::Bt601_625,
            "ITU_R_2020" => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unspecified,
        };
    }

    if let Some(s) = attachment(transfer_key) {
        cs.transfer = match s.as_str() {
            "ITU_R_709_2" => TransferFunction::Bt709,
            "SMPTE_ST_2084_PQ" => TransferFunction::SmpteSt2084,
            "ITU_R_2100_HLG" => TransferFunction::Hlg,
            _ => TransferFunction::Unspecified,
        };
    }

    cs
}

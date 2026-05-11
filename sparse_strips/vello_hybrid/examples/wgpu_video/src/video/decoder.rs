// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `VTDecompressionSession`-based video decoder configured for NV12,
//! IOSurface-backed, Metal-compatible output.
//!
//! Trimmed port of `lightspeed_av/backend/apple/video_toolbox.rs`:
//! - **No reorder window**: VT delivers frames in *decode* order. Sources with
//!   B-frames will therefore be presented out of display order. For this example's
//!   first cut that's acceptable; reintroducing the PTS-sorted release window from
//!   the source pipeline is straightforward when needed.
//! - **No flush / reset / format-change handling**: we assume a single resolution
//!   single-codec source played front-to-back exactly once.
//! - **No B-frame reorder tail at EOS**: same reason as above.
//!
//! The output is `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` (NV12) so VT does
//! no decode-time CSC. Today the example feeds these NV12 buffers to the
//! `RgbaConverter` for sampling by Vello; the same buffers are ready to plug into
//! a future Vello-side YCbCr sampler with no decoder changes.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};

use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_media::{CMFormatDescription, CMTime, CMTimeFlags, CMVideoFormatDescription};
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

use super::error::AvError;
use super::frame::{
    ColorMatrix, ColorPrimaries, ColorRange, ColorSpace, DecodedVideoFrame, TransferFunction,
    video_codec,
};
use super::metal_import::{MetalImporter, PlatformVideoFrameRetain};
use super::packet::{Packet, TrackInfo};

fn is_supported_codec(codec: super::frame::FourCC) -> bool {
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

struct DecodedEntry {
    pixel_buffer: CFRetained<CVPixelBuffer>,
    pts_ns: i64,
    color_space: ColorSpace,
}

struct CallbackState {
    queue: Mutex<VecDeque<DecodedEntry>>,
}

// SAFETY: queue sits behind a `Mutex`; entries hold CFType retains
// (`CVPixelBuffer`, atomic retain/release) over IOSurface storage that's read-only
// after VT decode finishes. The objc2-core-video crate doesn't auto-derive
// `Send`/`Sync` for `CVPixelBuffer` because the framework header lacks Sendable
// annotations; we assert the invariant here.
unsafe impl Send for CallbackState {}
unsafe impl Sync for CallbackState {}

pub(crate) struct VideoDecoder {
    /// Lazily-created on the first `send_packet` so we have a concrete
    /// `CMVideoFormatDescription` from the sample buffer.
    session: Option<CFRetained<VTDecompressionSession>>,
    importer: Arc<MetalImporter>,
    state: Arc<CallbackState>,
}

// SAFETY: sole-owner-driven (`&mut`-only API). The session is a CFType (atomic
// retain/release); `MetalImporter` is `Send + Sync`; `CallbackState` carries
// `unsafe impl Send`. The VT decode callback runs on a private dispatch queue
// reading the `Arc<CallbackState>` via the refcon, but `Drop` invalidates the
// session before any owned resource is released, which Apple documents as a
// barrier on outstanding callback work.
unsafe impl Send for VideoDecoder {}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        if let Some(session) = &self.session {
            // SAFETY: we created `session`. `invalidate` is the documented teardown
            // call; CFRetained's Drop releases the underlying CFType after.
            unsafe { session.invalidate() };
        }
    }
}

impl VideoDecoder {
    pub(crate) fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        track: &TrackInfo,
    ) -> Result<Self, AvError> {
        if !is_supported_codec(track.codec) {
            return Err(AvError::UnsupportedCodec(track.codec));
        }
        let importer = Arc::new(MetalImporter::new(adapter, device)?);
        Ok(Self {
            session: None,
            importer,
            state: Arc::new(CallbackState {
                queue: Mutex::new(VecDeque::new()),
            }),
        })
    }

    pub(crate) fn send_packet(&mut self, packet: &Packet) -> Result<(), AvError> {
        // SAFETY: typed accessor on a valid CMSampleBuffer.
        let format_desc: CFRetained<CMFormatDescription> =
            unsafe { packet.sample_buffer.format_description() }
                .ok_or_else(|| AvError::backend("Sample buffer missing format description", 0))?;

        if self.session.is_none() {
            self.create_session(&format_desc)?;
        }

        let Some(session) = self.session.as_deref() else {
            return Err(AvError::backend(
                "decompression session was not initialised",
                0,
            ));
        };

        let mut info_flags = VTDecodeInfoFlags::empty();
        // SAFETY: typed binding; `session` and `sample_buffer` are valid; the
        // info-flags out-pointer is a writable local.
        let status = unsafe {
            session.decode_frame(
                &packet.sample_buffer,
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

    /// Drain the next decoded NV12 frame, if any has been delivered by the VT callback
    /// since the previous call.
    ///
    /// Frames are surfaced in **decode order** (no reorder window). Single-resolution
    /// IPP-only sources see this as display order; sources with B-frames will see
    /// occasional out-of-order PTS.
    pub(crate) fn recv_frame(&mut self) -> Option<DecodedVideoFrame> {
        let entry = {
            let mut q = self.state.queue.lock().ok()?;
            q.pop_front()
        }?;

        let DecodedEntry {
            pixel_buffer,
            pts_ns,
            color_space,
        } = entry;

        if let Err(err) = self.importer.validate_nv12(&pixel_buffer) {
            log::warn!("validate_nv12 failed: {err}");
            return None;
        }

        let width = objc2_core_video::CVPixelBufferGetWidth(&pixel_buffer) as u32;
        let height = objc2_core_video::CVPixelBufferGetHeight(&pixel_buffer) as u32;
        Some(DecodedVideoFrame {
            width,
            height,
            pts_ns,
            color_space,
            retain: PlatformVideoFrameRetain::new_nv12(Arc::clone(&self.importer), pixel_buffer),
        })
    }

    fn create_session(
        &mut self,
        format_desc: &CFRetained<CMFormatDescription>,
    ) -> Result<(), AvError> {
        // CMVideoFormatDescription is a type alias for CMFormatDescription; the cast
        // is a no-op.
        let video_format_desc: &CMVideoFormatDescription = format_desc;

        let dest_attrs = build_destination_attrs();
        let callback_record = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(decompression_output_callback),
            decompressionOutputRefCon: Arc::as_ptr(&self.state) as *mut c_void,
        };

        let mut session_ptr: *mut VTDecompressionSession = ptr::null_mut();
        let dest_attrs_erased: &CFDictionary = (*dest_attrs).as_ref();
        // SAFETY: typed binding; all references outlive the call. The callback
        // record's refcon points at the `Arc<CallbackState>` which is owned by us.
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                video_format_desc,
                None,
                Some(dest_attrs_erased),
                &callback_record,
                NonNull::new_unchecked(&mut session_ptr),
            )
        };
        if status != 0 {
            return Err(AvError::backend(
                "VTDecompressionSessionCreate failed",
                status,
            ));
        }
        // SAFETY: VTDecompressionSession::create populated `session_ptr` with a +1
        // retained CFType.
        let session = unsafe { CFRetained::from_raw(NonNull::new_unchecked(session_ptr)) };
        self.session = Some(session);
        Ok(())
    }
}

/// Build the destination image-buffer attributes dictionary requesting NV12,
/// IOSurface-backed, Metal-compatible output buffers.
fn build_destination_attrs() -> CFRetained<CFDictionary<CFString, CFType>> {
    // SAFETY: each `kCV*Key` static is a framework-owned process-lifetime CFString
    // constant; reading them is always safe.
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

/// VT calls this on its private dispatch queue when a frame finishes decoding. We
/// retain the pixel buffer and push it to our queue.
unsafe extern "C-unwind" fn decompression_output_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: i32,
    _info_flags: VTDecodeInfoFlags,
    image_buffer: *mut CVImageBuffer,
    presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    let Some(image_buffer_nn) = NonNull::new(image_buffer) else {
        if status != 0 {
            log::warn!("VT decode callback delivered status={status} (frame dropped)");
        }
        return;
    };
    if status != 0 {
        log::warn!("VT decode callback delivered status={status} (frame dropped)");
        return;
    }

    // SAFETY: `refcon` is the `Arc::as_ptr` we passed in via the callback record; the
    // `Arc` is alive because we keep it in the decoder.
    let state = unsafe { &*(refcon as *const CallbackState) };

    // SAFETY: `image_buffer_nn` is a valid retained CVImageBuffer (CFType). The
    // destination attributes dictionary forced the buffer to be a planar pixel
    // buffer, so the cast to `CVPixelBuffer` is well-founded.
    let pixel_buffer: CFRetained<CVPixelBuffer> =
        unsafe { CFRetained::retain(image_buffer_nn.cast::<CVPixelBuffer>()) };

    let pts_ns = if presentation_time_stamp.flags.contains(CMTimeFlags::Valid)
        && presentation_time_stamp.timescale > 0
    {
        let value = presentation_time_stamp.value as i128;
        let scale = presentation_time_stamp.timescale as i128;
        ((value * 1_000_000_000) / scale) as i64
    } else {
        0
    };

    let color_space = read_color_space(&pixel_buffer);

    if let Ok(mut q) = state.queue.lock() {
        q.push_back(DecodedEntry {
            pixel_buffer,
            pts_ns,
            color_space,
        });
    }
}

fn read_color_space(pixel_buffer: &CVPixelBuffer) -> ColorSpace {
    let mut cs = ColorSpace::default();

    let pf = objc2_core_video::CVPixelBufferGetPixelFormatType(pixel_buffer);
    cs.range = if pf == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };

    let image_buffer: &CVImageBuffer = pixel_buffer;

    let attach_string = |key: &CFString| -> Option<CFRetained<CFString>> {
        // SAFETY: `key` is a framework CFString constant; the returned CFType (if
        // any) is retained on our behalf.
        let raw = unsafe { image_buffer.attachment(key, ptr::null_mut()) };
        // SAFETY: the framework guarantees these specific attachments are CFStrings.
        raw.map(|cf| unsafe { CFRetained::cast_unchecked::<CFString>(cf) })
    };

    // SAFETY: framework string constants live for process lifetime.
    let (matrix_key, primaries_key, transfer_key) = unsafe {
        (
            kCVImageBufferYCbCrMatrixKey,
            kCVImageBufferColorPrimariesKey,
            kCVImageBufferTransferFunctionKey,
        )
    };

    if let Some(s) = attach_string(matrix_key) {
        cs.matrix = match s.to_string().as_str() {
            "ITU_R_601_4" => ColorMatrix::Bt601,
            "ITU_R_709_2" => ColorMatrix::Bt709,
            "ITU_R_2020" => ColorMatrix::Bt2020Ncl,
            _ => ColorMatrix::Unspecified,
        };
    }

    if let Some(s) = attach_string(primaries_key) {
        cs.primaries = match s.to_string().as_str() {
            "ITU_R_709_2" => ColorPrimaries::Bt709,
            "SMPTE_C" => ColorPrimaries::Bt601_525,
            "EBU_3213" => ColorPrimaries::Bt601_625,
            "ITU_R_2020" => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unspecified,
        };
    }

    if let Some(s) = attach_string(transfer_key) {
        cs.transfer = match s.to_string().as_str() {
            "ITU_R_709_2" => TransferFunction::Bt709,
            "SMPTE_ST_2084_PQ" => TransferFunction::SmpteSt2084,
            "ITU_R_2100_HLG" => TransferFunction::Hlg,
            _ => TransferFunction::Unspecified,
        };
    }

    cs
}

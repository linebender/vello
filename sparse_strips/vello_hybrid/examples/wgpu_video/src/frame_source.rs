// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! "Decoded video frame" abstraction sitting between the rendering loop and the
//! macOS video pipeline.
//!
//! Currently a single implementation: [`VideoFileSource`], pulling NV12 frames from
//! [`crate::video::VideoPlayer`] (`AVAsset` demuxer + `VideoToolbox` decoder),
//! each exposed as a pair of zero-copy `wgpu::Texture` plane views (Y in `R8Unorm`,
//! Cb/Cr in `Rg8Unorm`) backed by the same `IOSurface` `VideoToolbox` decoded into.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vello_common::paint::{YCbCrInfo, YCbCrMatrix, YCbCrRange};
use wgpu::{Device, TextureView};

use crate::video::AvError;
use crate::video::{ColorMatrix, ColorRange, ColorSpace, PlayerFrame, VideoPlayer};

/// Number of decoded frames to keep buffered ahead of `current`. This is the
/// reorder window: the macOS VT decoder emits frames in decode order, where
/// B-frames trail their following reference frame, so we have to look ahead
/// by at least the source's `max_num_reorder_frames` and sort by PTS before
/// deciding which frame to display next. 4 covers typical H.264 / HEVC
/// settings (libx264 default `bf=3` plus B-pyramid); raise if a particular
/// source still shows B-frames out of order.
const REORDER_LOOKAHEAD: usize = 4;

/// A single decoded NV12 video frame, as seen by Vello.
///
/// Zero-copy: both plane views are backed by the same `IOSurface` that
/// `VideoToolbox` decoded into. The held [`PlayerFrame`] keeps the underlying
/// `wgpu::Texture` Y/UV planes and Apple-side retains (`CVMetalTexture`,
/// `CVPixelBuffer`) alive while wgpu samples through the views.
pub(crate) struct VideoFrame {
    pub(crate) y_view: TextureView,
    pub(crate) uv_view: TextureView,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) color_space: YCbCrInfo,
    /// Presentation timestamp in nanoseconds, propagated from the decoder.
    /// Used by [`VideoFileSource`] to pace playback at the source's native
    /// frame rate.
    pub(crate) pts_ns: i64,
    /// Holds the plane textures + `IOSurface` retains alive. Read by wgpu through
    /// the views; never accessed by name from Rust.
    #[expect(dead_code, reason = "kept alive to back the plane views")]
    player_frame: PlayerFrame,
}

impl VideoFrame {
    pub(crate) fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    pub(crate) fn y_view(&self) -> &TextureView {
        &self.y_view
    }

    pub(crate) fn uv_view(&self) -> &TextureView {
        &self.uv_view
    }

    pub(crate) fn color_space(&self) -> YCbCrInfo {
        self.color_space
    }
}

impl core::fmt::Debug for VideoFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VideoFrame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("color_space", &self.color_space)
            .finish_non_exhaustive()
    }
}

/// Source of decoded video frames.
///
/// `next_frame` is called once per render and the renderer binds the returned
/// frame's plane views via `vello_hybrid::TextureBindings::insert_ycbcr_nv12`.
/// The reference is valid until the next call.
pub(crate) trait FrameSource {
    fn next_frame(&mut self) -> &VideoFrame;

    /// Rewind to the beginning of the source. Called when the user hits `R`.
    fn restart(&mut self);
}

/// A `FrameSource` that pulls decoded frames from a [`VideoPlayer`] running the macOS
/// `AVAsset` / `VideoToolbox` pipeline, paced by the source's own presentation
/// timestamps so playback runs at native speed regardless of the host render rate.
///
/// Maintains a small PTS-sorted reorder buffer so that B-frame sources display
/// in correct order: VideoToolbox emits frames in decode order, where B-frames
/// follow the next reference frame in the stream, so without reordering a
/// PTS-paced loop would stall on the trailing reference and then race through
/// the buffered B-frames.
///
/// Caches the most recent frame so that:
/// * once the source reaches end-of-stream the renderer keeps showing the last frame
///   instead of crashing or going blank, and
/// * we always have something to return from `next_frame()` even before the very
///   first `RedrawRequested` (the constructor seeds the cache with the first frame).
///
/// Holds onto the source path + wgpu handles so [`Self::restart`] can rebuild the
/// underlying player from the beginning without going back through `main.rs`.
pub(crate) struct VideoFileSource {
    player: VideoPlayer,
    /// Frame currently being shown. Replaced by the smallest-PTS frame from
    /// `buffer` once wall-clock catches up to that frame's PTS.
    current: VideoFrame,
    /// Decoded frames sorted ascending by PTS, awaiting promotion. Acts as a
    /// reorder buffer for B-frame sources: the decoder emits frames in decode
    /// order, where B-frames trail their following reference frame, so we
    /// sort by PTS to recover display order before deciding when to promote.
    /// Held at [`REORDER_LOOKAHEAD`] entries (or until EOS) so the smallest-PTS
    /// front entry is guaranteed to be the actual next-display frame.
    buffer: VecDeque<VideoFrame>,
    path: PathBuf,
    /// `wgpu::Adapter` and `wgpu::Device` are Arc-internal in wgpu 29, so cloning is
    /// cheap and lets us rebuild the player on demand.
    adapter: wgpu::Adapter,
    device: Device,
    /// Set once the underlying [`VideoPlayer`] has reported end-of-stream
    /// (i.e. `next_frame()` returned `None`). Cleared by [`Self::restart`].
    eos: bool,
    /// `(wall_clock_anchor, pts_anchor_ns)`: the moment playback "started" mapped
    /// to the PTS at which it started. Set lazily on the first call to
    /// [`Self::next_frame`] after open/restart so we don't burn through frames
    /// while the window is still being created. Cleared on restart.
    anchor: Option<(Instant, i64)>,
}

impl VideoFileSource {
    /// Open `path`, pre-decode enough frames to fill the reorder buffer, and
    /// promote the smallest-PTS one as the initial `current`. Subsequent
    /// frames sit sorted-by-PTS in `buffer` waiting for their PTS deadlines.
    pub(crate) fn open(
        path: &Path,
        adapter: &wgpu::Adapter,
        device: &Device,
    ) -> Result<Self, AvError> {
        let (player, current, buffer) = open_player(path, adapter, device)?;
        Ok(Self {
            player,
            current,
            buffer,
            path: path.to_path_buf(),
            adapter: adapter.clone(),
            device: device.clone(),
            eos: false,
            anchor: None,
        })
    }

    /// Return the most recently decoded frame without advancing the source.
    pub(crate) fn current_frame(&self) -> &VideoFrame {
        &self.current
    }

    /// `true` once the underlying player has reached end-of-stream **and** the
    /// last decoded frame has been promoted to `current` (i.e. there are no
    /// more frames left to show until [`Self::restart`] is called).
    pub(crate) fn is_eos(&self) -> bool {
        self.eos && self.buffer.is_empty()
    }

    /// Insert `frame` into `buffer` keeping the buffer sorted ascending by PTS.
    /// `buffer` is small (`REORDER_LOOKAHEAD` entries), so a linear scan is the
    /// simplest implementation and effectively O(1) at this size.
    fn insert_sorted(&mut self, frame: VideoFrame) {
        let idx = self
            .buffer
            .iter()
            .position(|f| f.pts_ns > frame.pts_ns)
            .unwrap_or(self.buffer.len());
        self.buffer.insert(idx, frame);
    }

    /// Pull frames from the decoder until `buffer` holds [`REORDER_LOOKAHEAD`]
    /// entries or the source is exhausted. Called on each promotion so the
    /// look-ahead stays fully populated and we always pick the correct next-
    /// display frame in the presence of B-frames.
    fn refill_buffer(&mut self) {
        while !self.eos && self.buffer.len() < REORDER_LOOKAHEAD {
            match self.player.next_frame() {
                Some(next) => self.insert_sorted(player_frame_to_video_frame(next)),
                None => self.eos = true,
            }
        }
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> &VideoFrame {
        // Anchor wall-clock to the currently-displayed frame's PTS on the first
        // call after open/restart. Subsequent frames are then due when
        // `(buffer.front().pts_ns - pts_anchor) <= wall_clock_anchor.elapsed()`.
        let (wall_anchor, pts_anchor) = *self
            .anchor
            .get_or_insert_with(|| (Instant::now(), self.current.pts_ns));

        // Keep the look-ahead full so the front of `buffer` is the true
        // next-display frame even with B-frame reordering.
        self.refill_buffer();

        // Promote any buffered frames whose PTS deadline has passed. The loop
        // skips ahead through multiple frames if the renderer fell behind
        // (e.g. a stall), so playback resyncs to wall-clock instead of
        // accumulating lag.
        let elapsed_ns = i64::try_from(wall_anchor.elapsed().as_nanos()).unwrap_or(i64::MAX);
        while let Some(front_pts) = self.buffer.front().map(|f| f.pts_ns) {
            if front_pts.saturating_sub(pts_anchor) <= elapsed_ns {
                self.current = self.buffer.pop_front().expect("checked above");
                self.refill_buffer();
            } else {
                break;
            }
        }

        &self.current
    }

    fn restart(&mut self) {
        match open_player(&self.path, &self.adapter, &self.device) {
            Ok((player, current, buffer)) => {
                self.player = player;
                self.current = current;
                self.buffer = buffer;
                self.eos = false;
                self.anchor = None;
            }
            Err(err) => {
                log::warn!(
                    "failed to restart video pipeline at {}: {err}; \
                     keeping the previous (already-finished) player around so the \
                     last frame stays on screen",
                    self.path.display()
                );
            }
        }
    }
}

/// Open `path` and decode `REORDER_LOOKAHEAD + 1` frames upfront. Sort them by
/// PTS; the smallest becomes the source's initial `current` (the actual first-
/// display frame), the rest seed the reorder buffer. With B-frame sources this
/// "primes" the buffer so display order is correct from the very first promote.
fn open_player(
    path: &Path,
    adapter: &wgpu::Adapter,
    device: &Device,
) -> Result<(VideoPlayer, VideoFrame, VecDeque<VideoFrame>), AvError> {
    let mut player = VideoPlayer::open(path, adapter, device)?;
    let mut frames: Vec<VideoFrame> = Vec::with_capacity(REORDER_LOOKAHEAD + 1);
    for _ in 0..=REORDER_LOOKAHEAD {
        let Some(frame) = player.next_frame() else {
            break;
        };
        frames.push(player_frame_to_video_frame(frame));
    }
    if frames.is_empty() {
        return Err(AvError::Source("no frames in source".into()));
    }
    frames.sort_by_key(|f| f.pts_ns);
    let mut buffer: VecDeque<VideoFrame> = frames.into();
    let current = buffer
        .pop_front()
        .expect("frames is non-empty so buffer has at least one entry");
    Ok((player, current, buffer))
}

fn player_frame_to_video_frame(frame: PlayerFrame) -> VideoFrame {
    let color_space = map_color_space(frame.color_space);
    VideoFrame {
        y_view: frame.y_view.clone(),
        uv_view: frame.uv_view.clone(),
        width: frame.width,
        height: frame.height,
        color_space,
        pts_ns: frame.pts_ns,
        player_frame: frame,
    }
}

/// Translate the source-side metadata into the small set of fields Vello's shader
/// actually consumes. Anything `Unspecified` falls back to BT.709 limited range,
/// which matches the typical HD H.264 / HEVC content `VideoToolbox` produces.
fn map_color_space(src: ColorSpace) -> YCbCrInfo {
    let matrix = match src.matrix {
        ColorMatrix::Bt601 => YCbCrMatrix::Bt601,
        ColorMatrix::Bt709 | ColorMatrix::Unspecified => YCbCrMatrix::Bt709,
        ColorMatrix::Bt2020Ncl => YCbCrMatrix::Bt2020Ncl,
    };
    let range = match src.range {
        ColorRange::Limited => YCbCrRange::Limited,
        ColorRange::Full => YCbCrRange::Full,
    };
    YCbCrInfo { matrix, range }
}

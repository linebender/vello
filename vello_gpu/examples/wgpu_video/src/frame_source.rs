// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paces a [`VideoPlayer`]'s frames for the render loop.

use std::collections::VecDeque;
use std::mem;
use std::path::Path;
use std::time::Instant;

use wgpu::Device;

use crate::render_context::SubmissionTracker;
use crate::video::{AvError, VideoFrame, VideoPlayer};

/// How many decoded frames to buffer ahead of the one on screen.
///
/// `VideoToolbox` outputs frames in decode order, where a B-frame arrives after the
/// reference frame that follows it on screen. Sorting the buffer by timestamp restores
/// display order. 4 covers common H.264 / HEVC encodes (such as libx264's default of
/// 3 B-frames with B-pyramid); raise it if a video still shows frames out of order.
const REORDER_LOOKAHEAD: usize = 4;

/// Frame duration assumed until two timestamps have been seen (30 fps).
const DEFAULT_FRAME_INTERVAL_NS: i64 = 33_333_333;

/// Plays a video file at its native frame rate, showing each frame when its timestamp
/// comes due rather than once per render.
pub(crate) struct VideoFileSource {
    player: VideoPlayer,
    /// Frame on screen; replaced by the front of `upcoming` once that frame is due.
    current: VideoFrame,
    /// Decoded frames sorted by timestamp, kept [`REORDER_LOOKAHEAD`] long until the
    /// video ends so the front is always the next frame to show.
    upcoming: VecDeque<VideoFrame>,
    /// Frames taken off screen, each with the last submission that may have sampled
    /// it. Dropped once the GPU has finished that submission.
    retired: VecDeque<(u64, VideoFrame)>,
    /// Timestamp gap between the last two frames shown; how long the final frame stays
    /// up before the video counts as finished.
    frame_interval_ns: i64,
    /// The player has run out of frames.
    eos: bool,
    /// Wall-clock time and timestamp of the first frame shown since opening or
    /// restarting. Set on the first `advance` rather than at open, so no frames are
    /// skipped while the window is still being created.
    anchor: Option<(Instant, i64)>,
    /// Set after a failed restart, so a missing file doesn't retry every frame.
    restart_failed: bool,
}

impl VideoFileSource {
    /// Opens `path` and decodes the first few frames so playback starts in display order.
    pub(crate) fn open(
        path: &Path,
        adapter: &wgpu::Adapter,
        device: &Device,
    ) -> Result<Self, AvError> {
        let mut player = VideoPlayer::open(path, adapter, device)?;
        let (current, upcoming) = prime(&mut player)?;
        let color_space = current.color_space;
        if color_space.needs_color_management() {
            log::warn!(
                "{} is HDR or wide-gamut ({color_space:?}); colors will look washed out",
                path.display()
            );
        }
        Ok(Self {
            frame_interval_ns: initial_interval(&current, &upcoming),
            player,
            current,
            upcoming,
            retired: VecDeque::new(),
            eos: false,
            anchor: None,
            restart_failed: false,
        })
    }

    /// The frame to draw this render.
    pub(crate) fn current_frame(&self) -> &VideoFrame {
        &self.current
    }

    /// Moves to the latest frame that is due. After a stall this skips frames to catch
    /// up instead of falling further behind.
    pub(crate) fn advance(&mut self, submissions: &SubmissionTracker) {
        self.release_retired(submissions);

        let (wall_anchor, pts_anchor) = *self
            .anchor
            .get_or_insert_with(|| (Instant::now(), self.current.pts_ns));
        let elapsed_ns = elapsed_ns(wall_anchor);

        self.refill();
        let mut due: Option<VideoFrame> = None;
        while self
            .upcoming
            .front()
            .is_some_and(|f| f.pts_ns.saturating_sub(pts_anchor) <= elapsed_ns)
        {
            let frame = self.upcoming.pop_front().expect("checked above");
            let prev_pts = due.as_ref().unwrap_or(&self.current).pts_ns;
            if frame.pts_ns > prev_pts {
                self.frame_interval_ns = frame.pts_ns - prev_pts;
            }
            // A frame overtaken here was never drawn, so dropping it right away is safe.
            due = Some(frame);
            self.refill();
        }

        if let Some(frame) = due {
            let shown = mem::replace(&mut self.current, frame);
            self.retire(shown, submissions);
        }
    }

    /// Whether the last frame has been on screen for a full frame interval.
    pub(crate) fn is_finished(&self) -> bool {
        let Some((wall_anchor, pts_anchor)) = self.anchor else {
            return false;
        };
        let end_ns = (self.current.pts_ns - pts_anchor).saturating_add(self.frame_interval_ns);
        self.eos && self.upcoming.is_empty() && elapsed_ns(wall_anchor) >= end_ns
    }

    /// Rewinds to the first frame. On failure, keeps showing the current frame and
    /// stops trying until a restart succeeds.
    pub(crate) fn restart(&mut self, submissions: &SubmissionTracker) {
        if self.restart_failed {
            return;
        }
        let primed = self.player.rewind().and_then(|()| prime(&mut self.player));
        match primed {
            Ok((current, upcoming)) => {
                self.frame_interval_ns = initial_interval(&current, &upcoming);
                let shown = mem::replace(&mut self.current, current);
                self.retire(shown, submissions);
                self.upcoming = upcoming;
                self.eos = false;
                self.anchor = None;
            }
            Err(err) => {
                log::warn!("failed to restart video: {err}; keeping the last frame");
                self.restart_failed = true;
            }
        }
    }

    /// Decodes frames until `upcoming` is full or the video ends.
    fn refill(&mut self) {
        while !self.eos && self.upcoming.len() < REORDER_LOOKAHEAD {
            match self.player.next_frame() {
                Ok(Some(frame)) => insert_sorted(&mut self.upcoming, frame),
                Ok(None) => self.eos = true,
                Err(err) => {
                    log::warn!("decoding stopped early: {err}");
                    self.eos = true;
                }
            }
        }
    }

    /// Holds `frame` until the GPU finishes every submission made so far, any of which
    /// may sample it.
    fn retire(&mut self, frame: VideoFrame, submissions: &SubmissionTracker) {
        self.retired.push_back((submissions.submitted(), frame));
    }

    fn release_retired(&mut self, submissions: &SubmissionTracker) {
        let completed = submissions.completed();
        while self
            .retired
            .front()
            .is_some_and(|(last_use, _)| *last_use <= completed)
        {
            self.retired.pop_front();
        }
    }
}

/// Decodes `REORDER_LOOKAHEAD + 1` frames. The earliest by timestamp is shown first;
/// the rest fill the reorder buffer.
fn prime(player: &mut VideoPlayer) -> Result<(VideoFrame, VecDeque<VideoFrame>), AvError> {
    let mut upcoming = VecDeque::with_capacity(REORDER_LOOKAHEAD + 1);
    while upcoming.len() <= REORDER_LOOKAHEAD {
        let Some(frame) = player.next_frame()? else {
            break;
        };
        insert_sorted(&mut upcoming, frame);
    }
    let current = upcoming
        .pop_front()
        .ok_or_else(|| AvError::Source("no frames in source".into()))?;
    Ok((current, upcoming))
}

/// Inserts `frame` in timestamp order. The buffer is tiny, so a linear scan is fine.
fn insert_sorted(buffer: &mut VecDeque<VideoFrame>, frame: VideoFrame) {
    let idx = buffer
        .iter()
        .position(|f| f.pts_ns > frame.pts_ns)
        .unwrap_or(buffer.len());
    buffer.insert(idx, frame);
}

fn initial_interval(current: &VideoFrame, upcoming: &VecDeque<VideoFrame>) -> i64 {
    upcoming
        .front()
        .map(|next| next.pts_ns - current.pts_ns)
        .filter(|&interval| interval > 0)
        .unwrap_or(DEFAULT_FRAME_INTERVAL_NS)
}

fn elapsed_ns(since: Instant) -> i64 {
    i64::try_from(since.elapsed().as_nanos()).unwrap_or(i64::MAX)
}

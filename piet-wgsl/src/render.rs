//! Take an encoded scene and create a graph to render it

use crate::engine::{ShaderId, Recording, BufProxy};


pub fn render_reduced(tile_alloc: ShaderId) -> (Recording, BufProxy) {
    let mut recording = Recording::default();
    let path_buf = BufProxy::new(1024);
    let bump_buf = BufProxy::new(1024);
    recording.clear_all(bump_buf);
    recording.dispatch(
        tile_alloc,
        (1, 1, 1),
        [
            bump_buf,
            path_buf,
        ],
    );
    let download_buf = path_buf;
    recording.download(download_buf);
    (recording, download_buf)
}

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & alignment as usize - 1)
}

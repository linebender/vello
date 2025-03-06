use std::{num::NonZeroU64, sync::atomic::AtomicU64};

use peniko::kurbo::BezPath;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU64);

#[derive(Clone)]
pub struct Path {
    pub id: Id,
    pub path: BezPath,
    // TODO: Vello encoding. kurbo BezPath can be used in interim
    // Question: probably want to special-case rect, line, ellipse at least
    // Probably also rounded-rect (incl varying corner radii)
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Id {
    pub fn get() -> Self {
        let n = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(x) = n.checked_add(1) {
            Self(NonZeroU64::new(x).unwrap())
        } else {
            panic!("wow, overflow of u64, congratulations")
        }
    }
}

impl From<BezPath> for Path {
    fn from(path: BezPath) -> Self {
        let id = Id::get();
        Self { id, path }
    }
}

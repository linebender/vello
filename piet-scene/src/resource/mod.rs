mod ramp_cache;

use crate::brush::{Brush, Stop};
use ramp_cache::RampCache;

/// Context for caching resources across rendering operations.
#[derive(Default)]
pub struct ResourceContext {
    ramp_cache: RampCache,
}

impl ResourceContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&mut self) {
        self.ramp_cache.advance();
    }

    pub fn add_ramp(&mut self, stops: &[Stop]) -> u32 {
        self.ramp_cache.add(stops)
    }

    pub fn create_brush(&mut self, brush: &Brush) -> PersistentBrush {
        PersistentBrush { kind: 0, id: 0 }
    }

    pub fn destroy_brush(&mut self, brush: PersistentBrush) {}
}

/// Handle for a brush that is managed by the resource context.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct PersistentBrush {
    kind: u8,
    id: u64,
}

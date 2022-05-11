mod gradient;

use crate::brush::{Brush, Stop};
use gradient::RampCache;
use std::collections::HashMap;

/// Context for caching resources across rendering operations.
#[derive(Default)]
pub struct ResourceContext {
    ramps: RampCache,
    persistent_map: HashMap<u64, PersistentBrushData>,
}

impl ResourceContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&mut self) {
        self.ramps.advance();
    }

    pub fn clear(&mut self) {
        self.ramps.clear();
        self.persistent_map.clear();
    }

    pub fn add_ramp(&mut self, stops: &[Stop]) -> u32 {
        self.ramps.add(stops)
    }

    pub fn create_brush(&mut self, brush: &Brush) -> PersistentBrush {
        match brush {
            Brush::Persistent(dup) => return *dup,
            _ => {}
        }
        PersistentBrush { kind: 0, id: 0 }
    }

    pub fn destroy_brush(&mut self, brush: PersistentBrush) {}

    pub fn ramp_data(&self) -> &[u32] {
        &self.ramps.data()
    }
}

/// Handle for a brush that is managed by the resource context.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct PersistentBrush {
    kind: u8,
    id: u64,
}

struct PersistentBrushData {
    brush: Brush,
}

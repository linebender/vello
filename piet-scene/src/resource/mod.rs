mod gradient;

use crate::brush::GradientStop;
use gradient::RampCache;

/// Context for caching resources across rendering operations.
#[derive(Default)]
pub struct ResourceContext {
    ramps: RampCache,
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
    }

    pub fn add_ramp(&mut self, stops: &[GradientStop]) -> u32 {
        self.ramps.add(stops)
    }

    pub fn ramp_data(&self) -> &[u32] {
        &self.ramps.data()
    }
}

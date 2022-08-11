use crate::brush::GradientStop;
use core::ops::Range;

#[derive(Default)]
/// Collection of late bound resources for a scene or scene fragment.
pub struct ResourceBundle {
    /// Sequence of resource patches.
    pub patches: Vec<ResourcePatch>,
    /// Cache of gradient stops, referenced by range from the patches.
    pub stops: Vec<GradientStop>,
}

impl ResourceBundle {
    /// Clears the resource set.
    pub(crate) fn clear(&mut self) {
        self.patches.clear();
        self.stops.clear();
    }
}

#[derive(Clone)]
/// Description of a late bound resource.
pub enum ResourcePatch {
    /// Gradient ramp resource.
    Ramp {
        /// Byte offset to the ramp id in the draw data stream.
        offset: usize,
        /// Range of the gradient stops in the resource set.
        stops: Range<usize>,
    },
}

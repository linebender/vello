mod coarse;
pub use coarse::*;

mod fine;
pub use fine::*;

use crate::Recording;

use super::PassContext;

pub trait RenderPass: Send + Sync {
    type Output: Clone + Copy + 'static
    where
        Self: Sized;

    fn record(self, cx: PassContext<'_>) -> (Recording, Self::Output)
    where
        Self: Sized;
}

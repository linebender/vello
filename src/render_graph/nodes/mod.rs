mod coarse;
pub use coarse::*;

mod fine;
pub use fine::*;

pub trait RenderNode: Send + Sync {}

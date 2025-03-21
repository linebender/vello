use crate::paint::EncodedPaint::Solid;
use crate::util::ColorExt;
use vello_common::paint::Paint;

impl vello_common::coarse::EncodedPaint for EncodedPaint {
    type Solid = [u8; 4];

    fn as_solid_color(&self) -> Option<[u8; 4]> {
        match self {
            Solid(s) => Some(*s),
        }
    }
}

#[derive(Clone, Debug)]
pub enum EncodedPaint {
    Solid([u8; 4]),
}

impl From<Paint> for EncodedPaint {
    fn from(value: Paint) -> Self {
        match value {
            Paint::Solid(s) => Solid(s.premultiply().to_rgba8_fast()),
            _ => unimplemented!(),
        }
    }
}

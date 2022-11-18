use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct ClipEl {
    pub parent_ix: u32,
    pub pad: [u32; 3],
    pub bbox: [f32; 4],
}

pub fn parse_clip_els(data: &[u8]) -> Vec<ClipEl> {
    Vec::from(bytemuck::cast_slice(data))
}

use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct DrawMonoid {
    pub path_ix: u32,
    pub clip_ix: u32,
    pub scene_offset: u32,
    pub info_offset: u32,
}

pub fn parse_draw_monoids(data: &[u8]) -> Vec<DrawMonoid> {
    Vec::from(bytemuck::cast_slice(data))
}

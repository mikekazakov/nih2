use super::super::math::*;

pub struct MeshDataSection {
    pub start_index: usize,
    pub num_triangles: usize,
    pub material_index: usize,
}

#[derive(Default)]
pub struct MeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tex_coords: Vec<Vec2>,
    pub colors: Vec<Vec4>, // empty if absent
    pub indices: Vec<u32>,
    pub sections: Vec<MeshDataSection>,
    pub aabb: AABB,
}

use crate::math::*;

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vec4,
    pub world_position: Vec3,
    pub normal: Vec3,
    pub color: Vec4,
    pub tex_coord: Vec2,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            world_position: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            normal: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            color: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            tex_coord: Vec2 { x: 0.0, y: 0.0 },
        }
    }
}

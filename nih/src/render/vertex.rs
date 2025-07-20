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
            position: Vec4::new(0.0, 0.0, 0.0, 1.0),
            world_position: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
            color: Vec4::new(0.0, 0.0, 0.0, 0.0),
            tex_coord: Vec2::new(0.0, 0.0),
        }
    }
}

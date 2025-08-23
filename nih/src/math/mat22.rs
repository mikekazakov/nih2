use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat22(pub [f32; 4]);

impl Mat22 {
    pub fn identity() -> Mat22 {
        Mat22([
            1.0, 0.0, //
            0.0, 1.0, //
        ])
    }

    pub fn scale_uniform(s: f32) -> Mat22 {
        Mat22([
            s, 0.0, //
            0.0, s, //
        ])
    }

    pub fn scale_non_uniform(s: Vec2) -> Mat22 {
        Mat22([
            s.x, 0.0, //
            0.0, s.y, //
        ])
    }

    pub fn det(&self) -> f32 {
        let m = &self.0;
        m[0] * m[3] - m[1] * m[2]
    }
}

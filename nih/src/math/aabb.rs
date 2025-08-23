use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_points(positions: &[Vec3]) -> Self {
        if positions.is_empty() {
            return Self::default();
        }

        let mut min = positions[0];
        let mut max = positions[0];

        for &p in &positions[1..] {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }

        Self { min, max }
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Vec3::new(0.0, 0.0, 0.0), //
            max: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}

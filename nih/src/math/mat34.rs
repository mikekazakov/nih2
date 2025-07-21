use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat34(pub [f32; 12]);

impl Mat34 {
    pub fn identity() -> Mat34 {
        Mat34([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ])
    }

    pub fn scale_uniform(s: f32) -> Mat34 {
        Mat34([
            s, 0.0, 0.0, 0.0, //
            0.0, s, 0.0, 0.0, //
            0.0, 0.0, s, 0.0,
        ])
    }

    pub fn scale_non_uniform(s: Vec3) -> Mat34 {
        Mat34([
            s.x, 0.0, 0.0, 0.0, //
            0.0, s.y, 0.0, 0.0, //
            0.0, 0.0, s.z, 0.0,
        ])
    }

    pub fn rotate_xy(angle: f32) -> Mat34 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat34([
            cos, -sin, 0.0, 0.0, //
            sin, cos, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ])
    }

    pub fn rotate_yz(angle: f32) -> Mat34 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat34([
            1.0, 0.0, 0.0, 0.0, //
            0.0, cos, -sin, 0.0, //
            0.0, sin, cos, 0.0,
        ])
    }

    pub fn rotate_zx(angle: f32) -> Mat34 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat34([
            cos, 0.0, sin, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            -sin, 0.0, cos, 0.0,
        ])
    }

    pub fn translate(t: Vec3) -> Mat34 {
        Mat34([
            1.0, 0.0, 0.0, t.x, //
            0.0, 1.0, 0.0, t.y, //
            0.0, 0.0, 1.0, t.z,
        ])
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat34 {
        Mat34([
            2.0 / (right - left),
            0.0,
            0.0,
            -(right + left) / (right - left), //
            0.0,
            2.0 / (top - bottom),
            0.0,
            -(top + bottom) / (top - bottom), //
            0.0,
            0.0,
            -2.0 / (far - near),
            -(far + near) / (far - near),
        ])
    }

    pub fn as_mat33(&self) -> Mat33 {
        let m = &self.0;
        Mat33([
            m[0], m[1], m[2], //
            m[4], m[5], m[6], //
            m[8], m[9], m[10],
        ])
    }

    pub fn as_mat44(&self) -> Mat44 {
        let m = &self.0;
        Mat44([
            m[0], m[1], m[2], m[3], //
            m[4], m[5], m[6], m[7], //
            m[8], m[9], m[10], m[11], //
            0.0, 0.0, 0.0, 1.0,
        ])
    }
}

// Vec3 = Mat34 * Vec3
impl std::ops::Mul<Vec3> for Mat34 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.0[0] * v.x + self.0[1] * v.y + self.0[2] * v.z + self.0[3],
            y: self.0[4] * v.x + self.0[5] * v.y + self.0[6] * v.z + self.0[7],
            z: self.0[8] * v.x + self.0[9] * v.y + self.0[10] * v.z + self.0[11],
        }
    }
}

// Vec3 = &Mat34 * Vec3
impl std::ops::Mul<Vec3> for &Mat34 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.0[0] * v.x + self.0[1] * v.y + self.0[2] * v.z + self.0[3],
            y: self.0[4] * v.x + self.0[5] * v.y + self.0[6] * v.z + self.0[7],
            z: self.0[8] * v.x + self.0[9] * v.y + self.0[10] * v.z + self.0[11],
        }
    }
}

// Mat34 = Mat34 * Mat34
impl std::ops::Mul<Mat34> for Mat34 {
    type Output = Mat34;

    fn mul(self, other: Mat34) -> Mat34 {
        let mut result = [0.0f32; 12];

        // First three columns: rotation * rotation
        for row in 0..3 {
            for col in 0..3 {
                result[row * 4 + col] = self.0[row * 4 + 0] * other.0[0 * 4 + col]
                    + self.0[row * 4 + 1] * other.0[1 * 4 + col]
                    + self.0[row * 4 + 2] * other.0[2 * 4 + col];
            }
        }

        // Last column: a.rotation * b.translation + a.translation
        for row in 0..3 {
            result[row * 4 + 3] = self.0[row * 4 + 0] * other.0[3] +    // b.translation.x
                    self.0[row * 4 + 1] * other.0[7] +    // b.translation.y
                    self.0[row * 4 + 2] * other.0[11] +   // b.translation.z
                    self.0[row * 4 + 3]; // a.translation
        }

        Mat34(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2; // π/2

    #[test]
    fn test_identity() {
        let mat = Mat34::identity();
        let expected = Mat34([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_scale_uniform() {
        let mat = Mat34::scale_uniform(2.0);
        let expected = Mat34([2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        assert_eq!(mat, expected);
    }

    #[test]
    fn test_scale_non_uniform() {
        // let mat = Mat34::identity();
        let scale_vec = Vec3 { x: 2.0, y: 3.0, z: 4.0 };
        let expected_mat = Mat34([
            2.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, //
            0.0, 0.0, 4.0, 0.0,
        ]);
        assert_eq!(Mat34::scale_non_uniform(scale_vec), expected_mat);
    }

    #[test]
    fn test_mat34_translate() {
        let translation = Vec3 { x: 2.0, y: 3.0, z: 4.0 };
        let mat = Mat34::translate(translation);

        // Translation matrix should look like this:
        // [1 0 0 2]
        // [0 1 0 3]
        // [0 0 1 4]
        let expected = Mat34([
            1.0, 0.0, 0.0, 2.0, //
            0.0, 1.0, 0.0, 3.0, //
            0.0, 0.0, 1.0, 4.0,
        ]);

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_mat34_rotate_xy() {
        let angle = FRAC_PI_2; // 90 degrees
        let mat = Mat34::rotate_xy(angle);

        // Rotation around Z axis by 90 degrees
        // Should rotate (1, 0, 0) -> (0, 1, 0)
        let v = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        let rotated = mat * v;

        let expected = Vec3 { x: 0.0, y: 1.0, z: 0.0 };

        assert!((rotated.x - expected.x).abs() < 1e-6);
        assert!((rotated.y - expected.y).abs() < 1e-6);
        assert!((rotated.z - expected.z).abs() < 1e-6);
    }

    #[test]
    fn test_mat34_rotate_yz() {
        let angle = FRAC_PI_2; // 90 degrees
        let rot = Mat34::rotate_yz(angle);

        // Vector pointing in the Y direction
        let v = Vec3 { x: 0.0, y: 1.0, z: 0.0 };

        // Rotating around the X axis should push it toward Z
        let result = rot * v;

        let expected = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

        assert!((result.x - expected.x).abs() < 1e-6);
        assert!((result.y - expected.y).abs() < 1e-6);
        assert!((result.z - expected.z).abs() < 1e-6);
    }

    #[test]
    fn test_mat34_rotate_zx() {
        let angle = FRAC_PI_2; // 90 degrees
        let rot = Mat34::rotate_zx(angle);

        // Vector pointing in the Z direction
        let v = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

        // Rotating around the Y axis should push it toward X
        let result = rot * v;

        let expected = Vec3 { x: 1.0, y: 0.0, z: 0.0 };

        assert!((result.x - expected.x).abs() < 1e-6);
        assert!((result.y - expected.y).abs() < 1e-6);
        assert!((result.z - expected.z).abs() < 1e-6);
    }

    #[test]
    fn test_mat34_to_mat33() {
        let mat34 = Mat34([
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ]);

        let mat33 = mat34.as_mat33();

        let expected = Mat33([
            1.0, 2.0, 3.0, //
            5.0, 6.0, 7.0, //
            9.0, 10.0, 11.0,
        ]);

        for i in 0..9 {
            assert!((mat33.0[i] - expected.0[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mat34_mul_mat34() {
        let a = Mat34([
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 2.0, //
            0.0, 0.0, 1.0, 3.0,
        ]);

        let b = Mat34([
            0.0, -1.0, 0.0, 4.0, //
            1.0, 0.0, 0.0, 5.0, //
            0.0, 0.0, 1.0, 6.0,
        ]);

        let result = a * b;

        let expected = Mat34([
            0.0, -1.0, 0.0, 5.0, //
            1.0, 0.0, 0.0, 7.0, //
            0.0, 0.0, 1.0, 9.0, //
        ]);

        for i in 0..12 {
            assert!((result.0[i] - expected.0[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mat34_orthographic() {
        let ortho = Mat34::orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        let expected = Mat34([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, 0.0,
        ]);
        assert_eq!(ortho, expected);
    }
}

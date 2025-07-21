use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat44(pub [f32; 16]);

impl Mat44 {
    pub fn identity() -> Mat44 {
        Mat44([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn scale_uniform(s: f32) -> Mat44 {
        Mat44([
            s, 0.0, 0.0, 0.0, //
            0.0, s, 0.0, 0.0, //
            0.0, 0.0, s, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn scale_non_uniform(s: Vec3) -> Mat44 {
        Mat44([
            s.x, 0.0, 0.0, 0.0, //
            0.0, s.y, 0.0, 0.0, //
            0.0, 0.0, s.z, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn translate(s: Vec3) -> Mat44 {
        Mat44([
            1.0, 0.0, 0.0, s.x, //
            0.0, 1.0, 0.0, s.y, //
            0.0, 0.0, 1.0, s.z, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn rotate_xy(angle: f32) -> Mat44 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat44([
            cos, -sin, 0.0, 0.0, //
            sin, cos, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn rotate_yz(angle: f32) -> Mat44 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat44([
            1.0, 0.0, 0.0, 0.0, //
            0.0, cos, -sin, 0.0, //
            0.0, sin, cos, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn rotate_zx(angle: f32) -> Mat44 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat44([
            cos, 0.0, sin, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            -sin, 0.0, cos, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat44 {
        Mat44([
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
            -(far + near) / (far - near), //
            0.0,
            0.0,
            0.0,
            1.0,
        ])
    }

    // Z: [-1, 1]
    // near -> -1
    // far  -> +1
    pub fn perspective(near: f32, far: f32, fov_y: f32, aspect_ratio: f32) -> Mat44 {
        let top = near * (fov_y / 2.0).tan();
        let right = top * aspect_ratio;

        Mat44([
            near / right,
            0.0,
            0.0,
            0.0,
            0.0,
            near / top,
            0.0,
            0.0,
            0.0,
            0.0,
            -(far + near) / (far - near),
            -2.0 * far * near / (far - near),
            0.0,
            0.0,
            -1.0,
            0.0,
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

    pub fn as_mat34(&self) -> Mat34 {
        let m = &self.0;
        Mat34([
            m[0], m[1], m[2], m[3], //
            m[4], m[5], m[6], m[7], //
            m[8], m[9], m[10], m[11],
        ])
    }

    pub fn inverse(&self) -> Mat44 {
        let a = &self.0;

        let mut inv = Mat44::identity();
        let o = &mut inv.0;

        o[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
            + a[9] * a[7] * a[14]
            + a[13] * a[6] * a[11]
            - a[13] * a[7] * a[10];

        o[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
            - a[9] * a[3] * a[14]
            - a[13] * a[2] * a[11]
            + a[13] * a[3] * a[10];

        o[2] =
            a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7]
                - a[13] * a[3] * a[6];

        o[3] =
            -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7]
                + a[9] * a[3] * a[6];

        o[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
            - a[8] * a[7] * a[14]
            - a[12] * a[6] * a[11]
            + a[12] * a[7] * a[10];

        o[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
            + a[8] * a[3] * a[14]
            + a[12] * a[2] * a[11]
            - a[12] * a[3] * a[10];

        o[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
            - a[4] * a[3] * a[14]
            - a[12] * a[2] * a[7]
            + a[12] * a[3] * a[6];

        o[7] =
            a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7]
                - a[8] * a[3] * a[6];

        o[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
            + a[8] * a[7] * a[13]
            + a[12] * a[5] * a[11]
            - a[12] * a[7] * a[9];

        o[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
            - a[8] * a[3] * a[13]
            - a[12] * a[1] * a[11]
            + a[12] * a[3] * a[9];

        o[10] =
            a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7]
                - a[12] * a[3] * a[5];

        o[11] =
            -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7]
                + a[8] * a[3] * a[5];

        o[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
            - a[8] * a[6] * a[13]
            - a[12] * a[5] * a[10]
            + a[12] * a[6] * a[9];

        o[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
            + a[8] * a[2] * a[13]
            + a[12] * a[1] * a[10]
            - a[12] * a[2] * a[9];

        o[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
            - a[4] * a[2] * a[13]
            - a[12] * a[1] * a[6]
            + a[12] * a[2] * a[5];

        o[15] =
            a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6]
                - a[8] * a[2] * a[5];

        let det = a[0] * o[0] + a[1] * o[4] + a[2] * o[8] + a[3] * o[12];

        if det.abs() < 1e-6 {
            return Mat44::identity(); // Handle non-invertible matrix
        }

        let inv_det = 1.0 / det;
        for i in 0..16 {
            o[i] *= inv_det;
        }

        return inv;
    }
}

// Vec4 = Mat44 * Vec4
impl std::ops::Mul<Vec4> for Mat44 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        let m = &self.0;
        Vec4 {
            x: m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
            y: m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
            z: m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w,
            w: m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w,
        }
    }
}

// Mat44 = Mat44 * Mat44
impl std::ops::Mul for Mat44 {
    type Output = Mat44;

    fn mul(self, other: Mat44) -> Mat44 {
        let mut result = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[4 * i + j] += self.0[4 * i + k] * other.0[4 * k + j];
                }
            }
        }
        Mat44(result)
    }
}

// Mat44 = &Mat44 * &Mat44
impl std::ops::Mul<&Mat44> for &Mat44 {
    type Output = Mat44;

    fn mul(self, other: &Mat44) -> Mat44 {
        let mut result = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[4 * i + j] += self.0[4 * i + k] * other.0[4 * k + j];
                }
            }
        }
        Mat44(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec4::Vec4;

    #[test]
    fn test_mat44_identity_mul_vec4() {
        let m = Mat44::identity();
        let v = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 1.0 };
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat44_mul_mat44_identity() {
        let a = Mat44::identity();
        let b = Mat44::identity();
        let result = a * b;
        let expected = Mat44::identity();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mat44_ref_mul_mat44_ref_identity() {
        let a = Mat44::identity();
        let b = Mat44::identity();
        let result = &a * &b;
        let expected = Mat44::identity();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mat44_translate() {
        let t = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let m = Mat44::translate(t);
        let v = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
        let result = m * v;
        assert_eq!(result, Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 1.0 });
    }

    #[test]
    fn test_mat44_scale_uniform() {
        let m = Mat44::scale_uniform(2.0);
        let v = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 1.0 };
        let result = m * v;
        assert_eq!(result, Vec4 { x: 2.0, y: 4.0, z: 6.0, w: 1.0 });
    }

    #[test]
    fn test_mat44_scale_non_uniform() {
        let m = Mat44::scale_non_uniform(Vec3 { x: 2.0, y: 3.0, z: 4.0 });
        let v = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
        let result = m * v;
        assert_eq!(result, Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 1.0 });
    }

    #[test]
    fn test_mat44_as_mat33() {
        let m = Mat44::identity();
        let m33 = m.as_mat33();
        assert_eq!(m33.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mat44_as_mat34() {
        let m = Mat44::identity();
        let m34 = m.as_mat34();
        assert_eq!(m34.0, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_mat44_inverse_identity() {
        let m = Mat44::identity();
        let inv = m.inverse();
        assert_eq!(inv, Mat44::identity());
    }

    #[test]
    fn test_mat44_rotate_xy() {
        let m = Mat44::rotate_xy(std::f32::consts::FRAC_PI_2); // 90 deg
        let v = Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 1.0 };
        let result = m * v;
        assert!((result.x.abs() < 1e-6) && ((result.y - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_mat44_rotate_yz() {
        let m = Mat44::rotate_yz(std::f32::consts::FRAC_PI_2);
        let v = Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 1.0 };
        let result = m * v;
        assert!((result.y.abs() < 1e-6) && ((result.z - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_mat44_rotate_zx() {
        let m = Mat44::rotate_zx(std::f32::consts::FRAC_PI_2);
        let v = Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 1.0 };
        let result = m * v;
        assert!((result.z.abs() < 1e-6) && ((result.x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_mat44_inverse_non_invertible() {
        // A matrix with a row of zeros is not invertible
        let m = Mat44([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, // This row makes the matrix singular
        ]);
        let inv = m.inverse();
        assert_eq!(inv, Mat44::identity());
    }
}

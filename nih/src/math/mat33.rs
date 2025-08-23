use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat33(pub [f32; 9]);

impl Mat33 {
    pub fn identity() -> Mat33 {
        Mat33([
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ])
    }

    pub fn scale_uniform(s: f32) -> Mat33 {
        Mat33([
            s, 0.0, 0.0, //
            0.0, s, 0.0, //
            0.0, 0.0, s,
        ])
    }

    pub fn scale_non_uniform(s: Vec3) -> Mat33 {
        Mat33([
            s.x, 0.0, 0.0, //
            0.0, s.y, 0.0, //
            0.0, 0.0, s.z,
        ])
    }

    pub fn rotate_xy(angle: f32) -> Mat33 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat33([
            cos, -sin, 0.0, //
            sin, cos, 0.0, //
            0.0, 0.0, 1.0,
        ])
    }

    pub fn rotate_yz(angle: f32) -> Mat33 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat33([
            1.0, 0.0, 0.0, //
            0.0, cos, -sin, //
            0.0, sin, cos,
        ])
    }

    pub fn rotate_zx(angle: f32) -> Mat33 {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat33([
            cos, 0.0, sin, //
            0.0, 1.0, 0.0, //
            -sin, 0.0, cos,
        ])
    }

    pub fn inverse(&self) -> Self {
        let m = &self.0;
        let a = m[0];
        let b = m[1];
        let c = m[2];
        let d = m[3];
        let e = m[4];
        let f = m[5];
        let g = m[6];
        let h = m[7];
        let i = m[8];

        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

        if det.abs() < 1e-6 {
            return Mat33::identity(); // fallback to identity
        }

        let inv_det = 1.0 / det;

        let inv = [
            (e * i - f * h) * inv_det,
            -(b * i - c * h) * inv_det,
            (b * f - c * e) * inv_det,
            -(d * i - f * g) * inv_det,
            (a * i - c * g) * inv_det,
            -(a * f - c * d) * inv_det,
            (d * h - e * g) * inv_det,
            -(a * h - b * g) * inv_det,
            (a * e - b * d) * inv_det,
        ];

        Self(inv)
    }

    // TODO: unit test
    pub fn transpose(&self) -> Mat33 {
        let m = &self.0;
        Mat33([
            m[0], m[3], m[6], //
            m[1], m[4], m[7], //
            m[2], m[5], m[8], //
        ])
    }
}

// Vec3 = Mat33 * Vec3
impl std::ops::Mul<Vec3> for Mat33 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        let m = &self.0;

        Vec3 {
            x: m[0] * v.x + m[1] * v.y + m[2] * v.z,
            y: m[3] * v.x + m[4] * v.y + m[5] * v.z,
            z: m[6] * v.x + m[7] * v.y + m[8] * v.z,
        }
    }
}

// Mat33 = Mat33 * Mat33
impl std::ops::Mul for Mat33 {
    type Output = Mat33;

    fn mul(self, other: Mat33) -> Mat33 {
        let mut result = [0.0; 9];

        for row in 0..3 {
            for col in 0..3 {
                result[row * 3 + col] = self.0[row * 3 + 0] * other.0[0 * 3 + col]
                    + self.0[row * 3 + 1] * other.0[1 * 3 + col]
                    + self.0[row * 3 + 2] * other.0[2 * 3 + col];
            }
        }

        Mat33(result)
    }
}

// Helper en
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn mats_approx_eq(a: &Mat33, b: &Mat33, epsilon: f32) -> bool {
        a.0.iter().zip(b.0.iter()).all(|(x, y)| approx_eq(*x, *y, epsilon))
    }

    // Helper struct for approximate matrix comparison
    #[derive(Debug)]
    struct Mat33Approx(Mat33);

    impl PartialEq<Mat33Approx> for Mat33 {
        fn eq(&self, other: &Mat33Approx) -> bool {
            const EPSILON: f32 = 1e-4;
            for i in 0..9 {
                if (self.0[i] - other.0.0[i]).abs() > EPSILON {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    fn test_identity() {
        let identity = Mat33::identity();
        assert_eq!(identity.0[0], 1.0);
        assert_eq!(identity.0[1], 0.0);
        assert_eq!(identity.0[2], 0.0);
        assert_eq!(identity.0[3], 0.0);
        assert_eq!(identity.0[4], 1.0);
        assert_eq!(identity.0[5], 0.0);
        assert_eq!(identity.0[6], 0.0);
        assert_eq!(identity.0[7], 0.0);
        assert_eq!(identity.0[8], 1.0);
    }

    #[test]
    fn test_scale_vec3() {
        let scale_vec = Vec3 { x: 2.0, y: 3.0, z: 4.0 };
        let scale_mat = Mat33::scale_non_uniform(scale_vec);

        assert_eq!(scale_mat.0[0], 2.0);
        assert_eq!(scale_mat.0[4], 3.0);
        assert_eq!(scale_mat.0[8], 4.0);

        // Check that off-diagonal elements are zero
        assert_eq!(scale_mat.0[1], 0.0);
        assert_eq!(scale_mat.0[2], 0.0);
        assert_eq!(scale_mat.0[3], 0.0);
        assert_eq!(scale_mat.0[5], 0.0);
        assert_eq!(scale_mat.0[6], 0.0);
        assert_eq!(scale_mat.0[7], 0.0);
    }

    #[test]
    fn test_scale_f32() {
        let scale_factor = 2.5;
        let scale_mat = Mat33::scale_uniform(scale_factor);

        assert_eq!(scale_mat.0[0], 2.5);
        assert_eq!(scale_mat.0[4], 2.5);
        assert_eq!(scale_mat.0[8], 2.5);

        // Check that off-diagonal elements are zero
        assert_eq!(scale_mat.0[1], 0.0);
        assert_eq!(scale_mat.0[2], 0.0);
        assert_eq!(scale_mat.0[3], 0.0);
        assert_eq!(scale_mat.0[5], 0.0);
        assert_eq!(scale_mat.0[6], 0.0);
        assert_eq!(scale_mat.0[7], 0.0);
    }

    #[test]
    fn test_rotate_xy() {
        // Test 90-degree rotation
        let rot_90 = Mat33::rotate_xy(PI / 2.0);

        assert_eq!(rot_90, Mat33Approx(Mat33([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,],)));

        // Test 180-degree rotation
        let rot_180 = Mat33::rotate_xy(PI);

        assert_eq!(rot_180, Mat33Approx(Mat33([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,],)));
    }

    #[test]
    fn test_rotate_yz() {
        // Test 90-degree rotation
        let rot_90 = Mat33::rotate_yz(PI / 2.0);

        assert_eq!(rot_90, Mat33Approx(Mat33([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0,],)));

        // Test 180-degree rotation
        let rot_180 = Mat33::rotate_yz(PI);

        assert_eq!(rot_180, Mat33Approx(Mat33([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,],)));
    }

    #[test]
    fn test_rotate_zx() {
        // Test 90-degree rotation
        let rot_90 = Mat33::rotate_zx(PI / 2.0);

        assert_eq!(rot_90, Mat33Approx(Mat33([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0,],)));

        // Test 180-degree rotation
        let rot_180 = Mat33::rotate_zx(PI);

        assert_eq!(rot_180, Mat33Approx(Mat33([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,],)));
    }

    #[test]
    fn test_inverse() {
        let m = Mat33([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let inv = m.inverse();

        let identity = Mat33::identity();
        let product = m * inv;

        assert!(mats_approx_eq(&product, &identity, 1e-5));
    }
}

use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn length(self) -> f32 {
        dot(self, self).sqrt()
    }

    pub fn normalized(self) -> Vec4 {
        let len = self.length();
        self / len
    }

    pub fn clamped(self, min: f32, max: f32) -> Vec4 {
        Vec4 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
            w: self.w.clamp(min, max),
        }
    }

    pub fn xyz(self) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }
}

// a * b
impl Dot for Vec4 {
    fn dot(self, rhs: Vec4) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

// -Vec4
impl std::ops::Neg for Vec4 {
    type Output = Vec4;
    fn neg(self) -> Vec4 {
        Vec4 { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

// Vec4 + Vec4
impl std::ops::Add for Vec4 {
    type Output = Vec4;
    fn add(self, other: Vec4) -> Vec4 {
        Vec4 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z, w: self.w + other.w }
    }
}

// Vec4 - Vec4
impl std::ops::Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, other: Vec4) -> Vec4 {
        Vec4 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z, w: self.w - other.w }
    }
}

// Vec4 * f32
impl std::ops::Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, scalar: f32) -> Vec4 {
        Vec4 { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar, w: self.w * scalar }
    }
}

// f32 * Vec4
impl std::ops::Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, vec: Vec4) -> Vec4 {
        Vec4 { x: vec.x * self, y: vec.y * self, z: vec.z * self, w: vec.w * self }
    }
}

// Vec4 / f32
impl std::ops::Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, scalar: f32) -> Vec4 {
        Vec4 { x: self.x / scalar, y: self.y / scalar, z: self.z / scalar, w: self.w / scalar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4_creation_and_equality() {
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v3 = Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 };

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_eq!(v1.x, 1.0);
        assert_eq!(v1.y, 2.0);
        assert_eq!(v1.z, 3.0);
        assert_eq!(v1.w, 4.0);
    }

    #[test]
    fn test_vec4_negation() {
        let v = Vec4 { x: 2.0, y: -3.0, z: 4.0, w: -5.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -2.0);
        assert_eq!(neg_v.y, 3.0);
        assert_eq!(neg_v.z, -4.0);
        assert_eq!(neg_v.w, 5.0);
    }

    #[test]
    fn test_vec4_addition() {
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2 = Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 };
        let sum = v1 + v2;

        assert_eq!(sum.x, 6.0);
        assert_eq!(sum.y, 8.0);
        assert_eq!(sum.z, 10.0);
        assert_eq!(sum.w, 12.0);
    }

    #[test]
    fn test_vec4_subtraction() {
        let v1 = Vec4 { x: 5.0, y: 7.0, z: 9.0, w: 11.0 };
        let v2 = Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 5.0 };
        let diff = v1 - v2;

        assert_eq!(diff.x, 3.0);
        assert_eq!(diff.y, 4.0);
        assert_eq!(diff.z, 5.0);
        assert_eq!(diff.w, 6.0);
    }

    #[test]
    fn test_vec4_multiplication_by_scalar() {
        let v = Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 5.0 };
        let scaled = v * 2.0;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
        assert_eq!(scaled.z, 8.0);
        assert_eq!(scaled.w, 10.0);
    }

    #[test]
    fn test_scalar_multiplication_by_vec4() {
        let v = Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 5.0 };
        let scaled = 2.0 * v;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
        assert_eq!(scaled.z, 8.0);
        assert_eq!(scaled.w, 10.0);
    }

    #[test]
    fn test_vec4_division_by_scalar() {
        let v = Vec4 { x: 4.0, y: 6.0, z: 8.0, w: 10.0 };
        let divided = v / 2.0;

        assert_eq!(divided.x, 2.0);
        assert_eq!(divided.y, 3.0);
        assert_eq!(divided.z, 4.0);
        assert_eq!(divided.w, 5.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2 = Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 };
        let dot_result = dot(v1, v2);

        // 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0 = 5.0 + 12.0 + 21.0 + 32.0 = 70.0
        assert_eq!(dot_result, 70.0);
    }

    #[test]
    fn test_length() {
        let v = Vec4 { x: 3.0, y: 4.0, z: 0.0, w: 0.0 };
        let length = v.length();

        // sqrt(3² + 4² + 0² + 0²) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert_eq!(length, 5.0);

        let v2 = Vec4 { x: 1.0, y: 2.0, z: 2.0, w: 3.0 };
        let length2 = v2.length();

        // sqrt(1² + 2² + 2² + 3²) = sqrt(1 + 4 + 4 + 9) = sqrt(18) = 4.24...
        assert!((length2 - 18.0_f32.sqrt()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zero_vector_length() {
        let zero_vec = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let length = zero_vec.length();

        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_division() {
        {
            let v = Vec4 { x: 3.0, y: 4.0, z: 5.0, w: 6.0 };
            assert_eq!(v / 2.0, Vec4 { x: 1.5, y: 2.0, z: 2.5, w: 3.0 });
            assert_eq!(v / 0.5, Vec4 { x: 6.0, y: 8.0, z: 10.0, w: 12.0 });
        }
        {
            // division by zero
            let v = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
            let result = v / 0.0;

            // Division by zero for f32 results in infinity
            assert!(result.x.is_infinite());
            assert!(result.y.is_infinite());
            assert!(result.z.is_infinite());
            assert!(result.w.is_infinite());
        }
    }

    #[test]
    fn test_normalized() {
        let v = Vec4 { x: 3.0, y: 4.0, z: 0.0, w: 0.0 };
        let normalized = v.normalized();

        // The length of a normalized vector should be 1.0
        assert!((normalized.length() - 1.0).abs() < f32::EPSILON);

        // The direction should be preserved
        // For a vector (3,4,0,0) with length 5, the normalized vector should be (3/5, 4/5, 0, 0)
        assert!((normalized.x - 0.6).abs() < f32::EPSILON);
        assert!((normalized.y - 0.8).abs() < f32::EPSILON);
        assert!((normalized.z - 0.0).abs() < f32::EPSILON);
        assert!((normalized.w - 0.0).abs() < f32::EPSILON);

        // Test with a different vector
        let v2 = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
        let normalized2 = v2.normalized();

        // Length should be 1.0
        assert!((normalized2.length() - 1.0).abs() < f32::EPSILON);

        // For a vector (1,1,1,1) with length 2, the normalized vector should be (0.5, 0.5, 0.5, 0.5)
        let expected = 0.5;
        assert!((normalized2.x - expected).abs() < f32::EPSILON);
        assert!((normalized2.y - expected).abs() < f32::EPSILON);
        assert!((normalized2.z - expected).abs() < f32::EPSILON);
        assert!((normalized2.w - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zero_vector_normalized() {
        let zero_vec = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let normalized = zero_vec.normalized();

        // Normalizing a zero vector should result in NaN values
        assert!(normalized.x.is_nan());
        assert!(normalized.y.is_nan());
        assert!(normalized.z.is_nan());
        assert!(normalized.w.is_nan());
    }

    #[test]
    fn test_clamped() {
        // Test clamping all components within range
        let v1 = Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 5.0 };
        let clamped1 = v1.clamped(1.0, 6.0);
        assert_eq!(clamped1, v1); // All components are within range

        // Test clamping components below minimum
        let v2 = Vec4 { x: -1.0, y: 0.5, z: 2.0, w: 1.0 };
        let clamped2 = v2.clamped(1.0, 5.0);
        assert_eq!(clamped2, Vec4 { x: 1.0, y: 1.0, z: 2.0, w: 1.0 });

        // Test clamping components above maximum
        let v3 = Vec4 { x: 3.0, y: 6.0, z: 10.0, w: 7.0 };
        let clamped3 = v3.clamped(1.0, 5.0);
        assert_eq!(clamped3, Vec4 { x: 3.0, y: 5.0, z: 5.0, w: 5.0 });

        // Note: The Rust standard library's clamp function requires min <= max
        // and will panic if min > max, so we don't test that case.
    }

    #[test]
    fn test_xyz() {
        // Test normal vector
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v3 = v1.xyz();

        // Check that x, y, z components match
        assert_eq!(v3.x, v1.x);
        assert_eq!(v3.y, v1.y);
        assert_eq!(v3.z, v1.z);

        // Test zero vector
        let zero_vec = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let zero_vec3 = zero_vec.xyz();

        assert_eq!(zero_vec3.x, 0.0);
        assert_eq!(zero_vec3.y, 0.0);
        assert_eq!(zero_vec3.z, 0.0);

        // Test negative values
        let neg_vec = Vec4 { x: -1.0, y: -2.0, z: -3.0, w: -4.0 };
        let neg_vec3 = neg_vec.xyz();

        assert_eq!(neg_vec3.x, -1.0);
        assert_eq!(neg_vec3.y, -2.0);
        assert_eq!(neg_vec3.z, -3.0);
    }
}

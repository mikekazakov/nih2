use crate::math::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn length(self) -> f32 {
        dot(self, self).sqrt()
    }

    pub fn normalized(self) -> Vec2 {
        let len = self.length();
        self / len
    }
}

impl Dot for Vec2 {
    fn dot(self, rhs: Vec2) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }
}

// Distance from point `p` to line (v0, v1)
fn distance(v0: Vec2, v1: Vec2, p: Vec2) -> f32 {
    let v01 = v1 - v0;
    let len_sq = dot(v01, v01);

    if len_sq == 0.0 {
        return (p - v0).length();
    }

    let vp = p - v0;
    let t = dot(vp, v01) / len_sq;
    let a = v0 + v01 * t;
    (p - a).length()
}

// Squared distance (cheaper, for comparisons only)
fn distance2(v0: Vec2, v1: Vec2, p: Vec2) -> f32 {
    let v01 = v1 - v0;
    let len_sq = dot(v01, v01);

    if len_sq == 0.0 {
        let d = p - v0;
        return dot(d, d);
    }

    let vp = p - v0;
    let t = dot(vp, v01) / len_sq;
    let a = v0 + v01 * t;
    let d = p - a;
    dot(d, d)
}

// -Vec2
impl std::ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2 { x: -self.x, y: -self.y }
    }
}

// Vec2 + Vec2
impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }
}

// Vec2 - Vec2
impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }
}

// Vec2 * f32
impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, scalar: f32) -> Vec2 {
        Vec2 { x: self.x * scalar, y: self.y * scalar }
    }
}

// f32 * Vec2
impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2 { x: vec.x * self, y: vec.y * self }
    }
}

// Vec2 / f32
impl std::ops::Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, scalar: f32) -> Vec2 {
        Vec2 { x: self.x / scalar, y: self.y / scalar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_creation_and_equality() {
        let v1 = Vec2 { x: 1.0, y: 2.0 };
        let v2 = Vec2 { x: 1.0, y: 2.0 };
        let v3 = Vec2 { x: 3.0, y: 4.0 };

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_eq!(v1.x, 1.0);
        assert_eq!(v1.y, 2.0);
    }

    #[test]
    fn test_vec2_negation() {
        let v = Vec2 { x: 2.0, y: -3.0 };
        let neg_v = -v;
        assert_eq!(neg_v.x, -2.0);
        assert_eq!(neg_v.y, 3.0);
    }

    #[test]
    fn test_vec2_addition() {
        let v1 = Vec2 { x: 1.0, y: 2.0 };
        let v2 = Vec2 { x: 3.0, y: 4.0 };
        let sum = v1 + v2;

        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);
    }

    #[test]
    fn test_vec2_subtraction() {
        let v1 = Vec2 { x: 5.0, y: 7.0 };
        let v2 = Vec2 { x: 2.0, y: 3.0 };
        let diff = v1 - v2;

        assert_eq!(diff.x, 3.0);
        assert_eq!(diff.y, 4.0);
    }

    #[test]
    fn test_vec2_multiplication_by_scalar() {
        let v = Vec2 { x: 2.0, y: 3.0 };
        let scaled = v * 2.0;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
    }

    #[test]
    fn test_scalar_multiplication_by_vec2() {
        let v = Vec2 { x: 2.0, y: 3.0 };
        let scaled = 2.0 * v;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
    }

    #[test]
    fn test_vec2_division_by_scalar() {
        let v = Vec2 { x: 4.0, y: 6.0 };
        let divided = v / 2.0;

        assert_eq!(divided.x, 2.0);
        assert_eq!(divided.y, 3.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vec2 { x: 1.0, y: 2.0 };
        let v2 = Vec2 { x: 3.0, y: 4.0 };
        let dot_result = dot(v1, v2);

        // 1.0 * 3.0 + 2.0 * 4.0 = 3.0 + 8.0 = 11.0
        assert_eq!(dot_result, 11.0);
    }

    #[test]
    fn test_length() {
        let v = Vec2 { x: 3.0, y: 4.0 };
        let length = v.length();

        // sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert_eq!(length, 5.0);
    }

    #[test]
    fn test_zero_vector_length() {
        let zero_vec = Vec2 { x: 0.0, y: 0.0 };
        let length = zero_vec.length();

        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_division() {
        {
            let v = Vec2 { x: 3.0, y: 4.0 };
            assert_eq!(v / 2.0, Vec2 { x: 1.5, y: 2.0 });
            assert_eq!(v / 0.5, Vec2 { x: 6.0, y: 8.0 });
        }
        {
            // division by zero
            let v = Vec2 { x: 1.0, y: 2.0 };
            let result = v / 0.0;

            // Division by zero for f32 results in infinity
            assert!(result.x.is_infinite());
            assert!(result.y.is_infinite());
        }
    }

    #[test]
    fn test_normalized() {
        let v = Vec2 { x: 3.0, y: 4.0 };
        let normalized = v.normalized();

        // The length of a normalized vector should be 1.0
        assert!((normalized.length() - 1.0).abs() < f32::EPSILON);

        // The direction should be preserved
        // For a vector (3,4) with length 5, the normalized vector should be (3/5, 4/5)
        assert!((normalized.x - 0.6).abs() < f32::EPSILON);
        assert!((normalized.y - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zero_vector_normalized() {
        let zero_vec = Vec2 { x: 0.0, y: 0.0 };
        let normalized = zero_vec.normalized();

        // Normalizing a zero vector should result in NaN values
        assert!(normalized.x.is_nan());
        assert!(normalized.y.is_nan());
    }

    #[test]
    fn test_distance() {
        // Test case 1: Point not on the line
        {
            let v0 = Vec2 { x: 0.0, y: 0.0 };
            let v1 = Vec2 { x: 3.0, y: 0.0 }; // Horizontal line along x-axis
            let p = Vec2 { x: 1.5, y: 2.0 }; // Point above the line

            // Distance should be 2.0 (vertical distance to the line)
            assert_eq!(distance(v0, v1, p), 2.0);
        }

        // Test case 2: Point on the line
        {
            let v0 = Vec2 { x: 0.0, y: 0.0 };
            let v1 = Vec2 { x: 4.0, y: 4.0 }; // Diagonal line
            let p = Vec2 { x: 2.0, y: 2.0 }; // Point on the line

            // Distance should be 0.0
            assert!((distance(v0, v1, p) - 0.0).abs() < f32::EPSILON);
        }

        // Test case 3: Zero-length line segment (point to point distance)
        {
            let v0 = Vec2 { x: 1.0, y: 1.0 };
            let v1 = Vec2 { x: 1.0, y: 1.0 }; // Same point
            let p = Vec2 { x: 4.0, y: 5.0 };

            // Distance should be the distance between points
            // sqrt((4-1)² + (5-1)²) = sqrt(9 + 16) = sqrt(25) = 5.0
            assert_eq!(distance(v0, v1, p), 5.0);
        }
    }

    #[test]
    fn test_distance2() {
        // Test case 1: Point not on the line
        {
            let v0 = Vec2 { x: 0.0, y: 0.0 };
            let v1 = Vec2 { x: 3.0, y: 0.0 }; // Horizontal line along x-axis
            let p = Vec2 { x: 1.5, y: 2.0 }; // Point above the line

            // Squared distance should be 4.0 (2.0²)
            assert_eq!(distance2(v0, v1, p), 4.0);
        }

        // Test case 2: Point on the line
        {
            let v0 = Vec2 { x: 0.0, y: 0.0 };
            let v1 = Vec2 { x: 4.0, y: 4.0 }; // Diagonal line
            let p = Vec2 { x: 2.0, y: 2.0 }; // Point on the line

            // Squared distance should be 0.0
            assert!((distance2(v0, v1, p) - 0.0).abs() < f32::EPSILON);
        }

        // Test case 3: Zero-length line segment (point to point squared distance)
        {
            let v0 = Vec2 { x: 1.0, y: 1.0 };
            let v1 = Vec2 { x: 1.0, y: 1.0 }; // Same point
            let p = Vec2 { x: 4.0, y: 5.0 };

            // Squared distance should be (4-1)² + (5-1)² = 9 + 16 = 25.0
            assert_eq!(distance2(v0, v1, p), 25.0);
        }
    }

    #[test]
    fn test_distance_and_distance2_comprehensive() {
        // Test cases converted from the provided C++ test cases
        #[derive(Debug)]
        struct TestCase {
            v0: Vec2,
            v1: Vec2,
            p: Vec2,
            expected_distance: f32,
            expected_distance2: f32,
        }

        let test_cases = [
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: 0.0, y: 0.0 },
                expected_distance: 0.0,
                expected_distance2: 0.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: 0.0, y: 5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: 0.0, y: -5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: 5.0, y: 5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: 5.0, y: -5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: -5.0, y: 5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 3.0, y: 0.0 },
                p: Vec2 { x: -5.0, y: -5.0 },
                expected_distance: 5.0,
                expected_distance2: 25.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 2.0 },
                v1: Vec2 { x: 3.0, y: 2.0 },
                p: Vec2 { x: 0.0, y: 0.0 },
                expected_distance: 2.0,
                expected_distance2: 4.0,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 2.0 },
                v1: Vec2 { x: 3.0, y: 2.0 },
                p: Vec2 { x: 0.0, y: 5.0 },
                expected_distance: 3.0,
                expected_distance2: 9.0,
            },
            TestCase {
                v0: Vec2 { x: 4.0, y: 1.0 },
                v1: Vec2 { x: 4.0, y: 5.0 },
                p: Vec2 { x: 4.0, y: 1.0 },
                expected_distance: 0.0,
                expected_distance2: 0.0,
            },
            TestCase {
                v0: Vec2 { x: 4.0, y: 1.0 },
                v1: Vec2 { x: 4.0, y: 5.0 },
                p: Vec2 { x: 4.0, y: 5.0 },
                expected_distance: 0.0,
                expected_distance2: 0.0,
            },
            TestCase {
                v0: Vec2 { x: 4.0, y: 1.0 },
                v1: Vec2 { x: 4.0, y: 5.0 },
                p: Vec2 { x: 2.0, y: 5.0 },
                expected_distance: 2.0,
                expected_distance2: 4.0,
            },
            TestCase {
                v0: Vec2 { x: 4.0, y: 1.0 },
                v1: Vec2 { x: 4.0, y: 5.0 },
                p: Vec2 { x: 6.0, y: 5.0 },
                expected_distance: 2.0,
                expected_distance2: 4.0,
            },
            TestCase {
                v0: Vec2 { x: 2.0, y: 3.0 },
                v1: Vec2 { x: 3.0, y: 4.0 },
                p: Vec2 { x: 3.0, y: 3.0 },
                expected_distance: 0.70710678,
                expected_distance2: 0.5,
            },
            TestCase {
                v0: Vec2 { x: 2.0, y: 3.0 },
                v1: Vec2 { x: 3.0, y: 4.0 },
                p: Vec2 { x: 2.0, y: 4.0 },
                expected_distance: 0.70710678,
                expected_distance2: 0.5,
            },
            TestCase {
                v0: Vec2 { x: 0.0, y: 0.0 },
                v1: Vec2 { x: 0.0, y: 0.0 },
                p: Vec2 { x: 1.0, y: 0.0 },
                expected_distance: 1.0,
                expected_distance2: 1.0,
            },
        ];

        for tc in test_cases.iter() {
            let actual_distance = distance(tc.v0, tc.v1, tc.p);
            let actual_distance2 = distance2(tc.v0, tc.v1, tc.p);

            // For floating point comparisons, we use a small epsilon
            assert!(
                (actual_distance - tc.expected_distance).abs() < 1e-6,
                "distance({:?}, {:?}, {:?}) = {}, expected {}",
                tc.v0,
                tc.v1,
                tc.p,
                actual_distance,
                tc.expected_distance
            );

            assert!(
                (actual_distance2 - tc.expected_distance2).abs() < 1e-6,
                "distance2({:?}, {:?}, {:?}) = {}, expected {}",
                tc.v0,
                tc.v1,
                tc.p,
                actual_distance2,
                tc.expected_distance2
            );
        }
    }
}

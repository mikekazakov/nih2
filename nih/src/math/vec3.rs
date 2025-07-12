use super::vec4::Vec4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn length(self) -> f32 {
        dot(self, self).sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let len = self.length();
        self / len
    }

    pub fn clamped(self, min: f32, max: f32) -> Vec3 {
        Vec3 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    pub fn as_vector4(self) -> Vec4 {
        Vec4 {x: self.x, y: self.y, z: self.z, w: 0.}
    }

    pub fn as_point4(self) -> Vec4 {
        Vec4 {x: self.x, y: self.y, z: self.z, w: 1.}
    }
}

// a * b
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

// a x b
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

// lerp(a, b, t)
pub fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3 {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
    }
}

// -Vec3
impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// Vec3 + Vec3
impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// Vec3 - Vec3
impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Vec3 * f32
impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// f32 * Vec3
impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: vec.x * self,
            y: vec.y * self,
            z: vec.z * self,
        }
    }
}

// Vec3 / f32
impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_creation_and_equality() {
        let v1 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v2 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v3 = Vec3 {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        };

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_eq!(v1.x, 1.0);
        assert_eq!(v1.y, 2.0);
        assert_eq!(v1.z, 3.0);
    }

    #[test]
    fn test_vec3_negation() {
        let v = Vec3 {
            x: 2.0,
            y: -3.0,
            z: 4.0,
        };
        let neg_v = -v;
        assert_eq!(neg_v.x, -2.0);
        assert_eq!(neg_v.y, 3.0);
        assert_eq!(neg_v.z, -4.0);
    }

    #[test]
    fn test_vec3_addition() {
        let v1 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v2 = Vec3 {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        };
        let sum = v1 + v2;

        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);
    }

    #[test]
    fn test_vec3_subtraction() {
        let v1 = Vec3 {
            x: 5.0,
            y: 7.0,
            z: 9.0,
        };
        let v2 = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let diff = v1 - v2;

        assert_eq!(diff.x, 3.0);
        assert_eq!(diff.y, 4.0);
        assert_eq!(diff.z, 5.0);
    }

    #[test]
    fn test_vec3_multiplication_by_scalar() {
        let v = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let scaled = v * 2.0;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
        assert_eq!(scaled.z, 8.0);
    }

    #[test]
    fn test_scalar_multiplication_by_vec3() {
        let v = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let scaled = 2.0 * v;

        assert_eq!(scaled.x, 4.0);
        assert_eq!(scaled.y, 6.0);
        assert_eq!(scaled.z, 8.0);
    }

    #[test]
    fn test_vec3_division_by_scalar() {
        let v = Vec3 {
            x: 4.0,
            y: 6.0,
            z: 8.0,
        };
        let divided = v / 2.0;

        assert_eq!(divided.x, 2.0);
        assert_eq!(divided.y, 3.0);
        assert_eq!(divided.z, 4.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v2 = Vec3 {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        };
        let dot_result = dot(v1, v2);

        // 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0 = 4.0 + 10.0 + 18.0 = 32.0
        assert_eq!(dot_result, 32.0);
    }

    #[test]
    fn test_cross_product() {
        let v1 = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }; // unit vector along x-axis
        let v2 = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }; // unit vector along y-axis
        let cross_result = cross(v1, v2);

        // cross-product of x and y unit vectors should be z unit vector
        assert_eq!(
            cross_result,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            }
        );

        // Test anti-commutativity: a × b = -(b × a)
        let cross_reverse = cross(v2, v1);
        assert_eq!(
            cross_reverse,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0
            }
        );
        assert_eq!(cross_reverse, -cross_result);

        // Test with non-unit vectors
        let v3 = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let v4 = Vec3 {
            x: 5.0,
            y: 6.0,
            z: 7.0,
        };
        let cross_result2 = cross(v3, v4);

        // (2,3,4) × (5,6,7) = (3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5)
        // = (21 - 24, 20 - 14, 12 - 15) = (-3, 6, -3)
        assert_eq!(
            cross_result2,
            Vec3 {
                x: -3.0,
                y: 6.0,
                z: -3.0
            }
        );
    }

    #[test]
    fn test_lerp() {
        let v1 = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let v2 = Vec3 {
            x: 10.0,
            y: 20.0,
            z: 30.0,
        };

        // t = 0 should return v1
        let lerp_result1 = lerp(v1, v2, 0.0);
        assert_eq!(lerp_result1, v1);

        // t = 1 should return v2
        let lerp_result2 = lerp(v1, v2, 1.0);
        assert_eq!(lerp_result2, v2);

        // t = 0.5 should return the midpoint
        let lerp_result3 = lerp(v1, v2, 0.5);
        assert_eq!(
            lerp_result3,
            Vec3 {
                x: 5.0,
                y: 10.0,
                z: 15.0
            }
        );

        // t = 0.25 should return 25% of the way from v1 to v2
        let lerp_result4 = lerp(v1, v2, 0.25);
        assert_eq!(
            lerp_result4,
            Vec3 {
                x: 2.5,
                y: 5.0,
                z: 7.5
            }
        );
    }

    #[test]
    fn test_length() {
        let v = Vec3 {
            x: 3.0,
            y: 4.0,
            z: 0.0,
        };
        let length = v.length();

        // sqrt(3² + 4² + 0²) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert_eq!(length, 5.0);

        let v2 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 2.0,
        };
        let length2 = v2.length();

        // sqrt(1² + 2² + 2²) = sqrt(1 + 4 + 4) = sqrt(9) = 3.0
        assert_eq!(length2, 3.0);
    }

    #[test]
    fn test_zero_vector_length() {
        let zero_vec = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let length = zero_vec.length();

        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_division() {
        {
            let v = Vec3 {
                x: 3.0,
                y: 4.0,
                z: 5.0,
            };
            assert_eq!(
                v / 2.0,
                Vec3 {
                    x: 1.5,
                    y: 2.0,
                    z: 2.5
                }
            );
            assert_eq!(
                v / 0.5,
                Vec3 {
                    x: 6.0,
                    y: 8.0,
                    z: 10.0
                }
            );
        }
        {
            // division by zero
            let v = Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            let result = v / 0.0;

            // Division by zero for f32 results in infinity
            assert!(result.x.is_infinite());
            assert!(result.y.is_infinite());
            assert!(result.z.is_infinite());
        }
    }

    #[test]
    fn test_normalized() {
        let v = Vec3 {
            x: 3.0,
            y: 4.0,
            z: 0.0,
        };
        let normalized = v.normalized();

        // The length of a normalized vector should be 1.0
        assert!((normalized.length() - 1.0).abs() < f32::EPSILON);

        // The direction should be preserved
        // For a vector (3,4,0) with length 5, the normalized vector should be (3/5, 4/5, 0)
        assert!((normalized.x - 0.6).abs() < f32::EPSILON);
        assert!((normalized.y - 0.8).abs() < f32::EPSILON);
        assert!((normalized.z - 0.0).abs() < f32::EPSILON);

        // Test with a different vector
        let v2 = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let normalized2 = v2.normalized();

        // Length should be 1.0
        assert!((normalized2.length() - 1.0).abs() < f32::EPSILON);

        // For a vector (1,1,1) with length sqrt(3), the normalized vector should be (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
        let expected = 1.0 / 3.0_f32.sqrt();
        assert!((normalized2.x - expected).abs() < f32::EPSILON);
        assert!((normalized2.y - expected).abs() < f32::EPSILON);
        assert!((normalized2.z - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zero_vector_normalized() {
        let zero_vec = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let normalized = zero_vec.normalized();

        // Normalizing a zero vector should result in NaN values
        assert!(normalized.x.is_nan());
        assert!(normalized.y.is_nan());
        assert!(normalized.z.is_nan());
    }

    #[test]
    fn test_clamped() {
        // Test clamping all components within range
        let v1 = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        let clamped1 = v1.clamped(1.0, 5.0);
        assert_eq!(clamped1, v1); // All components are within range

        // Test clamping components below minimum
        let v2 = Vec3 {
            x: -1.0,
            y: 0.5,
            z: 2.0,
        };
        let clamped2 = v2.clamped(1.0, 5.0);
        assert_eq!(
            clamped2,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 2.0
            }
        );

        // Test clamping components above maximum
        let v3 = Vec3 {
            x: 3.0,
            y: 6.0,
            z: 10.0,
        };
        let clamped3 = v3.clamped(1.0, 5.0);
        assert_eq!(
            clamped3,
            Vec3 {
                x: 3.0,
                y: 5.0,
                z: 5.0
            }
        );

        // Note: The Rust standard library's clamp function requires min <= max
        // and will panic if min > max, so we don't test that case.
    }

    #[test]
    fn test_as_vector4() {
        // Test normal vector
        let v1 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v4 = v1.as_vector4();

        // Check that x, y, z components match
        assert_eq!(v4.x, v1.x);
        assert_eq!(v4.y, v1.y);
        assert_eq!(v4.z, v1.z);
        // Check that w component is 0.0 (vector representation)
        assert_eq!(v4.w, 0.0);

        // Test zero vector
        let zero_vec = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let zero_vec4 = zero_vec.as_vector4();

        assert_eq!(zero_vec4.x, 0.0);
        assert_eq!(zero_vec4.y, 0.0);
        assert_eq!(zero_vec4.z, 0.0);
        assert_eq!(zero_vec4.w, 0.0);

        // Test negative values
        let neg_vec = Vec3 {
            x: -1.0,
            y: -2.0,
            z: -3.0,
        };
        let neg_vec4 = neg_vec.as_vector4();

        assert_eq!(neg_vec4.x, -1.0);
        assert_eq!(neg_vec4.y, -2.0);
        assert_eq!(neg_vec4.z, -3.0);
        assert_eq!(neg_vec4.w, 0.0);
    }

    #[test]
    fn test_as_point4() {
        // Test normal vector
        let v1 = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v4 = v1.as_point4();

        // Check that x, y, z components match
        assert_eq!(v4.x, v1.x);
        assert_eq!(v4.y, v1.y);
        assert_eq!(v4.z, v1.z);
        // Check that w component is 1.0 (point representation)
        assert_eq!(v4.w, 1.0);

        // Test zero vector
        let zero_vec = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let zero_vec4 = zero_vec.as_point4();

        assert_eq!(zero_vec4.x, 0.0);
        assert_eq!(zero_vec4.y, 0.0);
        assert_eq!(zero_vec4.z, 0.0);
        assert_eq!(zero_vec4.w, 1.0);

        // Test negative values
        let neg_vec = Vec3 {
            x: -1.0,
            y: -2.0,
            z: -3.0,
        };
        let neg_vec4 = neg_vec.as_point4();

        assert_eq!(neg_vec4.x, -1.0);
        assert_eq!(neg_vec4.y, -2.0);
        assert_eq!(neg_vec4.z, -3.0);
        assert_eq!(neg_vec4.w, 1.0);
    }
}

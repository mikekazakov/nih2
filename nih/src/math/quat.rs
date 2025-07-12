use super::vec3::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Quat {
    pub fn identity() -> Quat {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Quat {
        let half = angle * 0.5;
        let sin = half.sin();
        let cos = half.cos();

        Self {
            x: axis.x * sin,
            y: axis.y * sin,
            z: axis.z * sin,
            w: cos,
        }
    }

    pub fn inverse(self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    pub fn from_look_rotation(forward: Vec3, up: Vec3) -> Quat {
        let f = forward.normalized();
        let r = cross(up, f).normalized(); // right = up × forward
        let u = cross(f, r); // real up = forward × right

        // Rotation matrix columns: r, u, f
        let m00 = r.x;
        let m01 = u.x;
        let m02 = f.x;
        let m10 = r.y;
        let m11 = u.y;
        let m12 = f.y;
        let m20 = r.z;
        let m21 = u.z;
        let m22 = f.z;

        let trace = m00 + m11 + m22;

        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat {
                w: 0.25 * s,
                x: (m21 - m12) / s,
                y: (m02 - m20) / s,
                z: (m10 - m01) / s,
            }
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Quat {
                w: (m21 - m12) / s,
                x: 0.25 * s,
                y: (m01 + m10) / s,
                z: (m02 + m20) / s,
            }
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Quat {
                w: (m02 - m20) / s,
                x: (m01 + m10) / s,
                y: 0.25 * s,
                z: (m12 + m21) / s,
            }
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Quat {
                w: (m10 - m01) / s,
                x: (m02 + m20) / s,
                y: (m12 + m21) / s,
                z: 0.25 * s,
            }
        }
    }

    pub fn normalized(self) -> Quat {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len == 0.0 {
            Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            } // Identity fallback
        } else {
            let inv_len = 1.0 / len;
            Quat {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                w: self.w * inv_len,
            }
        }
    }
}

impl std::ops::Mul for Quat {
    type Output = Quat;

    fn mul(self, rhs: Quat) -> Quat {
        Quat {
            x: self.w * rhs.x + rhs.w * self.x + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y + rhs.w * self.y + self.z * rhs.x - self.x * rhs.z,
            z: self.w * rhs.z + rhs.w * self.z + self.x * rhs.y - self.y * rhs.x,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

impl std::ops::Mul<Vec3> for Quat {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        let tx = 2.0 * (self.y * v.z - self.z * v.y);
        let ty = 2.0 * (self.z * v.x - self.x * v.z);
        let tz = 2.0 * (self.x * v.y - self.y * v.x);

        let rx = v.x + self.w * tx + (self.y * tz - self.z * ty);
        let ry = v.y + self.w * ty + (self.z * tx - self.x * tz);
        let rz = v.z + self.w * tz + (self.x * ty - self.y * tx);

        Vec3 {
            x: rx,
            y: ry,
            z: rz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Constants to match the C++ tests
    const PI_2: f32 = PI / 2.0;
    const PI_4: f32 = PI / 4.0;
    const PI_3: f32 = PI / 3.0;

    // Helper struct for approximate quaternion comparison (similar to C++ QuatApprox)
    #[derive(Debug)]
    struct QuatApprox(Quat);

    impl PartialEq<QuatApprox> for Quat {
        fn eq(&self, other: &QuatApprox) -> bool {
            const EPSILON: f32 = 1e-4;
            (self.x - other.0.x).abs() < EPSILON
                && (self.y - other.0.y).abs() < EPSILON
                && (self.z - other.0.z).abs() < EPSILON
                && (self.w - other.0.w).abs() < EPSILON
        }
    }

    // Helper struct for approximate vector comparison (similar to C++ Vec3Approx)
    #[derive(Debug)]
    struct Vec3Approx(Vec3);

    impl PartialEq<Vec3Approx> for Vec3 {
        fn eq(&self, other: &Vec3Approx) -> bool {
            const EPSILON: f32 = 1e-4;
            (self.x - other.0.x).abs() < EPSILON
                && (self.y - other.0.y).abs() < EPSILON
                && (self.z - other.0.z).abs() < EPSILON
        }
    }

    #[test]
    fn test_quat_creation_and_equality() {
        let q1 = Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let q2 = Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let q3 = Quat {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };

        assert_eq!(q1, q2);
        assert_ne!(q1, q3);
        assert_eq!(q1.x, 0.0);
        assert_eq!(q1.y, 0.0);
        assert_eq!(q1.z, 0.0);
        assert_eq!(q1.w, 1.0);
    }

    #[test]
    fn test_identity() {
        let identity = Quat::identity();

        // Identity quaternion should be (0, 0, 0, 1)
        assert_eq!(identity.x, 0.0);
        assert_eq!(identity.y, 0.0);
        assert_eq!(identity.z, 0.0);
        assert_eq!(identity.w, 1.0);

        // Multiplying any quaternion by identity should return the original quaternion
        let q = Quat {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            w: 0.5,
        };
        let result = identity * q;
        assert_eq!(result, q);

        let result2 = q * identity;
        assert_eq!(result2, q);
    }

    #[test]
    fn test_from_axis_angle() {
        // Test rotation around x-axis by 90 degrees
        let axis_x = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let angle_90 = PI / 2.0;
        let q_x_90 = Quat::from_axis_angle(axis_x, angle_90);

        // For 90 degrees around x-axis, quaternion should be (sin(45°), 0, 0, cos(45°))
        let expected_sin = (PI / 4.0).sin();
        let expected_cos = (PI / 4.0).cos();
        assert!((q_x_90.x - expected_sin).abs() < 1e-6);
        assert!((q_x_90.y - 0.0).abs() < 1e-6);
        assert!((q_x_90.z - 0.0).abs() < 1e-6);
        assert!((q_x_90.w - expected_cos).abs() < 1e-6);

        // Test rotation around y-axis by 180 degrees
        let axis_y = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let angle_180 = PI;
        let q_y_180 = Quat::from_axis_angle(axis_y, angle_180);

        // For 180 degrees around y-axis, quaternion should be (0, sin(90°), 0, cos(90°))
        assert!((q_y_180.x - 0.0).abs() < 1e-6);
        assert!((q_y_180.y - 1.0).abs() < 1e-6);
        assert!((q_y_180.z - 0.0).abs() < 1e-6);
        assert!((q_y_180.w - 0.0).abs() < 1e-6);

        // Test with non-unit axis
        let non_unit_axis = Vec3 {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        };
        let q_non_unit = Quat::from_axis_angle(non_unit_axis, angle_90);

        // The function should normalize the axis, so the result should be the same as with a unit axis
        assert!((q_non_unit.x - expected_sin * 2.0).abs() < 1e-6);
        assert!((q_non_unit.y - 0.0).abs() < 1e-6);
        assert!((q_non_unit.z - 0.0).abs() < 1e-6);
        assert!((q_non_unit.w - expected_cos).abs() < 1e-6);
    }

    #[test]
    fn test_from_axis_angle_cpp_cases() {
        // Test cases from the C++ implementation
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                0.0
            ),
            QuatApprox(Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0
            })
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                0.0
            ),
            QuatApprox(Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0
            })
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                0.0
            ),
            QuatApprox(Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0
            })
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                PI_2
            ),
            QuatApprox(Quat {
                x: 0.7071,
                y: 0.0,
                z: 0.0,
                w: 0.7071
            })
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                PI
            ),
            QuatApprox(Quat {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                w: 0.0
            })
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                PI_4
            ),
            QuatApprox(Quat {
                x: 0.0,
                y: 0.0,
                z: 0.38268343,
                w: 0.92387953
            })
        );

        // Normalized vector (1/√3, 1/√3, 1/√3)
        let normalized_vec = Vec3 {
            x: 0.57735027,
            y: 0.57735027,
            z: 0.57735027,
        };
        assert_eq!(
            Quat::from_axis_angle(normalized_vec, PI_2),
            QuatApprox(Quat {
                x: 0.40824829,
                y: 0.40824829,
                z: 0.40824829,
                w: 0.70710678
            })
        );
    }

    #[test]
    fn test_inverse() {
        // Test inverse of identity
        let identity = Quat::identity();
        let inv_identity = identity.inverse();
        assert_eq!(inv_identity, identity);

        // Test inverse of arbitrary quaternion
        let q = Quat {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            w: 0.5,
        };
        let inv_q = q.inverse();

        // For a unit quaternion, the inverse is the conjugate (negated x, y, z components)
        assert_eq!(inv_q.x, -q.x);
        assert_eq!(inv_q.y, -q.y);
        assert_eq!(inv_q.z, -q.z);
        assert_eq!(inv_q.w, q.w);

        // q * q^-1 should be identity
        let result = q * inv_q;
        assert!((result.x - 0.0).abs() < 1e-6);
        assert!((result.y - 0.0).abs() < 1e-6);
        assert!((result.z - 0.0).abs() < 1e-6);
        assert!((result.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_look_rotation() {
        // Test looking along positive z-axis with up as positive y-axis
        let forward = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let up = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let q = Quat::from_look_rotation(forward, up);

        // This should be identity quaternion (no rotation needed)
        assert!((q.x - 0.0).abs() < 1e-6);
        assert!((q.y - 0.0).abs() < 1e-6);
        assert!((q.z - 0.0).abs() < 1e-6);
        assert!((q.w - 1.0).abs() < 1e-6);

        // Test looking along negative z-axis (180 degree rotation around y-axis)
        let backward = Vec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        };
        let q_back = Quat::from_look_rotation(backward, up);

        // Rotating a vector along positive z by this quaternion should give negative z
        let rotated = q_back * forward;
        assert!((rotated.x - 0.0).abs() < 1e-6);
        assert!((rotated.y - 0.0).abs() < 1e-6);
        assert!((rotated.z + 1.0).abs() < 1e-6);

        // Test looking along positive x-axis (90 degree rotation around y-axis)
        let right = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let q_right = Quat::from_look_rotation(right, up);

        // Rotating a vector along positive z by this quaternion should give positive x
        let rotated = q_right * forward;
        assert!((rotated.x - 1.0).abs() < 1e-6);
        assert!((rotated.y - 0.0).abs() < 1e-6);
        assert!((rotated.z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_look_rotation_cpp_cases() {
        // Test cases from the C++ implementation
        let forward = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let up = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };

        // Identity (looking forward with up as +Y)
        assert_eq!(
            Quat::from_look_rotation(forward, up),
            QuatApprox(Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0
            })
        );

        // Look right (+X)
        let right = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let q_right = Quat::from_look_rotation(right, up);
        assert_eq!(
            q_right * forward,
            Vec3Approx(Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0
            })
        );

        // Look up (+Y), using +Z as up to avoid zero cross
        let up_dir = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let z_up = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let q_up = Quat::from_look_rotation(up_dir, z_up);
        assert_eq!(
            q_up * forward,
            Vec3Approx(Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0
            })
        );

        // Look backward (-Z)
        let backward = Vec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        };
        let q_back = Quat::from_look_rotation(backward, up);
        assert_eq!(
            q_back * forward,
            Vec3Approx(Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0
            })
        );

        // Arbitrary direction
        let arbitrary = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let q_arbitrary = Quat::from_look_rotation(arbitrary, up);
        assert_eq!(
            q_arbitrary * forward,
            Vec3Approx(Vec3 {
                x: 0.57735027,
                y: 0.57735027,
                z: 0.57735027
            })
        );
    }

    #[test]
    fn test_normalized() {
        // Test normalizing an already normalized quaternion
        let q = Quat::identity();
        let normalized = q.normalized();
        assert_eq!(normalized, q);

        // Test normalizing a non-unit quaternion
        let q2 = Quat {
            x: 2.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
        let normalized2 = q2.normalized();

        // Length of normalized quaternion should be 1.0
        let length = (normalized2.x * normalized2.x
            + normalized2.y * normalized2.y
            + normalized2.z * normalized2.z
            + normalized2.w * normalized2.w)
            .sqrt();
        assert!((length - 1.0).abs() < 1e-6);

        // Direction should be preserved
        assert_eq!(normalized2.x, 1.0);
        assert_eq!(normalized2.y, 0.0);
        assert_eq!(normalized2.z, 0.0);
        assert_eq!(normalized2.w, 0.0);

        // Test normalizing a zero quaternion (should return identity)
        let zero_q = Quat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        };
        let normalized_zero = zero_q.normalized();
        assert_eq!(normalized_zero, Quat::identity());
    }

    #[test]
    fn test_quat_multiplication() {
        // Test multiplication of identity quaternions
        let identity = Quat::identity();
        let result = identity * identity;
        assert_eq!(result, identity);

        // Test multiplication of arbitrary quaternions
        // Rotation around x-axis by 90 degrees
        let q_x_90 = Quat::from_axis_angle(
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            PI / 2.0,
        );
        // Rotation around y-axis by 90 degrees
        let q_y_90 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            PI / 2.0,
        );

        // Applying rotations in sequence: first around x, then around y
        let combined = q_y_90 * q_x_90;

        // Test the combined rotation on a vector
        let v = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let rotated = combined * v;

        // Verify the rotation step by step:
        // First, rotate v around x-axis by 90 degrees
        let v_rotated_x = q_x_90 * v;

        // When rotating (0,0,1) 90° around x-axis, we get (0,-1,0)
        // This follows the right-hand rule for rotations
        assert!((v_rotated_x.x - 0.0).abs() < 1e-6);
        assert!((v_rotated_x.y + 1.0).abs() < 1e-6);
        assert!((v_rotated_x.z - 0.0).abs() < 1e-6);

        // Then, rotate the result around y-axis by 90 degrees
        let v_rotated_xy = q_y_90 * v_rotated_x;

        // The y-component remains unchanged when rotating around y-axis
        assert!((v_rotated_xy.y + 1.0).abs() < 1e-6);

        // The combined rotation should match the step-by-step rotation
        assert!((rotated.x - v_rotated_xy.x).abs() < 1e-6);
        assert!((rotated.y - v_rotated_xy.y).abs() < 1e-6);
        assert!((rotated.z - v_rotated_xy.z).abs() < 1e-6);
    }

    #[test]
    fn test_quat_multiplication_cpp_cases() {
        // Test cases from the C++ implementation
        let identity = Quat::identity();

        // Identity multiplication
        assert_eq!(
            identity
                * Quat::from_axis_angle(
                    Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0
                    },
                    PI_4
                ),
            QuatApprox(Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                PI_4
            ))
        );

        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                PI_4
            ) * identity,
            QuatApprox(Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                PI_4
            ))
        );

        // Non-commutativity test
        let q_x = Quat::from_axis_angle(
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            PI_2,
        );
        let q_y = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            PI_2,
        );
        assert_ne!(q_x * q_y, QuatApprox(q_y * q_x));

        // Rotation around same axis is additive
        let q_z_1 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            PI_2,
        );
        let q_z_2 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            PI_2,
        );
        assert_eq!(
            q_z_1 * q_z_2,
            QuatApprox(Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                PI
            ))
        );

        // Associativity test
        let q_x_45 = Quat::from_axis_angle(
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            PI_4,
        );
        let q_y_45 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            PI_4,
        );
        let q_z_45 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            PI_4,
        );

        assert_eq!(
            (q_x_45 * q_y_45) * q_z_45,
            QuatApprox(q_x_45 * (q_y_45 * q_z_45))
        );
    }

    #[test]
    fn test_quat_vec_multiplication() {
        // Test rotation of a vector by identity quaternion (no rotation)
        let identity = Quat::identity();
        let v = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let rotated = identity * v;
        assert_eq!(rotated, v);

        // Test rotation around x-axis by 90 degrees
        let q_x_90 = Quat::from_axis_angle(
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            PI / 2.0,
        );
        let v2 = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let rotated2 = q_x_90 * v2;

        // y-axis rotated 90° around x-axis becomes negative z-axis
        // This follows the right-hand rule for rotations
        assert!((rotated2.x - 0.0).abs() < 1e-6);
        assert!((rotated2.y - 0.0).abs() < 1e-6);
        assert!((rotated2.z - 1.0).abs() < 1e-6);

        // Test rotation around y-axis by 90 degrees
        let q_y_90 = Quat::from_axis_angle(
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            PI / 2.0,
        );
        let v3 = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let rotated3 = q_y_90 * v3;

        // z-axis rotated 90° around y-axis becomes positive x-axis
        // This follows the right-hand rule for rotations
        assert!((rotated3.x - 1.0).abs() < 1e-6);
        assert!((rotated3.y - 0.0).abs() < 1e-6);
        assert!((rotated3.z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_quat_vec_multiplication_cpp_cases() {
        // Test cases from the C++ implementation

        // Rotation around y-axis by 90 degrees
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                PI_2
            ) * Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0
            })
        );

        // Rotation around x-axis by 180 degrees
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                PI
            ) * Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: -1.0,
                z: 0.0
            })
        );

        // Identity rotation
        assert_eq!(
            Quat::identity()
                * Vec3 {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0
                },
            Vec3Approx(Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            })
        );

        // Rotation around x-axis by 90 degrees
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                PI_2
            ) * Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: -1.0,
                z: 0.0
            })
        );

        // Rotation around z-axis by 90 degrees
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                PI_2
            ) * Vec3 {
                x: 1.0,
                y: 1.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: -1.0,
                y: 1.0,
                z: 0.0
            })
        );

        // Rotation around z-axis by 90 degrees (another vector)
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                PI_2
            ) * Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0
            })
        );

        // Rotation around z-axis by 45 degrees
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0
                },
                PI_4
            ) * Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: 0.7071067811865475,
                y: 0.7071067811865475,
                z: 0.0
            })
        );

        // Rotation around x-axis by 120 degrees
        let two_pi_3 = 2.0 * PI_3;
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0
                },
                two_pi_3
            ) * Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: -0.5,
                z: 0.8660254037844386
            })
        );

        // Rotation around y-axis by 360 degrees (full circle)
        assert_eq!(
            Quat::from_axis_angle(
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0
                },
                2.0 * PI
            ) * Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            },
            Vec3Approx(Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            })
        );

        // Rotation around normalized vector (1/√2, 1/√2, 0) by 180 degrees
        let normalized_vec = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        }
        .normalized();
        assert_eq!(
            Quat::from_axis_angle(normalized_vec, PI)
                * Vec3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0
                },
            Vec3Approx(Vec3 {
                x: 1.0,
                y: 1.0,
                z: -1.0
            })
        );
    }
}

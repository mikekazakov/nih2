use super::*;
use std::f32::consts::PI;

pub fn sphere_to_aa_lines(nsegments: i32) -> Vec<Vec3> {
    if nsegments < 4 {
        return Vec::new();
    }

    let dphi = 2.0 * PI / nsegments as f32;
    let mut points = Vec::with_capacity((nsegments * 6) as usize);

    for i in 0..nsegments {
        let phi0 = dphi * i as f32;
        let phi1 = dphi * (i + 1) as f32;

        // XY plane
        points.push(Vec3::new(phi0.cos(), phi0.sin(), 0.0));
        points.push(Vec3::new(phi1.cos(), phi1.sin(), 0.0));

        // XZ plane
        points.push(Vec3::new(phi0.cos(), 0.0, phi0.sin()));
        points.push(Vec3::new(phi1.cos(), 0.0, phi1.sin()));

        // YZ plane
        points.push(Vec3::new(0.0, phi0.cos(), phi0.sin()));
        points.push(Vec3::new(0.0, phi1.cos(), phi1.sin()));
    }

    points
}

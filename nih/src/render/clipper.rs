use super::vertex::Vertex;
use crate::math::*;
use arrayvec::ArrayVec;
use std::mem::swap;

pub fn clip_triangle(input_vertices: &[Vertex; 3]) -> ArrayVec<Vertex, 7> {
    const CLIP_PLANES: [Vec4; 6] = [
        Vec4::new(1.0, 0.0, 0.0, 1.0),  // Left
        Vec4::new(-1.0, 0.0, 0.0, 1.0), // Right
        Vec4::new(0.0, 1.0, 0.0, 1.0),  // Bottom
        Vec4::new(0.0, -1.0, 0.0, 1.0), // Top
        Vec4::new(0.0, 0.0, 1.0, 1.0),  // Near
        Vec4::new(0.0, 0.0, -1.0, 1.0), // Far
    ];
    let mut buffer_b: [Vertex; 7] = [Vertex::default(); 7];
    let mut buffer_a: [Vertex; 7] = [Vertex::default(); 7];
    buffer_a[..3].clone_from_slice(input_vertices);
    let mut input = &mut buffer_a;
    let mut output = &mut buffer_b;

    let mut in_count = 3;

    for &plane in &CLIP_PLANES {
        if in_count == 0 {
            break;
        }
        let mut out_count = 0;
        let mut v0 = input[in_count - 1];
        let mut d0 = dot(v0.position, plane);

        for i in 0..in_count {
            let v1 = input[i];
            let d1 = dot(v1.position, plane);
            let inside0 = d0 >= 0.0;
            let inside1 = d1 >= 0.0;
            if inside0 && inside1 {
                output[out_count] = v1;
                out_count += 1;
            } else if inside0 && !inside1 {
                let t = d0 / (d0 - d1);
                output[out_count] = interpolate_vertex(&v0, &v1, t);
                out_count += 1;
            } else if !inside0 && inside1 {
                let t = d0 / (d0 - d1);
                output[out_count] = interpolate_vertex(&v0, &v1, t);
                out_count += 1;
                output[out_count] = v1;
                out_count += 1;
            }
            v0 = v1;
            d0 = d1;
        }

        swap(&mut input, &mut output);
        in_count = out_count;
    }

    ArrayVec::from_iter(input[..in_count].iter().copied())
}

fn interpolate_vertex(v0: &Vertex, v1: &Vertex, t: f32) -> Vertex {
    let t1 = 1.0 - t;
    Vertex {
        position: t1 * v0.position + t * v1.position,
        world_position: t1 * v0.world_position + t * v1.world_position,
        normal: t1 * v0.normal + t * v1.normal,
        color: t1 * v0.color + t * v1.color,
        tex_coord: t1 * v0.tex_coord + t * v1.tex_coord,
    }
}

pub fn clip_line(input_points: &[Vec4; 2]) -> ArrayVec<Vec4, 2> {
    const CLIP_PLANES: [Vec4; 6] = [
        Vec4::new(1.0, 0.0, 0.0, 1.0),  // Left
        Vec4::new(-1.0, 0.0, 0.0, 1.0), // Right
        Vec4::new(0.0, 1.0, 0.0, 1.0),  // Bottom
        Vec4::new(0.0, -1.0, 0.0, 1.0), // Top
        Vec4::new(0.0, 0.0, 1.0, 1.0),  // Near
        Vec4::new(0.0, 0.0, -1.0, 1.0), // Far
    ];
    let mut p0 = input_points[0];
    let mut p1 = input_points[1];
    for &plane in &CLIP_PLANES {
        let d0 = dot(p0, plane);
        let d1 = dot(p1, plane);
        let inside0 = d0 >= 0.0;
        let inside1 = d1 >= 0.0;
        if !inside0 && !inside1 {
            return ArrayVec::new();
        } else if inside0 && inside1 {
            continue;
        } else {
            let t = d0 / (d0 - d1);
            let clipped = (1.0 - t) * p0 + t * p1;
            if !inside0 {
                p0 = clipped;
            } else {
                p1 = clipped;
            }
        }
    }
    ArrayVec::from([p0, p1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_triangle() {
        #[derive(Debug)]
        struct TestCase {
            name: &'static str,
            input: [Vertex; 3],
            expected: Vec<Vec4>,
        }

        let test_cases = [
            TestCase {
                name: "No clipping",
                input: [
                    Vertex { position: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(0.5, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(0.0, 0.5, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                    Vec4::new(0.5, 0.0, 0.0, 1.0),
                    Vec4::new(0.0, 0.5, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Fully outside left plane",
                input: [
                    Vertex { position: Vec4::new(-2.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(-1.5, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(-1.1, 0.5, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![],
            },
            TestCase {
                name: "Partial clipping against left plane",
                input: [
                    Vertex { position: Vec4::new(-2.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(0.0, -1.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(-1.0, 0.0, 0.0, 1.0),
                    Vec4::new(-1.0, -0.5, 0.0, 1.0),
                    Vec4::new(0.0, -1.0, 0.0, 1.0),
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against top plane",
                input: [
                    Vertex { position: Vec4::new(0.0, -2.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(1.0, -2.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(1.0, 0.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(0.5, -1.0, 0.0, 1.0),
                    Vec4::new(1.0, -1.0, 0.0, 1.0),
                    Vec4::new(1.0, 0.0, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against right plane",
                input: [
                    Vertex { position: Vec4::new(0.0, 1.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(2.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(2.0, 1.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(1.0, 1.0, 0.0, 1.0),
                    Vec4::new(0.0, 1.0, 0.0, 1.0),
                    Vec4::new(1.0, 0.5, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against bottom plane",
                input: [
                    Vertex { position: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(1.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(1.0, 2.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(0.5, 1.0, 0.0, 1.0),
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                    Vec4::new(1.0, 0.0, 0.0, 1.0),
                    Vec4::new(1.0, 1.0, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against right and bottom planes",
                input: [
                    Vertex { position: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(3.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(0.0, 3.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(0.0, 1.0, 0.0, 1.0),
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                    Vec4::new(1.0, 0.0, 0.0, 1.0),
                    Vec4::new(1.0, 1.0, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against left and right planes",
                input: [
                    Vertex { position: Vec4::new(-2.0, 0.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(2.0, -1.0, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(2.0, 1.0, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(1.0, 0.75, 0.0, 1.0),
                    Vec4::new(-1.0, 0.25, 0.0, 1.0),
                    Vec4::new(-1.0, -0.25, 0.0, 1.0),
                    Vec4::new(1.0, -0.75, 0.0, 1.0),
                ],
            },
            TestCase {
                name: "Partial clipping against left, bottom and right planes",
                input: [
                    Vertex { position: Vec4::new(0.0, 1.2, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(2.0, -0.8, 0.0, 1.0), ..Default::default() },
                    Vertex { position: Vec4::new(-2.0, -0.8, 0.0, 1.0), ..Default::default() },
                ],
                expected: vec![
                    Vec4::new(-1.0, 0.2, 0.0, 1.0),
                    Vec4::new(-0.2, 1.0, 0.0, 1.0),
                    Vec4::new(0.2, 1.0, 0.0, 1.0),
                    Vec4::new(1.0, 0.2, 0.0, 1.0),
                    Vec4::new(1.0, -0.8, 0.0, 1.0),
                    Vec4::new(-1.0, -0.8, 0.0, 1.0),
                ],
            },
        ];

        for case in &test_cases {
            let result = clip_triangle(&case.input);

            assert_eq!(result.len(), case.expected.len(), "Vertex count mismatch in test: {}", case.name);

            for (actual, expected) in result.iter().zip(&case.expected) {
                let delta = actual.position - *expected;
                let epsilon = 1e-5;
                assert!(
                    delta.x.abs() < epsilon
                        && delta.y.abs() < epsilon
                        && delta.z.abs() < epsilon
                        && delta.w.abs() < epsilon,
                    "Vertex mismatch in test {}: got {:?}, expected {:?}",
                    case.name,
                    actual.position,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_clip_line_cases() {
        #[derive(Debug)]
        struct TestCase {
            name: &'static str,
            input: [Vec4; 2],
            expected: Vec<Vec4>,
        }

        let test_cases = [
            TestCase {
                name: "Fully inside",
                input: [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.5, 0.5, 0.0, 1.0)],
                expected: Vec::from([Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.5, 0.5, 0.0, 1.0)]),
            },
            TestCase {
                name: "Fully outside (left)",
                input: [Vec4::new(-2.0, 0.0, 0.0, 1.0), Vec4::new(-1.5, 0.0, 0.0, 1.0)],
                expected: Vec::new(),
            },
            TestCase {
                name: "Partially clipped (left to inside)",
                input: [Vec4::new(-2.0, 0.0, 0.0, 1.0), Vec4::new(0.5, 0.0, 0.0, 1.0)],
                expected: Vec::from([Vec4::new(-1.0, 0.0, 0.0, 1.0), Vec4::new(0.5, 0.0, 0.0, 1.0)]),
            },
            TestCase {
                name: "Partially clipped (inside to top)",
                input: [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 2.0, 0.0, 1.0)],
                expected: Vec::from([Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0)]),
            },
            TestCase {
                name: "Clipped on both ends",
                input: [Vec4::new(-2.0, -2.0, 0.0, 1.0), Vec4::new(2.0, 2.0, 0.0, 1.0)],
                expected: Vec::from([Vec4::new(-1.0, -1.0, 0.0, 1.0), Vec4::new(1.0, 1.0, 0.0, 1.0)]),
            },
            TestCase {
                name: "Clipped against near and far planes",
                input: [Vec4::new(0.0, 0.0, -2.0, 1.0), Vec4::new(0.0, 0.0, 2.0, 1.0)],
                expected: Vec::from([Vec4::new(0.0, 0.0, -1.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)]),
            },
            TestCase {
                name: "Diagonal across entire clip space",
                input: [Vec4::new(-2.0, -2.0, -2.0, 1.0), Vec4::new(2.0, 2.0, 2.0, 1.0)],
                expected: Vec::from([Vec4::new(-1.0, -1.0, -1.0, 1.0), Vec4::new(1.0, 1.0, 1.0, 1.0)]),
            },
            TestCase {
                name: "Inside to outside (bottom)",
                input: [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.0, -2.0, 0.0, 1.0)],
                expected: Vec::from([Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(0.0, -1.0, 0.0, 1.0)]),
            },
            TestCase {
                name: "Crossing only far plane",
                input: [Vec4::new(0.0, 0.0, 0.5, 1.0), Vec4::new(0.0, 0.0, 2.0, 1.0)],
                expected: Vec::from([Vec4::new(0.0, 0.0, 0.5, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)]),
            },
            TestCase {
                name: "Line entirely behind far plane",
                input: [Vec4::new(0.0, 0.0, 2.1, 1.0), Vec4::new(0.5, 0.5, 2.2, 1.0)],
                expected: Vec::new(),
            },
        ];

        for case in &test_cases {
            let result = clip_line(&case.input);

            assert_eq!(result.len(), case.expected.len(), "Point count mismatch in test: {}", case.name);

            for (actual, expected) in result.iter().zip(&case.expected) {
                let delta = *actual - *expected;
                let epsilon = 1e-5;
                assert!(
                    delta.x.abs() < epsilon
                        && delta.y.abs() < epsilon
                        && delta.z.abs() < epsilon
                        && delta.w.abs() < epsilon,
                    "Point mismatch in test {}: got {:?}, expected {:?}",
                    case.name,
                    actual,
                    expected
                );
            }
        }
    }
}

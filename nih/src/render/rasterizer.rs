use super::super::math::*;
use super::*;
// use arrayvec::ArrayVec;
// use std::mem::swap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    /// No culling — all triangles are rendered.
    None,

    /// Cull clockwise-wound triangles.
    CW,

    /// Cull counter-clockwise-wound triangles.
    CCW,
}

#[derive(Debug, Clone, Copy)]
pub struct RasterizationCommand<'a> {
    pub world_positions: &'a [Vec3],
    pub normals: &'a [Vec3],    // empty if absent, will be derived automatically
    pub tex_coords: &'a [Vec2], // empty if absent
    pub colors: &'a [Vec4],     // empty if absent, .color will be used

    /// Triangle indices: [t0v0, t0v1, t0v2, t1v0, t1v1, t1v2, ...].
    /// Optional, monotonic indices to cover all world positions will be assumed if none is provided
    pub indices: &'a [u32],
    pub model: Mat34,
    pub view: Mat44,
    pub projection: Mat44,
    pub culling: CullMode,
    pub color: Vec4,
}

pub struct Rasterizer {
    viewport: Viewport,
    viewport_scale: ViewportScale,
    vertices: Vec<Vertex>,
}

impl Rasterizer {
    pub fn new() -> Self {
        return Rasterizer {
            viewport: Viewport::new(0, 0, 1, 1), //
            viewport_scale: ViewportScale::default(),
            vertices: Vec::new(), //
        };
    }

    pub fn setup(&mut self, viewport: Viewport) {
        assert!(viewport.xmax > viewport.xmin);
        assert!(viewport.ymax > viewport.ymin);
        self.viewport = viewport;
        self.viewport_scale = ViewportScale::new(viewport);
        self.vertices.clear();
    }

    pub fn commit(&mut self, command: &RasterizationCommand) {
        let use_explicit_indices = !command.indices.is_empty();
        let input_triangles_num = if use_explicit_indices {
            command.indices.len() / 3
        } else {
            command.world_positions.len() / 3
        };

        if input_triangles_num == 0 {
            return;
        }

        let view_projection = command.projection * command.view;
        let normal_matrix = command.view.as_mat33() * command.model.as_mat33().inverse();
        let viewport_scale = self.viewport_scale;

        for i in 0..input_triangles_num {
            let index = |n: usize| {
                if use_explicit_indices {
                    command.indices[i * 3 + n] as usize
                } else {
                    i * 3 + n
                }
            };
            let i0 = index(0);
            let i1 = index(1);
            let i2 = index(2);

            let mut input_vertices = [Vertex::default(); 3];
            input_vertices[0].world_position = command.model * command.world_positions[i0];
            input_vertices[1].world_position = command.model * command.world_positions[i1];
            input_vertices[2].world_position = command.model * command.world_positions[i2];
            input_vertices[0].position = view_projection * input_vertices[0].world_position.as_point4();
            input_vertices[1].position = view_projection * input_vertices[1].world_position.as_point4();
            input_vertices[2].position = view_projection * input_vertices[2].world_position.as_point4();
            if command.normals.is_empty() {
                let edge1 = input_vertices[1].world_position - input_vertices[0].world_position;
                let edge2 = input_vertices[2].world_position - input_vertices[0].world_position;
                let face_normal = cross(edge1, edge2).normalized();
                input_vertices[0].normal = face_normal;
                input_vertices[1].normal = face_normal;
                input_vertices[2].normal = face_normal;
            } else {
                input_vertices[0].normal = normal_matrix * command.normals[i0];
                input_vertices[1].normal = normal_matrix * command.normals[i1];
                input_vertices[2].normal = normal_matrix * command.normals[i2];
            }
            if command.colors.is_empty() {
                input_vertices[0].color = command.color;
                input_vertices[1].color = command.color;
                input_vertices[2].color = command.color;
            } else {
                input_vertices[0].color = command.colors[i0] * command.color;
                input_vertices[1].color = command.colors[i1] * command.color;
                input_vertices[2].color = command.colors[i2] * command.color;
            }
            if command.tex_coords.is_empty() {
                input_vertices[0].tex_coord = Vec2::new(0.0, 0.0);
                input_vertices[1].tex_coord = Vec2::new(0.0, 0.0);
                input_vertices[2].tex_coord = Vec2::new(0.0, 0.0);
            } else {
                input_vertices[0].tex_coord = command.tex_coords[i0];
                input_vertices[1].tex_coord = command.tex_coords[i1];
                input_vertices[2].tex_coord = command.tex_coords[i2];
            }

            // pub fn clip_triangle(input_vertices: &[Vertex; 3]) -> ArrayVec<Vertex, 7> {
            let clipped_vertices = clip_triangle(&input_vertices);
            if clipped_vertices.is_empty() {
                continue;
            }

            // let mut clipped_vertex_idx = 1usize;
            // while clipped_vertex_idx + 1 < clipped_vertices.len() {
            for clipped_vertex_idx in 1..clipped_vertices.len() - 1 {
                let mut vertices = [
                    clipped_vertices[0],                      //
                    clipped_vertices[clipped_vertex_idx],     //
                    clipped_vertices[clipped_vertex_idx + 1], //
                ];
                // clipped_vertex_idx += 1;

                vertices[0].position = perspective_divide(vertices[0].position);
                vertices[1].position = perspective_divide(vertices[1].position);
                vertices[2].position = perspective_divide(vertices[2].position);
                vertices[0].position = viewport_scale.apply(vertices[0].position);
                vertices[1].position = viewport_scale.apply(vertices[1].position);
                vertices[2].position = viewport_scale.apply(vertices[2].position);

                let v01 = vertices[1].position.xy() - vertices[0].position.xy();
                let v02 = vertices[2].position.xy() - vertices[0].position.xy();
                let ccw = Mat22([v01.x, v02.x, v01.y, v02.y]).det() < 0.0;

                if (command.culling == CullMode::CW && !ccw) || (command.culling == CullMode::CCW && ccw) {
                    continue;
                }

                if ccw {
                    vertices.swap(2, 1);
                }

                self.vertices.extend_from_slice(&vertices);
            }
        }
    }

    pub fn draw(&mut self, framebuffer: &mut Framebuffer) {
        if framebuffer.color_buffer.is_none() || self.vertices.is_empty() {
            return;
        }

        // let mut lines: Vec<Vec2> = Vec::new();
        // let tri_num = self.vertices.len() / 3;
        // for i in 0..tri_num {
        //     lines.push(self.vertices[i * 3 + 0].position.xy());
        //     lines.push(self.vertices[i * 3 + 1].position.xy());
        //     lines.push(self.vertices[i * 3 + 1].position.xy());
        //     lines.push(self.vertices[i * 3 + 2].position.xy());
        //     lines.push(self.vertices[i * 3 + 2].position.xy());
        //     lines.push(self.vertices[i * 3 + 0].position.xy());
        // }
        // draw_screen_lines_unclipped(framebuffer, &lines, Vec4::new(1.0, 1.0, 1.0, 1.0));

        self.draw_triangles(framebuffer, &self.vertices);
    }

    // fn idx_to_color_hash(mut x: u32) -> u32 {
    //     // Mix the bits using a few bitwise operations and multiplications
    //     x ^= x >> 16;
    //     x = x.wrapping_mul(0x85ebca6b);
    //     x ^= x >> 13;
    //     x = x.wrapping_mul(0xc2b2ae35);
    //     x ^= x >> 16;
    //
    //     x | 0xFF
    // }

    fn is_top_left(edge: Vec2) -> bool {
        if edge.y < 0.0 {
            return true; // left edge
        }

        if edge.y.abs() < 0.0001 && edge.x > 0.0 {
            return true; // top edge
        }

        return false;
    }

    fn draw_triangles(&self, framebuffer: &mut Framebuffer, vertices: &[Vertex]) {
        let triangles_num = vertices.len() / 3;
        if triangles_num == 0 {
            return;
        }

        let rt_xmin = self.viewport.xmin.max(0) as i32;
        let rt_xmax = self.viewport.xmax.min(framebuffer.width()) as i32 - 1;
        let rt_ymin = self.viewport.ymin.max(0) as i32;
        let rt_ymax = self.viewport.ymax.min(framebuffer.height()) as i32 - 1;

        for i in 0..triangles_num {
            let v0 = &vertices[i * 3 + 0];
            let v1 = &vertices[i * 3 + 1];
            let v2 = &vertices[i * 3 + 2];

            let v01 = (v1.position - v0.position).xy();
            let v12 = (v2.position - v1.position).xy();
            let v20 = (v0.position - v2.position).xy();
            let v02 = (v2.position - v0.position).xy();

            let area_x_2 = v01.x * v02.y - v01.y * v02.x;

            if area_x_2 < 1.0 {
                continue; // TODO: treat degenerate triangles separately
            }

            // const f32 det012 = det2D(v1.position - v0.position, v2.position - v0.position);

            let is_v01_top_left = Self::is_top_left(v01);
            let is_v12_top_left = Self::is_top_left(v12);
            let is_v20_top_left = Self::is_top_left(v20);
            let v01_bias = if is_v01_top_left { 0.0 } else { -0.001 };
            let v12_bias = if is_v12_top_left { 0.0 } else { -0.001 };
            let v20_bias = if is_v20_top_left { 0.0 } else { -0.001 };

            let xmin = rt_xmin.max(v0.position.x.min(v1.position.x).min(v2.position.x) as i32);
            let xmax = rt_xmax.min(v0.position.x.max(v1.position.x).max(v2.position.x) as i32);
            let ymin = rt_ymin.max(v0.position.y.min(v1.position.y).min(v2.position.y) as i32);
            let ymax = rt_ymax.min(v0.position.y.max(v1.position.y).max(v2.position.y) as i32);

            let p_min = Vec2::new(xmin as f32 + 0.5, ymin as f32 + 0.5);
            let v0p_min = p_min - v0.position.xy();
            let v1p_min = p_min - v1.position.xy();
            let v2p_min = p_min - v2.position.xy();
            let edge0_min = v12.x * v1p_min.y - v12.y * v1p_min.x + v12_bias;
            let edge1_min = v20.x * v2p_min.y - v20.y * v2p_min.x + v20_bias;
            let edge2_min = v01.x * v0p_min.y - v01.y * v0p_min.x + v01_bias;
            let edge0_dx = -v12.y;
            let edge1_dx = -v20.y;
            let edge2_dx = -v01.y;
            let edge0_dy = v12.x;
            let edge1_dy = v20.x;
            let edge2_dy = v01.x;

            let mut edge0_row = edge0_min;
            let mut edge1_row = edge1_min;
            let mut edge2_row = edge2_min;

            // Precompute z interpolation increments
            // TODO: optimize/streamline this
            let z0 = (v0.position.z * 0.5 + 0.5) * 65535.0;
            let z1 = (v1.position.z * 0.5 + 0.5) * 65535.0;
            let z2 = (v2.position.z * 0.5 + 0.5) * 65535.0;
            let z_f32_min = z0 * edge0_min / area_x_2 + z1 * edge1_min / area_x_2 + z2 * edge2_min / area_x_2;
            let z_f32_dx = (z0 * edge0_dx + z1 * edge1_dx + z2 * edge2_dx) / area_x_2;
            let z_f32_dy = (z0 * edge0_dy + z1 * edge1_dy + z2 * edge2_dy) / area_x_2;
            let z_24x8_min = (z_f32_min * 256.0) as i32 as u32;
            let z_24x8_dx = (z_f32_dx * 256.0) as i32;
            let z_24x8_dy = (z_f32_dy * 256.0) as i32;

            let mut z_24x8_row = z_24x8_min; // starting depth at each consequent row
            for y in ymin..=ymax {
                let mut edge0 = edge0_row;
                let mut edge1 = edge1_row;
                let mut edge2 = edge2_row;
                let mut z_24x8 = z_24x8_row;

                for x in xmin..=xmax {
                    if edge0 >= 0.0 && edge1 >= 0.0 && edge2 >= 0.0 {
                        let mut discard = false;
                        if let Some(buffer) = &mut framebuffer.depth_buffer {
                            let z_u16 = (z_24x8 >> 8) as u16;
                            let dst_z = buffer.at_mut(x as u16, y as u16);
                            if z_u16 < *dst_z {
                                *dst_z = z_u16;
                            } else {
                                discard = true;
                            }
                        }
                        if !discard {
                            let l0 = edge0 * v0.position.w / area_x_2;
                            let l1 = edge1 * v1.position.w / area_x_2;
                            let l2 = edge2 * v2.position.w / area_x_2;
                            let wp = l0 + l1 + l2;
                            let edge0_pc = l0 / wp;
                            let edge1_pc = l1 / wp;
                            let edge2_pc = l2 / wp;
                            let color_fp = v0.color * edge0_pc + v1.color * edge1_pc + v2.color * edge2_pc;
                            let color = RGBA::new(
                                (color_fp.x * 255.0).clamp(0.0, 255.0) as u8, //
                                (color_fp.y * 255.0).clamp(0.0, 255.0) as u8, //
                                (color_fp.z * 255.0).clamp(0.0, 255.0) as u8, //
                                (color_fp.w * 255.0).clamp(0.0, 255.0) as u8,
                            )
                            .to_u32();

                            if let Some(buffer) = &mut framebuffer.color_buffer {
                                *buffer.at_mut(x as u16, y as u16) = color;
                            }
                        }
                    }
                    edge0 += edge0_dx;
                    edge1 += edge1_dx;
                    edge2 += edge2_dx;
                    z_24x8 = z_24x8.overflowing_add_signed(z_24x8_dx).0;
                }
                edge0_row += edge0_dy;
                edge1_row += edge1_dy;
                edge2_row += edge2_dy;
                z_24x8_row = z_24x8_row.overflowing_add_signed(z_24x8_dy).0;
            }
        }
    }
}

fn perspective_divide(v: Vec4) -> Vec4 {
    return Vec4::new(v.x / v.w, v.y / v.w, v.z / v.w, 1.0 / v.w);
}

#[derive(Debug, Clone, Copy)]
struct ViewportScale {
    xa: f32,
    xc: f32, // x' = x * xa + xc
    ya: f32,
    yc: f32, // y' = y * ya + yc
}

impl ViewportScale {
    fn new(viewport: Viewport) -> Self {
        let dx = (viewport.xmax - viewport.xmin) as f32;
        let dy = (viewport.ymax - viewport.ymin) as f32;
        ViewportScale {
            xa: dx * 0.5,                          //
            xc: (viewport.xmin as f32) + dx * 0.5, //
            ya: dy * -0.5,                         //
            yc: (viewport.ymin as f32) + dy * 0.5, //
        }
    }

    fn apply(&self, v: Vec4) -> Vec4 {
        return Vec4::new(v.x * self.xa + self.xc, v.y * self.ya + self.yc, v.z, v.w);
    }
}

impl Default for ViewportScale {
    fn default() -> Self {
        return ViewportScale { xa: 0.0, xc: 0.0, ya: 0.0, yc: 0.0 };
    }
}

impl Default for RasterizationCommand<'_> {
    fn default() -> Self {
        Self {
            world_positions: &[],
            normals: &[],
            tex_coords: &[],
            colors: &[],
            indices: &[],
            model: Mat34::identity(),
            view: Mat44::identity(),
            projection: Mat44::identity(),
            culling: CullMode::None,
            color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba, RgbaImage};
    use rstest::rstest;
    use std::path::Path;

    fn reference_path<P: AsRef<Path>>(reference: P) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/reference_images/")
            .join(reference)
    }

    fn save_albedo_next_to_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(64, 64, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn save_depth_next_to_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[1], bytes[0], 0, 255]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(64, 64, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn compare_albedo_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) -> bool {
        const ERROR_TOLERANCE: u8 = 2; // acceptable difference per channel, 2 ~= 1%
        let reference_path = reference_path(reference);

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(64, 64, raw_rgba).unwrap();

        let img2: RgbaImage = image::open(reference_path).unwrap().into_rgba8();

        if img1.dimensions() != img2.dimensions() {
            return false;
        }

        img1.pixels().zip(img2.pixels()).all(|(p1, p2)| {
            let diff_r = (p1[0] as i16 - p2[0] as i16).abs() as u8;
            let diff_g = (p1[1] as i16 - p2[1] as i16).abs() as u8;
            let diff_b = (p1[2] as i16 - p2[2] as i16).abs() as u8;
            let diff_a = (p1[3] as i16 - p2[3] as i16).abs() as u8;
            diff_r <= ERROR_TOLERANCE
                && diff_g <= ERROR_TOLERANCE
                && diff_b <= ERROR_TOLERANCE
                && diff_a <= ERROR_TOLERANCE
        })
    }

    fn compare_depth_against_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) -> bool {
        let reference_path = reference_path(reference);
        let reference_image: RgbaImage = image::open(reference_path).unwrap().into_rgba8();
        if reference_image.width() != 64 || reference_image.height() != 64 {
            return false;
        }

        for (x, y, pixel) in reference_image.enumerate_pixels() {
            // reconstruct depth from R and G components
            let reference_depth = (((pixel[0] as u16) << 8) | (pixel[1] as u16)) as i32;
            let actual_depth = result.at(x as u16, y as u16) as i32;
            let diff = (reference_depth - actual_depth).abs();
            if diff > 100 {
                return false; // 100 / 65535 ~= 0.15% error tolerance
            }
        }
        true
    }

    fn assert_albedo_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let equal = compare_albedo_against_reference(result, &reference);
        if !equal {
            save_albedo_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn assert_depth_against_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) {
        let equal = compare_depth_against_reference(result, &reference);
        if !equal {
            save_depth_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn render_to_64x64_albedo(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = Buffer::<u32>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);

        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);

        color_buffer
    }

    fn render_to_64x64_depth(command: &RasterizationCommand) -> Buffer<u16> {
        let mut color_buffer = Buffer::<u32>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut depth_buffer = Buffer::<u16>::new(64, 64);
        depth_buffer.fill(65535);
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        framebuffer.depth_buffer = Some(&mut depth_buffer);

        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);

        depth_buffer
    }

    #[rstest]
    #[case(Vec4::new(0.0, 0.0, 0.0, 1.0), "rasterizer_triangle_simple_black.png")]
    #[case(Vec4::new(1.0, 1.0, 1.0, 1.0), "rasterizer_triangle_simple_white.png")]
    #[case(Vec4::new(1.0, 0.0, 0.0, 1.0), "rasterizer_triangle_simple_red.png")]
    #[case(Vec4::new(0.0, 1.0, 0.0, 1.0), "rasterizer_triangle_simple_green.png")]
    #[case(Vec4::new(0.0, 0.0, 1.0, 1.0), "rasterizer_triangle_simple_blue.png")]
    #[case(Vec4::new(1.0, 1.0, 0.0, 1.0), "rasterizer_triangle_simple_yellow.png")]
    #[case(Vec4::new(1.0, 0.0, 1.0, 1.0), "rasterizer_triangle_simple_purple.png")]
    #[case(Vec4::new(0.0, 1.0, 1.0, 1.0), "rasterizer_triangle_simple_cyan.png")]
    fn triangle_simple(#[case] color: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            color,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_0.png")]
    #[case(Vec2::new(-0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_2.png")]
    #[case(Vec2::new(-0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_3.png")]
    #[case(Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_4.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_6.png")]
    #[case(Vec2::new(0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_7.png")]
    #[case(Vec2::new(1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_top_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_bottom_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 1.0), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.75), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_3.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_4.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.25), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_6.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.75), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_7.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -1.0), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_left_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 1.0), "rasterizer_triangle_orientation_right_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.75), "rasterizer_triangle_orientation_right_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer_triangle_orientation_right_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.25), "rasterizer_triangle_orientation_right_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.0), "rasterizer_triangle_orientation_right_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.25), "rasterizer_triangle_orientation_right_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer_triangle_orientation_right_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.75), "rasterizer_triangle_orientation_right_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -1.0), "rasterizer_triangle_orientation_right_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.0), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_3.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_4.png")]
    #[case(Vec2::new(0.25, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_5.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_6.png")]
    #[case(Vec2::new(0.5, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_7.png")]
    #[case(Vec2::new(0.5, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer_triangle_orientation_other_8.png")]
    fn triangle_orientation(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.75), Vec2::new(1.0, -1.0), "rasterizer_triangle_thin_00.png")]
    #[case(Vec2::new(-0.75, 1.0), Vec2::new(-1.0, 1.0), Vec2::new(1.0, -1.0), "rasterizer_triangle_thin_01.png")]
    #[case(Vec2::new(1.0, 1.0), Vec2::new(0.75, 1.0), Vec2::new(-1.0, -1.0), "rasterizer_triangle_thin_02.png")]
    #[case(Vec2::new(1.0, 0.75), Vec2::new(1.0, 1.0), Vec2::new(-1.0, -1.0), "rasterizer_triangle_thin_03.png")]
    #[case(Vec2::new(1.0, -0.75), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0), "rasterizer_triangle_thin_04.png")]
    #[case(Vec2::new(1.0, -1.0), Vec2::new(0.75, -1.0), Vec2::new(-1.0, 1.0), "rasterizer_triangle_thin_05.png")]
    #[case(Vec2::new(-0.75, -1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer_triangle_thin_06.png")]
    #[case(Vec2::new(-1.0, -1.0), Vec2::new(-1.0, -0.75), Vec2::new(1.0, 1.0), "rasterizer_triangle_thin_07.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(0.0, 1.0), Vec2::new(0.0, -1.0), "rasterizer_triangle_thin_08.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer_triangle_thin_09.png")]
    #[case(Vec2::new(0.0, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer_triangle_thin_10.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), "rasterizer_triangle_thin_11.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer_triangle_thin_12.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer_triangle_thin_13.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.0, -1.0), Vec2::new(0.0, 1.0), "rasterizer_triangle_thin_14.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer_triangle_thin_15.png")]
    #[case(Vec2::new(0.0, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer_triangle_thin_16.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer_triangle_thin_17.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.25), "rasterizer_triangle_thin_18.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.25), "rasterizer_triangle_thin_19.png")]
    fn triangle_thin(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_00.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.75), "rasterizer_depth_interpolation_large_01.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.5), "rasterizer_depth_interpolation_large_02.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.25), "rasterizer_depth_interpolation_large_03.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.0), "rasterizer_depth_interpolation_large_04.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.25), "rasterizer_depth_interpolation_large_05.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.5), "rasterizer_depth_interpolation_large_06.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.75), "rasterizer_depth_interpolation_large_07.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer_depth_interpolation_large_08.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_09.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_10.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_11.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_12.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_13.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_14.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_15.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_16.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_17.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_18.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_19.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_20.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_21.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_22.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_23.png")]
    #[case(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer_depth_interpolation_large_24.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer_depth_interpolation_large_25.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_00.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_01.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_02.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_03.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_04.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_05.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_06.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_07.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_08.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_09.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_10.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_11.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_12.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_13.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_14.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_15.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer_depth_interpolation_tilted_16.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.75), "rasterizer_depth_interpolation_tilted_17.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.5), "rasterizer_depth_interpolation_tilted_18.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.25), "rasterizer_depth_interpolation_tilted_19.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.0), "rasterizer_depth_interpolation_tilted_20.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.25), "rasterizer_depth_interpolation_tilted_21.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.5), "rasterizer_depth_interpolation_tilted_22.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.75), "rasterizer_depth_interpolation_tilted_23.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer_depth_interpolation_tilted_24.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer_depth_interpolation_tilted_25.png")]
    fn depth_interpolation(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            projection: Mat44::orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            ..Default::default()
        };
        assert_depth_against_reference(&render_to_64x64_depth(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer_color_interpolation_simple_0.png")]
    #[case(Vec3::new(-1.75, -1.75, -3.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer_color_interpolation_simple_1.png")]
    #[case(Vec3::new(-3.75, -3.75, -7.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer_color_interpolation_simple_2.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(1.75, -1.75, -3.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer_color_interpolation_simple_3.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(3.75, -3.75, -7.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer_color_interpolation_simple_4.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 1.75, -3.5), "rasterizer_color_interpolation_simple_5.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 3.75, -7.5), "rasterizer_color_interpolation_simple_6.png")]
    fn color_interpolation(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            colors: &[Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)],
            projection: Mat44::perspective(0.1, 10.0, std::f32::consts::PI / 3.0, 1.),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }
}

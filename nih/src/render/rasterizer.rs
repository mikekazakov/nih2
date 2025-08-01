use super::super::math::*;
use super::*;
use arrayvec::ArrayVec;
use std::mem::swap;

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
    pub normals: &'a [Vec3],    // TODO: support deriving the normals?
    pub tex_coords: &'a [Vec2], // empty if absent
    pub colors: &'a [Vec4],     // empty if absent
    pub indices: &'a [u32],
    pub model: Mat34,
    pub view: Mat44,
    pub projection: Mat44,
    pub culling: CullMode,
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
        if command.indices.len() < 3 {
            return;
        }
        let view_projection = command.projection * command.view;
        let normal_matrix = command.view.as_mat33() * command.model.as_mat33().inverse();
        let viewport_scale = self.viewport_scale;

        let input_triangles_num = command.indices.len() / 3;
        for i in 0..input_triangles_num {
            let input_triangle_idx = i * 3;
            let i0 = command.indices[input_triangle_idx + 0] as usize;
            let i1 = command.indices[input_triangle_idx + 1] as usize;
            let i2 = command.indices[input_triangle_idx + 2] as usize;

            let mut input_vertices = [Vertex::default(); 3];
            input_vertices[0].world_position = command.model * command.world_positions[i0];
            input_vertices[1].world_position = command.model * command.world_positions[i1];
            input_vertices[2].world_position = command.model * command.world_positions[i2];
            input_vertices[0].position = view_projection * input_vertices[0].world_position.as_point4();
            input_vertices[1].position = view_projection * input_vertices[1].world_position.as_point4();
            input_vertices[2].position = view_projection * input_vertices[2].world_position.as_point4();
            // TODO: derive automatically?
            input_vertices[0].normal = normal_matrix * command.normals[i0];
            input_vertices[1].normal = normal_matrix * command.normals[i1];
            input_vertices[2].normal = normal_matrix * command.normals[i2];
            if command.colors.is_empty() {
                input_vertices[0].color = Vec4::new(1.0, 1.0, 1.0, 1.0);
                input_vertices[1].color = Vec4::new(1.0, 1.0, 1.0, 1.0);
                input_vertices[2].color = Vec4::new(1.0, 1.0, 1.0, 1.0);
            } else {
                input_vertices[0].color = command.colors[i0];
                input_vertices[1].color = command.colors[i1];
                input_vertices[2].color = command.colors[i2];
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

    fn idx_to_color_hash(mut x: u32) -> u32 {
        // Mix the bits using a few bitwise operations and multiplications
        x ^= x >> 16;
        x = x.wrapping_mul(0x85ebca6b);
        x ^= x >> 13;
        x = x.wrapping_mul(0xc2b2ae35);
        x ^= x >> 16;

        x | 0xFF
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

            let xmin = rt_xmin.max(v0.position.x.min(v1.position.x).min(v2.position.x) as i32);
            let xmax = rt_xmax.min(v0.position.x.max(v1.position.x).max(v2.position.x) as i32);
            let ymin = rt_ymin.max(v0.position.y.min(v1.position.y).min(v2.position.y) as i32);
            let ymax = rt_ymax.min(v0.position.y.max(v1.position.y).max(v2.position.y) as i32);
            for y in ymin..=ymax {
                for x in xmin..=xmax {
                    let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let v0p = p - v0.position.xy();
                    let v1p = p - v1.position.xy();
                    let v2p = p - v2.position.xy();

                    let det01p = Mat22([v01.x, v0p.x, v01.y, v0p.y]).det();
                    let det12p = Mat22([v12.x, v1p.x, v12.y, v1p.y]).det();
                    let det20p = Mat22([v20.x, v2p.x, v20.y, v2p.y]).det();

                    if det01p >= 0.0 && det12p >= 0.0 && det20p >= 0.0 {
                        if let Some(buffer) = &mut framebuffer.color_buffer {
                            // *buffer.at_mut(x as u16, y as u16) = RGBA::new(127, 127, 127, 255).to_u32();
                            *buffer.at_mut(x as u16, y as u16) = Self::idx_to_color_hash(i as u32);
                        }
                    }
                }
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
        }
    }
}

use super::super::math::*;
use super::*;
use arrayvec::ArrayVec;
use std::cmp::{max, min};
use std::ops::Add;
use std::ptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    /// No culling — all triangles are rendered.
    None,

    /// Cull clockwise-wound triangles.
    CW,

    /// Cull counter-clockwise-wound triangles.
    CCW,
}

#[derive(Debug, Clone)]
pub struct RasterizationCommand<'a> {
    pub world_positions: &'a [Vec3],
    pub normals: &'a [Vec3],    // if no normals are provided, they will be derived automatically
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
    pub texture: Option<std::sync::Arc<Texture>>,

    // Set the filter to be used when sampling the texture.
    // Default: nearest.
    pub sampling_filter: SamplerFilter,

    // Sets whether the rasterizer should use alpha blending when writing fragments to the framebuffer.
    // If disabled, the fragment color will be written as is.
    // Default: disabled.
    pub alpha_blending: bool,
}

#[derive(Debug, Clone)]
struct ScheduledCommand {
    texture: Option<std::sync::Arc<Texture>>,
    sampling_filter: SamplerFilter,
    alpha_blending: bool,
}

#[derive(Debug, Clone, Copy)]
struct ScheduledTriangle {
    // index of a rasterization command
    cmd: u16,

    // index of the triangle's first vertex
    tri_start: u16,
}

#[derive(Debug, Clone, Copy)]
struct TileBinningBounds {
    xmin_24_8: i32,
    ymin_24_8: i32,
    xmax_24_8: i32,
    ymax_24_8: i32,
}

struct Tile {
    triangles: Vec<ScheduledTriangle>,
    local_viewport: Viewport,
    binning_bounds: TileBinningBounds,
}

#[derive(Debug, Clone, Copy)]
pub struct RasterizerStatistics {
    // The number of triangles that were requested to be rasterized.
    pub committed_triangles: usize,

    // The number of triangles that were scheduled for rasterization after culling and clipping.
    pub scheduled_triangles: usize,

    // The number of triangles rasterized across all tiles.
    // (the same triangle can be rasterized multiple times if it is visible in multiple tiles)
    pub binned_triangles: usize,

    // The number of factual rasterized pixels.
    // Gathered only in Debug builds.
    pub fragments_drawn: usize,
}

#[derive(Debug, Clone, Copy)]
struct PerTileStatistics {
    pub fragments_drawn: usize,
}

pub struct Rasterizer {
    viewport: Viewport,
    viewport_scale: ViewportScale,
    vertices: Vec<Vertex>,
    commands: Vec<ScheduledCommand>,
    tiles: Vec<Tile>,
    tiles_x: u16,
    tiles_y: u16,
    stats: RasterizerStatistics,
    debug_coloring: bool,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            triangles: Vec::new(),
            local_viewport: Viewport::new(0, 0, 1, 1),
            binning_bounds: TileBinningBounds { xmin_24_8: 0, ymin_24_8: 0, xmax_24_8: 0, ymax_24_8: 0 },
        }
    }
}

impl Rasterizer {
    pub const TILE_WIDTH: usize = 64;
    pub const TILE_HEIGHT: usize = 64;

    pub fn new() -> Self {
        return Rasterizer {
            viewport: Viewport::new(0, 0, 1, 1),
            viewport_scale: ViewportScale::default(),
            vertices: Vec::new(),
            commands: Vec::new(),
            tiles: Vec::new(),
            tiles_x: 1,
            tiles_y: 1,
            stats: RasterizerStatistics::new(),
            debug_coloring: false,
        };
    }

    pub fn setup(&mut self, viewport: Viewport) {
        assert!(viewport.xmax > viewport.xmin);
        assert!(viewport.ymax > viewport.ymin);
        let width_px = (viewport.xmax - viewport.xmin) as usize;
        let height_px = (viewport.ymax - viewport.ymin) as usize;
        let tiles_x = (width_px + Self::TILE_WIDTH - 1) / Self::TILE_WIDTH;
        let tiles_y = (height_px + Self::TILE_HEIGHT - 1) / Self::TILE_HEIGHT;
        let tiles_num = tiles_x * tiles_y;

        self.tiles_x = tiles_x as u16;
        self.tiles_y = tiles_y as u16;
        self.tiles.resize_with(tiles_num, Tile::default);
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                let tile = &mut self.tiles[y * tiles_x + x];
                tile.triangles.clear();
                tile.local_viewport = Viewport {
                    xmin: viewport.xmin + x as u16 * Self::TILE_WIDTH as u16,
                    ymin: viewport.ymin + y as u16 * Self::TILE_HEIGHT as u16,
                    xmax: (viewport.xmin + (x as u16 + 1) * Self::TILE_WIDTH as u16).min(viewport.xmax),
                    ymax: (viewport.ymin + (y as u16 + 1) * Self::TILE_HEIGHT as u16).min(viewport.ymax),
                };
                tile.binning_bounds = TileBinningBounds {
                    xmin_24_8: (x * Self::TILE_WIDTH) as i32 * 256,
                    ymin_24_8: (y * Self::TILE_HEIGHT) as i32 * 256,
                    xmax_24_8: (x * Self::TILE_WIDTH + Self::TILE_WIDTH - 1) as i32 * 256 + 255,
                    ymax_24_8: (y * Self::TILE_HEIGHT + Self::TILE_HEIGHT - 1) as i32 * 256 + 255,
                };
            }
        }

        self.viewport = viewport;
        self.viewport_scale = ViewportScale::new(viewport);
        self.vertices.clear();
        self.commands.clear();
        self.stats = RasterizerStatistics::new();
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

        self.stats.committed_triangles += input_triangles_num;

        let view_projection = command.projection * command.view;
        let normal_matrix = command.model.as_mat33().inverse().transpose();
        let viewport_scale = self.viewport_scale;
        let scheduled_vertices_start = self.vertices.len();

        let command_color_alpha_premul: Vec4 = Vec4::new(
            command.color.x * command.color.w,
            command.color.y * command.color.w,
            command.color.z * command.color.w,
            command.color.w,
        );

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
                input_vertices[0].normal = (normal_matrix * command.normals[i0]).normalized();
                input_vertices[1].normal = (normal_matrix * command.normals[i1]).normalized();
                input_vertices[2].normal = (normal_matrix * command.normals[i2]).normalized();
            }
            if command.colors.is_empty() {
                input_vertices[0].color = command_color_alpha_premul;
                input_vertices[1].color = command_color_alpha_premul;
                input_vertices[2].color = command_color_alpha_premul;
            } else {
                // TODO: branch out if command_color_alpha_premul==(1,1,1,1)
                input_vertices[0].color = Vec4::new(
                    command.colors[i0].x * command.colors[i0].w * command_color_alpha_premul.x,
                    command.colors[i0].y * command.colors[i0].w * command_color_alpha_premul.y,
                    command.colors[i0].z * command.colors[i0].w * command_color_alpha_premul.z,
                    command.colors[i0].w * command_color_alpha_premul.w,
                );
                input_vertices[1].color = Vec4::new(
                    command.colors[i1].x * command.colors[i1].w * command_color_alpha_premul.x,
                    command.colors[i1].y * command.colors[i1].w * command_color_alpha_premul.y,
                    command.colors[i1].z * command.colors[i1].w * command_color_alpha_premul.z,
                    command.colors[i1].w * command_color_alpha_premul.w,
                );
                input_vertices[2].color = Vec4::new(
                    command.colors[i2].x * command.colors[i2].w * command_color_alpha_premul.x,
                    command.colors[i2].y * command.colors[i2].w * command_color_alpha_premul.y,
                    command.colors[i2].z * command.colors[i2].w * command_color_alpha_premul.z,
                    command.colors[i2].w * command_color_alpha_premul.w,
                );
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

            // TODO: cull earlier????
            // Why try clipping the triangle if it's not visible?

            let clipped_vertices = clip_triangle(&input_vertices);
            if clipped_vertices.is_empty() {
                continue;
            }

            for clipped_vertex_idx in 1..clipped_vertices.len() - 1 {
                let mut vertices = [
                    clipped_vertices[0],                      //
                    clipped_vertices[clipped_vertex_idx],     //
                    clipped_vertices[clipped_vertex_idx + 1], //
                ];

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

        if scheduled_vertices_start == self.vertices.len() {
            return;
        }
        self.stats.scheduled_triangles += (self.vertices.len() - scheduled_vertices_start) / 3;

        // When debug triangle coloring is enabled, textures are disabled.
        let command_texture = if self.debug_coloring {
            None
        } else {
            command.texture.clone()
        };

        // When debug triangle coloring is enabled, color the triangles using their indices.
        if self.debug_coloring {
            for vert_idx in (scheduled_vertices_start..self.vertices.len()).step_by(3) {
                let color = debug_color(vert_idx as u32);
                self.vertices[vert_idx + 0].color = color;
                self.vertices[vert_idx + 1].color = color;
                self.vertices[vert_idx + 2].color = color;
            }
        }

        // Reuse the last command or create a new one
        let required_scheduled_command = ScheduledCommand {
            texture: command_texture,
            sampling_filter: command.sampling_filter,
            alpha_blending: command.alpha_blending,
        };
        if self.commands.is_empty() || self.commands.last().unwrap() != &required_scheduled_command {
            self.commands.push(required_scheduled_command);
        }
        let scheduled_command_index = (self.commands.len() - 1) as u16;

        // Now bin each scheduled triangle
        let xmin = self.viewport.xmin as u32;
        let ymin = self.viewport.ymin as u32;
        for vert_idx in (scheduled_vertices_start..self.vertices.len()).step_by(3) {
            let v0 = &self.vertices[vert_idx + 0];
            let v1 = &self.vertices[vert_idx + 1];
            let v2 = &self.vertices[vert_idx + 2];
            let v_xmin = v0.position.x.min(v1.position.x).min(v2.position.x) as u32;
            let v_xmax = v0.position.x.max(v1.position.x).max(v2.position.x) as u32;
            let v_ymin = v0.position.y.min(v1.position.y).min(v2.position.y) as u32;
            let v_ymax = v0.position.y.max(v1.position.y).max(v2.position.y) as u32;
            // TODO: add less crude discarding by running simple edge functions
            // TODO: check if this min() is required
            let ind_xmin = ((v_xmin - xmin) / Self::TILE_WIDTH as u32).min(self.tiles_x as u32 - 1);
            let ind_ymin = ((v_ymin - ymin) / Self::TILE_HEIGHT as u32).min(self.tiles_y as u32 - 1);
            let ind_xmax = ((v_xmax - xmin) / Self::TILE_WIDTH as u32).min(self.tiles_x as u32 - 1);
            let ind_ymax = ((v_ymax - ymin) / Self::TILE_HEIGHT as u32).min(self.tiles_y as u32 - 1);
            if ind_xmin == ind_xmax || ind_ymin == ind_ymax {
                // The triangle is fully contained in a single tile or it a horizontal or vertical line, bin it in the appropriate tiles.
                // No additional overlap checks are required.
                for ind_y in ind_ymin..=ind_ymax {
                    for ind_x in ind_xmin..=ind_xmax {
                        let tile = &mut self.tiles[ind_y as usize * self.tiles_x as usize + ind_x as usize];
                        tile.triangles
                            .push(ScheduledTriangle { cmd: scheduled_command_index, tri_start: vert_idx as u16 });
                        self.stats.binned_triangles += 1;
                    }
                }
            } else {
                // The triangle spans 2x2 or more tiles, bin in the appropriate tiles, but only after running simple edge functions check
                let iv0_x_24_8 = (v0.position.x * 256.0).round() as i32;
                let iv0_y_24_8 = (v0.position.y * 256.0).round() as i32;
                let iv1_x_24_8 = (v1.position.x * 256.0).round() as i32;
                let iv1_y_24_8 = (v1.position.y * 256.0).round() as i32;
                let iv2_x_24_8 = (v2.position.x * 256.0).round() as i32;
                let iv2_y_24_8 = (v2.position.y * 256.0).round() as i32;
                let iv01_x_24_8 = iv1_x_24_8 - iv0_x_24_8;
                let iv01_y_24_8 = iv1_y_24_8 - iv0_y_24_8;
                let iv12_x_24_8 = iv2_x_24_8 - iv1_x_24_8;
                let iv12_y_24_8 = iv2_y_24_8 - iv1_y_24_8;
                let iv20_x_24_8 = iv0_x_24_8 - iv2_x_24_8;
                let iv20_y_24_8 = iv0_y_24_8 - iv2_y_24_8;
                let is_tile_fully_outside = |tile_bounds: TileBinningBounds| {
                    let iv1_xmin_24_8 = tile_bounds.xmin_24_8 - iv1_x_24_8;
                    let iv1_ymin_24_8 = tile_bounds.ymin_24_8 - iv1_y_24_8;
                    let iv1_xmax_24_8 = tile_bounds.xmax_24_8 - iv1_x_24_8;
                    let iv1_ymax_24_8 = tile_bounds.ymax_24_8 - iv1_y_24_8;
                    let iv2_xmin_24_8 = tile_bounds.xmin_24_8 - iv2_x_24_8;
                    let iv2_ymin_24_8 = tile_bounds.ymin_24_8 - iv2_y_24_8;
                    let iv2_xmax_24_8 = tile_bounds.xmax_24_8 - iv2_x_24_8;
                    let iv2_ymax_24_8 = tile_bounds.ymax_24_8 - iv2_y_24_8;
                    let iv0_xmin_24_8 = tile_bounds.xmin_24_8 - iv0_x_24_8;
                    let iv0_ymin_24_8 = tile_bounds.ymin_24_8 - iv0_y_24_8;
                    let iv0_xmax_24_8 = tile_bounds.xmax_24_8 - iv0_x_24_8;
                    let iv0_ymax_24_8 = tile_bounds.ymax_24_8 - iv0_y_24_8;
                    let e0_lb = iv12_x_24_8 as i64 * iv1_ymin_24_8 as i64 - iv12_y_24_8 as i64 * iv1_xmin_24_8 as i64;
                    let e0_rb = iv12_x_24_8 as i64 * iv1_ymin_24_8 as i64 - iv12_y_24_8 as i64 * iv1_xmax_24_8 as i64;
                    let e0_lt = iv12_x_24_8 as i64 * iv1_ymax_24_8 as i64 - iv12_y_24_8 as i64 * iv1_xmin_24_8 as i64;
                    let e0_rt = iv12_x_24_8 as i64 * iv1_ymax_24_8 as i64 - iv12_y_24_8 as i64 * iv1_xmax_24_8 as i64;
                    let e1_lb = iv20_x_24_8 as i64 * iv2_ymin_24_8 as i64 - iv20_y_24_8 as i64 * iv2_xmin_24_8 as i64;
                    let e1_rb = iv20_x_24_8 as i64 * iv2_ymin_24_8 as i64 - iv20_y_24_8 as i64 * iv2_xmax_24_8 as i64;
                    let e1_lt = iv20_x_24_8 as i64 * iv2_ymax_24_8 as i64 - iv20_y_24_8 as i64 * iv2_xmin_24_8 as i64;
                    let e1_rt = iv20_x_24_8 as i64 * iv2_ymax_24_8 as i64 - iv20_y_24_8 as i64 * iv2_xmax_24_8 as i64;
                    let e2_lb = iv01_x_24_8 as i64 * iv0_ymin_24_8 as i64 - iv01_y_24_8 as i64 * iv0_xmin_24_8 as i64;
                    let e2_rb = iv01_x_24_8 as i64 * iv0_ymin_24_8 as i64 - iv01_y_24_8 as i64 * iv0_xmax_24_8 as i64;
                    let e2_lt = iv01_x_24_8 as i64 * iv0_ymax_24_8 as i64 - iv01_y_24_8 as i64 * iv0_xmin_24_8 as i64;
                    let e2_rt = iv01_x_24_8 as i64 * iv0_ymax_24_8 as i64 - iv01_y_24_8 as i64 * iv0_xmax_24_8 as i64;
                    (e0_lb < 0 && e0_rb < 0 && e0_lt < 0 && e0_rt < 0)
                        || (e1_lb < 0 && e1_rb < 0 && e1_lt < 0 && e1_rt < 0)
                        || (e2_lb < 0 && e2_rb < 0 && e2_lt < 0 && e2_rt < 0)
                };

                for ind_y in ind_ymin..=ind_ymax {
                    for ind_x in ind_xmin..=ind_xmax {
                        let tile = &mut self.tiles[ind_y as usize * self.tiles_x as usize + ind_x as usize];
                        if is_tile_fully_outside(tile.binning_bounds) {
                            continue;
                        }
                        tile.triangles
                            .push(ScheduledTriangle { cmd: scheduled_command_index, tri_start: vert_idx as u16 });
                        self.stats.binned_triangles += 1;
                    }
                }
            }
        }
    }

    pub fn draw(&mut self, framebuffer: &mut Framebuffer) {
        if self.vertices.is_empty() {
            return;
        }

        struct Job {
            framebuffer_tile: FramebufferTile,
            render_tile: *const Tile,
            statistics: PerTileStatistics,
        }
        unsafe impl Send for Job {}
        unsafe impl Sync for Job {}

        let mut jobs = Vec::<Job>::new();
        for y in 0..self.tiles_y {
            for x in 0..self.tiles_x {
                let idx = (y * self.tiles_x + x) as usize;
                let render_tile: *const Tile = &mut self.tiles[idx];
                let framebuffer_tile = framebuffer.tile(x, y);
                jobs.push(Job { framebuffer_tile, render_tile: render_tile, statistics: PerTileStatistics::default() });
            }
        }
        use rayon::prelude::*;
        jobs.par_iter_mut().for_each(|job| {
            let render_tile = unsafe { &*job.render_tile };
            if render_tile.triangles.is_empty() {
                return;
            }

            let viewport = render_tile.local_viewport;
            let vertices = &self.vertices;

            let mut tile_verts = ArrayVec::<Vertex, 384>::new(); // up to 128 triangles
            let mut cmd_idx = render_tile.triangles.first().unwrap().cmd;

            for tri in &render_tile.triangles {
                if tile_verts.is_full() || tri.cmd != cmd_idx {
                    let call_stats = self.draw_triangles_dispatch(
                        &mut job.framebuffer_tile,
                        viewport,
                        &tile_verts,
                        &self.commands[cmd_idx as usize],
                    );
                    job.statistics = job.statistics + call_stats;
                    tile_verts.clear();
                    cmd_idx = tri.cmd;
                }

                tile_verts.push(vertices[tri.tri_start as usize + 0]);
                tile_verts.push(vertices[tri.tri_start as usize + 1]);
                tile_verts.push(vertices[tri.tri_start as usize + 2]);
            }

            if !tile_verts.is_empty() {
                let call_stats = self.draw_triangles_dispatch(
                    &mut job.framebuffer_tile,
                    viewport,
                    &tile_verts,
                    &self.commands[cmd_idx as usize],
                );
                job.statistics = job.statistics + call_stats;
            }
        });

        for job in jobs {
            self.stats.fragments_drawn += job.statistics.fragments_drawn;
        }
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

    fn encode_normal_as_u32(nx: f32, ny: f32, nz: f32) -> u32 {
        unsafe {
            let x8: u8 = (nx * 127.5 + 127.5).to_int_unchecked();
            let y8: u8 = (ny * 127.5 + 127.5).to_int_unchecked();
            let z8: u8 = (nz * 127.5 + 127.5).to_int_unchecked();
            (x8 as u32) | ((y8 as u32) << 8) | ((z8 as u32) << 16)
        }
    }

    fn is_top_left_24_8(edge_x: i32, edge_y: i32) -> bool {
        ( edge_y < 0 ) || // left edge
        ( edge_y == 0 && edge_x > 0 ) // top edge
        // NB!
        // This says "an edge that is exactly horizontal", but perhaps some epsilon is still needed...
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules
    }

    fn draw_triangles_dispatch(
        &self,
        framebuffer: &mut FramebufferTile,
        local_viewport: Viewport,
        vertices: &[Vertex],
        command: &ScheduledCommand,
    ) -> PerTileStatistics {
        type DrawFunction =
            fn(&Rasterizer, &mut FramebufferTile, Viewport, &[Vertex], &ScheduledCommand) -> PerTileStatistics;
        macro_rules! comb {
            ( $( $($a:literal,)* $b:ty $(,$c:ty)* );+ ) => {
                comb!(
                    $( $($a,)* false $(,$c)* );+ ;
                    $( $($a,)* true $(,$c)* );+
                )
            };
            ( $( $($a:literal),+ );+ )                  => {
                [$( Rasterizer::draw_triangles::<$($a),+>, )+]
            };
        }
        const DRAW_FUNCTIONS: [DrawFunction; 32] = comb!(bool, bool, bool, bool, bool);
        let has_color: bool = framebuffer.color_buffer.is_some();
        let has_depth: bool = framebuffer.depth_buffer.is_some();
        let has_normal: bool = framebuffer.normal_buffer.is_some();
        let has_texture: bool = command.texture.is_some();
        let do_alpha_blending: bool = command.alpha_blending;

        let idx: usize = (has_color as usize) << 0
            | (has_depth as usize) << 1
            | (has_normal as usize) << 2
            | (has_texture as usize) << 3
            | (do_alpha_blending as usize) << 4;
        DRAW_FUNCTIONS[idx](self, framebuffer, local_viewport, vertices, command)
    }

    fn draw_triangles<
        const HAS_COLOR_BUFFER: bool,
        const HAS_DEPTH_BUFFER: bool,
        const HAS_NORMAL_BUFFER: bool,
        const HAS_TEXTURE: bool,
        const ALPHA_BLENDING: bool,
    >(
        &self,
        framebuffer: &mut FramebufferTile,
        local_viewport: Viewport,
        vertices: &[Vertex],
        command: &ScheduledCommand,
    ) -> PerTileStatistics {
        assert!(local_viewport.xmin >= framebuffer.origin_x());
        assert!(local_viewport.xmax >= framebuffer.origin_x());
        assert!(local_viewport.ymin >= framebuffer.origin_y());
        assert!(local_viewport.ymax >= framebuffer.origin_y());
        debug_assert_eq!(HAS_COLOR_BUFFER, framebuffer.color_buffer.is_some());
        debug_assert_eq!(HAS_DEPTH_BUFFER, framebuffer.depth_buffer.is_some());
        debug_assert_eq!(HAS_NORMAL_BUFFER, framebuffer.normal_buffer.is_some());
        let mut statistics = PerTileStatistics::default();
        let triangles_num = vertices.len() / 3;
        if triangles_num == 0 {
            return statistics;
        }

        let tile_origin = Vec2::new(framebuffer.origin_x() as f32, framebuffer.origin_y() as f32);
        let tile_origin_x_24_8: i32 = framebuffer.origin_x() as i32 * 256;
        let tile_origin_y_24_8: i32 = framebuffer.origin_y() as i32 * 256;

        let rt_xmin = (max(local_viewport.xmin, framebuffer.origin_x()) - framebuffer.origin_x()) as i32;
        let rt_xmax = (min(local_viewport.xmax, framebuffer.origin_x() + framebuffer.width())
            - framebuffer.origin_x()
            - 1) as i32;
        let rt_ymin = (max(local_viewport.ymin, framebuffer.origin_y()) - framebuffer.origin_y()) as i32;
        let rt_ymax = (min(local_viewport.ymax, framebuffer.origin_y() + framebuffer.height())
            - framebuffer.origin_y()
            - 1) as i32;

        for i in 0..triangles_num {
            let v0 = &vertices[i * 3 + 0];
            let v1 = &vertices[i * 3 + 1];
            let v2 = &vertices[i * 3 + 2];

            // Calculate the triangle's vertice positions relative to the tile origin
            let v0_xy = v0.position.xy() - tile_origin;
            let v1_xy = v1.position.xy() - tile_origin;
            let v2_xy = v2.position.xy() - tile_origin;
            let v0_x_24_8: i32 = (v0.position.x * 256.0).round() as i32 - tile_origin_x_24_8;
            let v0_y_24_8: i32 = (v0.position.y * 256.0).round() as i32 - tile_origin_y_24_8;
            let v1_x_24_8: i32 = (v1.position.x * 256.0).round() as i32 - tile_origin_x_24_8;
            let v1_y_24_8: i32 = (v1.position.y * 256.0).round() as i32 - tile_origin_y_24_8;
            let v2_x_24_8: i32 = (v2.position.x * 256.0).round() as i32 - tile_origin_x_24_8;
            let v2_y_24_8: i32 = (v2.position.y * 256.0).round() as i32 - tile_origin_y_24_8;

            // Calculate the edge vectors of the triangle
            let v01 = v1_xy - v0_xy;
            let v12 = v2_xy - v1_xy;
            let v20 = v0_xy - v2_xy;
            let v02 = v2_xy - v0_xy;
            let v01_x_24_8: i32 = v1_x_24_8 - v0_x_24_8;
            let v01_y_24_8: i32 = v1_y_24_8 - v0_y_24_8;
            let v12_x_24_8: i32 = v2_x_24_8 - v1_x_24_8;
            let v12_y_24_8: i32 = v2_y_24_8 - v1_y_24_8;
            let v20_x_24_8: i32 = v0_x_24_8 - v2_x_24_8;
            let v20_y_24_8: i32 = v0_y_24_8 - v2_y_24_8;

            // Calculate the doubled triangle's area
            let area_x_2: f32 = v01.x * v02.y - v01.y * v02.x;
            if area_x_2 < 1.0 {
                continue; // TODO: treat degenerate triangles separately
            }

            // Set up the sampler
            let sampler: Sampler = if HAS_TEXTURE {
                let texture = command.texture.as_ref().unwrap();
                let t01: Vec2 = v1.tex_coord - v0.tex_coord;
                let t02: Vec2 = v2.tex_coord - v0.tex_coord;
                let texel_area_x_2: f32 = (t01.x * t02.y - t02.x * t01.y).abs()
                    * texture.mips[0].width as f32
                    * texture.mips[0].height as f32;
                let rho2: f32 = texel_area_x_2 / area_x_2;
                let lod: f32 = 0.5 * rho2.log2();
                Sampler::new(texture, command.sampling_filter, lod)
            } else {
                Sampler::default()
            };
            let sampler_uv_scale: SamplerUVScale = sampler.uv_scale();

            // Set up the edge function biases to follow the top-left fill rule
            let is_v01_top_left: bool = Self::is_top_left_24_8(v01_x_24_8, v01_y_24_8);
            let is_v12_top_left: bool = Self::is_top_left_24_8(v12_x_24_8, v12_y_24_8);
            let is_v20_top_left: bool = Self::is_top_left_24_8(v20_x_24_8, v20_y_24_8);
            let v01_bias_x24_8: i32 = if is_v01_top_left { 0 } else { -1 };
            let v12_bias_x24_8: i32 = if is_v12_top_left { 0 } else { -1 };
            let v20_bias_x24_8: i32 = if is_v20_top_left { 0 } else { -1 };

            let xmin = rt_xmin.max(v0_xy.x.min(v1_xy.x).min(v2_xy.x) as i32);
            let xmax = rt_xmax.min(v0_xy.x.max(v1_xy.x).max(v2_xy.x) as i32);
            let ymin = rt_ymin.max(v0_xy.y.min(v1_xy.y).min(v2_xy.y) as i32);
            let ymax = rt_ymax.min(v0_xy.y.max(v1_xy.y).max(v2_xy.y) as i32);
            debug_assert!(xmax >= 0);
            debug_assert!(ymin >= 0);
            debug_assert!(xmax < Framebuffer::TILE_WITH as i32);
            debug_assert!(ymax < Framebuffer::TILE_HEIGHT as i32);

            // Calculate the min point of the triangle in the tile and that point relative to the edges (as f32)
            let p_min = Vec2::new(xmin as f32 + 0.5, ymin as f32 + 0.5);
            let v0p_min = p_min - v0_xy;
            let v1p_min = p_min - v1_xy;
            let v2p_min = p_min - v2_xy;

            // Calculate the min point of the triangle in the tile and that point relative to the edges (as 24.8)
            let p_min_x_24_8: i32 = xmin * 256 + 128;
            let p_min_y_24_8: i32 = ymin * 256 + 128;
            let v0p_min_x_24_8: i32 = p_min_x_24_8 - v0_x_24_8;
            let v0p_min_y_24_8: i32 = p_min_y_24_8 - v0_y_24_8;
            let v1p_min_x_24_8: i32 = p_min_x_24_8 - v1_x_24_8;
            let v1p_min_y_24_8: i32 = p_min_y_24_8 - v1_y_24_8;
            let v2p_min_x_24_8: i32 = p_min_x_24_8 - v2_x_24_8;
            let v2p_min_y_24_8: i32 = p_min_y_24_8 - v2_y_24_8;

            // Precompute edge functions start values and increments as f32
            let edge0_min = v12.x * v1p_min.y - v12.y * v1p_min.x;
            let edge1_min = v20.x * v2p_min.y - v20.y * v2p_min.x;
            let edge2_min = v01.x * v0p_min.y - v01.y * v0p_min.x;
            let edge0_dx = -v12.y;
            let edge1_dx = -v20.y;
            let edge2_dx = -v01.y;
            let edge0_dy = v12.x;
            let edge1_dy = v20.x;
            let edge2_dy = v01.x;

            // Precompute edge functions start values and increments as 24.8
            let edge0_min_24_8: i32 =
                ((v12_x_24_8 as i64 * v1p_min_y_24_8 as i64 - v12_y_24_8 as i64 * v1p_min_x_24_8 as i64) / 256) as i32
                    + v12_bias_x24_8;
            let edge1_min_24_8: i32 =
                ((v20_x_24_8 as i64 * v2p_min_y_24_8 as i64 - v20_y_24_8 as i64 * v2p_min_x_24_8 as i64) / 256) as i32
                    + v20_bias_x24_8;
            let edge2_min_24_8: i32 =
                ((v01_x_24_8 as i64 * v0p_min_y_24_8 as i64 - v01_y_24_8 as i64 * v0p_min_x_24_8 as i64) / 256) as i32
                    + v01_bias_x24_8;
            let edge0_24x8_dx: i32 = -v12_y_24_8;
            let edge1_24x8_dx: i32 = -v20_y_24_8;
            let edge2_24x8_dx: i32 = -v01_y_24_8;
            let edge0_24x8_dy: i32 = v12_x_24_8;
            let edge1_24x8_dy: i32 = v20_x_24_8;
            let edge2_24x8_dy: i32 = v01_x_24_8;

            // Precompute z start value and interpolation increments
            // TODO: optimize/streamline this
            let z0 = (v0.position.z * 0.5 + 0.5) * 65535.0;
            let z1 = (v1.position.z * 0.5 + 0.5) * 65535.0;
            let z2 = (v2.position.z * 0.5 + 0.5) * 65535.0;
            let z_f32_min = z0 * edge0_min / area_x_2 + z1 * edge1_min / area_x_2 + z2 * edge2_min / area_x_2;
            let z_f32_dx = (z0 * edge0_dx + z1 * edge1_dx + z2 * edge2_dx) / area_x_2;
            let z_f32_dy = (z0 * edge0_dy + z1 * edge1_dy + z2 * edge2_dy) / area_x_2;
            let z_24_8_min = (z_f32_min * 256.0) as i32 as u32;
            let z_24x8_dx = (z_f32_dx * 256.0) as i32;
            let z_24x8_dy = (z_f32_dy * 256.0) as i32;

            // Express per-vertex edgefunctions, 1/w, colors/w and N/w as Vectors-3 to simplify the setup math
            let edge_min_v3 = Vec3::new(edge0_min, edge1_min, edge2_min);
            let edge_dx_v3 = Vec3::new(edge0_dx, edge1_dx, edge2_dx);
            let edge_dy_v3 = Vec3::new(edge0_dy, edge1_dy, edge2_dy);
            let inv_w_v3 = Vec3::new(v0.position.w, v1.position.w, v2.position.w);
            let r_over_w_v3 =
                Vec3::new(v0.color.x * v0.position.w, v1.color.x * v1.position.w, v2.color.x * v2.position.w);
            let g_over_w_v3 =
                Vec3::new(v0.color.y * v0.position.w, v1.color.y * v1.position.w, v2.color.y * v2.position.w);
            let b_over_w_v3 =
                Vec3::new(v0.color.z * v0.position.w, v1.color.z * v1.position.w, v2.color.z * v2.position.w);
            let a_over_w_v3 =
                Vec3::new(v0.color.w * v0.position.w, v1.color.w * v1.position.w, v2.color.w * v2.position.w);
            let nx_over_w_v3 =
                Vec3::new(v0.normal.x * v0.position.w, v1.normal.x * v1.position.w, v2.normal.x * v2.position.w);
            let ny_over_w_v3 =
                Vec3::new(v0.normal.y * v0.position.w, v1.normal.y * v1.position.w, v2.normal.y * v2.position.w);
            let nz_over_w_v3 =
                Vec3::new(v0.normal.z * v0.position.w, v1.normal.z * v1.position.w, v2.normal.z * v2.position.w);
            let u_over_w_v3 = Vec3::new(
                (v0.tex_coord.x + sampler_uv_scale.bias) * sampler_uv_scale.scale * v0.position.w,
                (v1.tex_coord.x + sampler_uv_scale.bias) * sampler_uv_scale.scale * v1.position.w,
                (v2.tex_coord.x + sampler_uv_scale.bias) * sampler_uv_scale.scale * v2.position.w,
            );
            let v_over_w_v3 = Vec3::new(
                (v0.tex_coord.y + sampler_uv_scale.bias) * sampler_uv_scale.scale * v0.position.w,
                (v1.tex_coord.y + sampler_uv_scale.bias) * sampler_uv_scale.scale * v1.position.w,
                (v2.tex_coord.y + sampler_uv_scale.bias) * sampler_uv_scale.scale * v2.position.w,
            );

            // Precompute color/w start values and interpolation increments
            let r_over_w_min: f32 = dot(edge_min_v3, r_over_w_v3);
            let r_over_w_dx: f32 = dot(edge_dx_v3, r_over_w_v3);
            let r_over_w_dy: f32 = dot(edge_dy_v3, r_over_w_v3);
            let g_over_w_min: f32 = dot(edge_min_v3, g_over_w_v3);
            let g_over_w_dx: f32 = dot(edge_dx_v3, g_over_w_v3);
            let g_over_w_dy: f32 = dot(edge_dy_v3, g_over_w_v3);
            let b_over_w_min: f32 = dot(edge_min_v3, b_over_w_v3);
            let b_over_w_dx: f32 = dot(edge_dx_v3, b_over_w_v3);
            let b_over_w_dy: f32 = dot(edge_dy_v3, b_over_w_v3);
            let a_over_w_min: f32 = dot(edge_min_v3, a_over_w_v3);
            let a_over_w_dx: f32 = dot(edge_dx_v3, a_over_w_v3);
            let a_over_w_dy: f32 = dot(edge_dy_v3, a_over_w_v3);

            // Precompute normal/w start values and interpolation increments
            let nx_over_w_min: f32 = dot(edge_min_v3, nx_over_w_v3);
            let nx_over_w_dx: f32 = dot(edge_dx_v3, nx_over_w_v3);
            let nx_over_w_dy: f32 = dot(edge_dy_v3, nx_over_w_v3);
            let ny_over_w_min: f32 = dot(edge_min_v3, ny_over_w_v3);
            let ny_over_w_dx: f32 = dot(edge_dx_v3, ny_over_w_v3);
            let ny_over_w_dy: f32 = dot(edge_dy_v3, ny_over_w_v3);
            let nz_over_w_min: f32 = dot(edge_min_v3, nz_over_w_v3);
            let nz_over_w_dx: f32 = dot(edge_dx_v3, nz_over_w_v3);
            let nz_over_w_dy: f32 = dot(edge_dy_v3, nz_over_w_v3);

            // Precompute texture coordinates start values and interpolation increments
            let u_over_w_min: f32 = dot(edge_min_v3, u_over_w_v3);
            let u_over_w_dx: f32 = dot(edge_dx_v3, u_over_w_v3);
            let u_over_w_dy: f32 = dot(edge_dy_v3, u_over_w_v3);
            let v_over_w_min: f32 = dot(edge_min_v3, v_over_w_v3);
            let v_over_w_dx: f32 = dot(edge_dx_v3, v_over_w_v3);
            let v_over_w_dy: f32 = dot(edge_dy_v3, v_over_w_v3);

            // Precompute 1/w start value and interpolation increments
            let inv_w_min: f32 = dot(edge_min_v3, inv_w_v3);
            let inv_w_dx: f32 = dot(edge_dx_v3, inv_w_v3);
            let inv_w_dy: f32 = dot(edge_dy_v3, inv_w_v3);

            // Set up initial target pointers
            let mut color_row_ptr: *mut u32 = if HAS_COLOR_BUFFER {
                unsafe {
                    framebuffer
                        .color_buffer
                        .as_mut()
                        .unwrap_unchecked()
                        .ptr
                        .add((ymin * Framebuffer::TILE_WITH as i32 + xmin) as usize)
                }
            } else {
                ptr::null_mut()
            };
            let mut depth_row_ptr: *mut u16 = if HAS_DEPTH_BUFFER {
                unsafe {
                    framebuffer
                        .depth_buffer
                        .as_mut()
                        .unwrap_unchecked()
                        .ptr
                        .add((ymin * Framebuffer::TILE_WITH as i32 + xmin) as usize)
                }
            } else {
                ptr::null_mut()
            };
            let mut normal_row_ptr: *mut u32 = if HAS_NORMAL_BUFFER {
                unsafe {
                    framebuffer
                        .normal_buffer
                        .as_mut()
                        .unwrap_unchecked()
                        .ptr
                        .add((ymin * Framebuffer::TILE_WITH as i32 + xmin) as usize)
                }
            } else {
                ptr::null_mut()
            };

            // Set up the initial values at each consequent row
            let mut edge0_row_24_8: i32 = edge0_min_24_8; // starting v12 edgefunction value
            let mut edge1_row_24_8: i32 = edge1_min_24_8; // starting v20 edgefunction value
            let mut edge2_row_24_8: i32 = edge2_min_24_8; // starting v01 edgefunction value
            let mut z_24_8_row: u32 = z_24_8_min; // starting depth
            let mut r_over_w_row: f32 = r_over_w_min; // starting r/w
            let mut g_over_w_row: f32 = g_over_w_min; // starting g/w
            let mut b_over_w_row: f32 = b_over_w_min; // starting b/w
            let mut a_over_w_row: f32 = a_over_w_min; // starting a/w
            let mut nx_over_w_row: f32 = nx_over_w_min; // starting nx/w
            let mut ny_over_w_row: f32 = ny_over_w_min; // starting ny/w
            let mut nz_over_w_row: f32 = nz_over_w_min; // starting nz/w
            let mut u_over_w_row: f32 = u_over_w_min; // starting u/w
            let mut v_over_w_row: f32 = v_over_w_min; // starting v/w
            let mut inv_w_row: f32 = inv_w_min; // starting 1/w

            for _y in ymin..=ymax {
                let mut edge0_24_8: i32 = edge0_row_24_8;
                let mut edge1_24_8: i32 = edge1_row_24_8;
                let mut edge2_24_8: i32 = edge2_row_24_8;
                let mut inv_w: f32 = inv_w_row;
                let mut r_over_w: f32 = r_over_w_row;
                let mut g_over_w: f32 = g_over_w_row;
                let mut b_over_w: f32 = b_over_w_row;
                let mut a_over_w: f32 = a_over_w_row;
                let mut nx_over_w: f32 = nx_over_w_row;
                let mut ny_over_w: f32 = ny_over_w_row;
                let mut nz_over_w: f32 = nz_over_w_row;
                let mut u_over_w: f32 = u_over_w_row;
                let mut v_over_w: f32 = v_over_w_row;
                let mut z_24_8: u32 = z_24_8_row;
                let mut color_ptr: *mut u32 = if HAS_COLOR_BUFFER {
                    color_row_ptr
                } else {
                    ptr::null_mut()
                };
                let mut depth_ptr: *mut u16 = if HAS_DEPTH_BUFFER {
                    depth_row_ptr
                } else {
                    ptr::null_mut()
                };
                let mut normal_ptr: *mut u32 = if HAS_NORMAL_BUFFER {
                    normal_row_ptr
                } else {
                    ptr::null_mut()
                };

                for _x in xmin..=xmax {
                    'fragment: {
                        if edge0_24_8 < 0 || edge1_24_8 < 0 || edge2_24_8 < 0 {
                            break 'fragment; // discard - out of triangle bounds
                        }

                        if HAS_DEPTH_BUFFER {
                            let z_u16: u16 = (z_24_8 >> 8) as u16;
                            unsafe {
                                if z_u16 < *depth_ptr {
                                    *depth_ptr = z_u16;
                                } else {
                                    break 'fragment; // discard - failed the depth test
                                }
                            }
                        }

                        let inv_inv_w: f32 = 1.0 / inv_w;

                        if HAS_COLOR_BUFFER {
                            // Fetch a corresponding texel color
                            let tex_fragment = if HAS_TEXTURE {
                                let u: f32 = u_over_w * inv_inv_w;
                                let v: f32 = v_over_w * inv_inv_w;
                                sampler.sample_prescaled(u, v)
                            } else {
                                RGBA::new(255, 255, 255, 255)
                            };

                            // Recover interpolated per-fragment color
                            let interpolated_r: f32 = r_over_w * inv_inv_w;
                            let interpolated_g: f32 = g_over_w * inv_inv_w;
                            let interpolated_b: f32 = b_over_w * inv_inv_w;
                            let interpolated_a: f32 = a_over_w * inv_inv_w;

                            // Multiply the interpolated and texel colors
                            let r: u8 = (interpolated_r * tex_fragment.r as f32).clamp(0.0, 255.0) as u8;
                            let g: u8 = (interpolated_g * tex_fragment.g as f32).clamp(0.0, 255.0) as u8;
                            let b: u8 = (interpolated_b * tex_fragment.b as f32).clamp(0.0, 255.0) as u8;
                            let a: u8 = (interpolated_a * tex_fragment.a as f32).clamp(0.0, 255.0) as u8;

                            // Build the dest color
                            let color: u32 = if ALPHA_BLENDING {
                                let dest: RGBA = RGBA::from_u32(unsafe { *color_ptr });
                                let inv_a: u32 = (255 - a) as u32;
                                RGBA::new(
                                    r + ((dest.r as u32 * inv_a) / 255) as u8,
                                    g + ((dest.g as u32 * inv_a) / 255) as u8,
                                    b + ((dest.b as u32 * inv_a) / 255) as u8,
                                    255,
                                )
                                .to_u32()
                            } else {
                                RGBA::new(r, g, b, 255).to_u32()
                            };

                            // Write the fragment color into the framebuffer
                            unsafe {
                                *color_ptr = color;
                            }
                        }

                        if HAS_NORMAL_BUFFER {
                            unsafe {
                                *normal_ptr = Self::encode_normal_as_u32(
                                    nx_over_w * inv_inv_w,
                                    ny_over_w * inv_inv_w,
                                    nz_over_w * inv_inv_w,
                                );
                            }
                        }

                        if cfg!(debug_assertions) {
                            statistics.fragments_drawn += 1;
                        }
                    }
                    edge0_24_8 += edge0_24x8_dx;
                    edge1_24_8 += edge1_24x8_dx;
                    edge2_24_8 += edge2_24x8_dx;
                    z_24_8 = z_24_8.overflowing_add_signed(z_24x8_dx).0;
                    inv_w += inv_w_dx;
                    r_over_w += r_over_w_dx;
                    g_over_w += g_over_w_dx;
                    b_over_w += b_over_w_dx;
                    a_over_w += a_over_w_dx;
                    nx_over_w += nx_over_w_dx;
                    ny_over_w += ny_over_w_dx;
                    nz_over_w += nz_over_w_dx;
                    u_over_w += u_over_w_dx;
                    v_over_w += v_over_w_dx;
                    if HAS_COLOR_BUFFER {
                        unsafe {
                            color_ptr = color_ptr.add(1);
                        }
                    }
                    if HAS_DEPTH_BUFFER {
                        unsafe {
                            depth_ptr = depth_ptr.add(1);
                        }
                    }
                    if HAS_NORMAL_BUFFER {
                        unsafe {
                            normal_ptr = normal_ptr.add(1);
                        }
                    }
                }
                edge0_row_24_8 += edge0_24x8_dy;
                edge1_row_24_8 += edge1_24x8_dy;
                edge2_row_24_8 += edge2_24x8_dy;
                z_24_8_row = z_24_8_row.overflowing_add_signed(z_24x8_dy).0;
                inv_w_row += inv_w_dy;
                r_over_w_row += r_over_w_dy;
                g_over_w_row += g_over_w_dy;
                b_over_w_row += b_over_w_dy;
                a_over_w_row += a_over_w_dy;
                nx_over_w_row += nx_over_w_dy;
                ny_over_w_row += ny_over_w_dy;
                nz_over_w_row += nz_over_w_dy;
                u_over_w_row += u_over_w_dy;
                v_over_w_row += v_over_w_dy;
                if HAS_COLOR_BUFFER {
                    unsafe {
                        color_row_ptr = color_row_ptr.add(Framebuffer::TILE_WITH as usize);
                    }
                }
                if HAS_DEPTH_BUFFER {
                    unsafe {
                        depth_row_ptr = depth_row_ptr.add(Framebuffer::TILE_WITH as usize);
                    }
                }
                if HAS_NORMAL_BUFFER {
                    unsafe {
                        normal_row_ptr = normal_row_ptr.add(Framebuffer::TILE_WITH as usize);
                    }
                }
            }
        }
        statistics
    }

    pub fn statistics(&self) -> RasterizerStatistics {
        self.stats
    }

    pub fn set_debug_coloring(&mut self, debug_coloring: bool) {
        self.debug_coloring = debug_coloring;
    }
}

fn debug_color(idx: u32) -> Vec4 {
    fn hash(mut x: u32) -> u32 {
        x = (x ^ 61) ^ (x >> 16);
        x = x.wrapping_add(x << 3);
        x ^= x >> 4;
        x = x.wrapping_mul(0x27d4eb2d);
        x ^ (x >> 15)
    }
    let h = hash(idx);
    let r = (h & 0xff) as u8;
    let g = ((h >> 8) & 0xff) as u8;
    let b = ((h >> 16) & 0xff) as u8;
    Vec4::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
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
            texture: None,
            sampling_filter: SamplerFilter::Nearest,
            alpha_blending: false,
        }
    }
}

impl Default for ScheduledCommand {
    fn default() -> Self {
        ScheduledCommand { texture: None, sampling_filter: SamplerFilter::Nearest, alpha_blending: false }
    }
}

impl PartialEq for ScheduledCommand {
    fn eq(&self, other: &Self) -> bool {
        if self.sampling_filter != other.sampling_filter {
            return false;
        }
        if self.alpha_blending != other.alpha_blending {
            return false;
        }
        match (&self.texture, &other.texture) {
            (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl Eq for ScheduledCommand {}

impl Default for PerTileStatistics {
    fn default() -> Self {
        Self { fragments_drawn: 0 }
    }
}

impl Add for PerTileStatistics {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { fragments_drawn: self.fragments_drawn + other.fragments_drawn }
    }
}

impl RasterizerStatistics {
    pub fn new() -> Self {
        Self { committed_triangles: 0, scheduled_triangles: 0, binned_triangles: 0, fragments_drawn: 0 }
    }

    pub fn smoothed(&self, alpha: usize, prev_smooth: RasterizerStatistics) -> Self {
        assert!(alpha <= 100);
        let alpha1 = 100 - alpha;
        let smooth = |curr: usize, prev: usize| ((alpha * curr) + (alpha1 * prev)) / 100;
        RasterizerStatistics {
            committed_triangles: smooth(self.committed_triangles, prev_smooth.committed_triangles),
            scheduled_triangles: smooth(self.scheduled_triangles, prev_smooth.scheduled_triangles),
            binned_triangles: smooth(self.binned_triangles, prev_smooth.binned_triangles),
            fragments_drawn: smooth(self.fragments_drawn, prev_smooth.fragments_drawn),
        }
    }
}

impl Default for RasterizerStatistics {
    fn default() -> Self {
        Self::new()
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
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
        img1.save(actual_path).unwrap();
    }

    fn save_normals_next_to_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let mut actual_path = reference_path(reference);
        actual_path.set_extension("actual.png");

        let raw_rgba: Vec<u8> = result
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], 255]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
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
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();
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
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();

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

    fn compare_normals_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) -> bool {
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
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(result.width as u32, result.height as u32, raw_rgba).unwrap();

        let img2: RgbaImage = image::open(reference_path).unwrap().into_rgba8();

        if img1.dimensions() != img2.dimensions() {
            return false;
        }

        img1.pixels().zip(img2.pixels()).all(|(p1, p2)| {
            let diff_r = (p1[0] as i16 - p2[0] as i16).abs() as u8;
            let diff_g = (p1[1] as i16 - p2[1] as i16).abs() as u8;
            let diff_b = (p1[2] as i16 - p2[2] as i16).abs() as u8;
            diff_r <= ERROR_TOLERANCE && diff_g <= ERROR_TOLERANCE && diff_b <= ERROR_TOLERANCE
        })
    }

    fn compare_depth_against_reference<P: AsRef<Path>>(result: &Buffer<u16>, reference: P) -> bool {
        let reference_path = reference_path(reference);
        let reference_image: RgbaImage = image::open(reference_path).unwrap().into_rgba8();
        if reference_image.width() != result.width as u32 || reference_image.height() != result.height as u32 {
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

    fn assert_normals_against_reference<P: AsRef<Path>>(result: &Buffer<u32>, reference: P) {
        let equal = compare_normals_against_reference(result, &reference);
        if !equal {
            save_normals_next_to_reference(result, &reference);
        }
        assert!(equal);
    }

    fn render_to_64x64_albedo(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    fn render_to_64x64_albedo_wbg(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(255, 255, 255, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    fn render_to_256x256_albedo(command: &RasterizationCommand) -> Buffer<u32> {
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(256, 256);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 256, 256));
        rasterizer.commit(&command);
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
        color_buffer.as_flat_buffer()
    }

    // render_to_64x64_normals?

    fn render_to_64x64_depth(command: &RasterizationCommand) -> Buffer<u16> {
        // let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        // color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(64, 64);
        depth_buffer.fill(65535);
        let mut framebuffer = Framebuffer::default();
        // framebuffer.color_buffer = Some(&mut color_buffer);
        framebuffer.depth_buffer = Some(&mut depth_buffer);

        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);

        depth_buffer.as_flat_buffer()
    }

    #[rstest]
    #[case(Vec4::new(0.0, 0.0, 0.0, 1.0), "rasterizer/triangle/simple/black.png")]
    #[case(Vec4::new(1.0, 1.0, 1.0, 1.0), "rasterizer/triangle/simple/white.png")]
    #[case(Vec4::new(1.0, 0.0, 0.0, 1.0), "rasterizer/triangle/simple/red.png")]
    #[case(Vec4::new(0.0, 1.0, 0.0, 1.0), "rasterizer/triangle/simple/green.png")]
    #[case(Vec4::new(0.0, 0.0, 1.0, 1.0), "rasterizer/triangle/simple/blue.png")]
    #[case(Vec4::new(1.0, 1.0, 0.0, 1.0), "rasterizer/triangle/simple/yellow.png")]
    #[case(Vec4::new(1.0, 0.0, 1.0, 1.0), "rasterizer/triangle/simple/purple.png")]
    #[case(Vec4::new(0.0, 1.0, 1.0, 1.0), "rasterizer/triangle/simple/cyan.png")]
    fn triangle_simple(#[case] color: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            color,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_0.png")]
    #[case(Vec2::new(-0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_2.png")]
    #[case(Vec2::new(-0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_3.png")]
    #[case(Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_4.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_6.png")]
    #[case(Vec2::new(0.75, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_7.png")]
    #[case(Vec2::new(1.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/top_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.25, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(0.75, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(1.0, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/bottom_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 1.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.75), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_3.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_4.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.25), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_5.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_6.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -0.75), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_7.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, -1.0), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/left_8.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 1.0), "rasterizer/triangle/orientation/right_0.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.75), "rasterizer/triangle/orientation/right_1.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5), "rasterizer/triangle/orientation/right_2.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.25), "rasterizer/triangle/orientation/right_3.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.0), "rasterizer/triangle/orientation/right_4.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.25), "rasterizer/triangle/orientation/right_5.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/triangle/orientation/right_6.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.75), "rasterizer/triangle/orientation/right_7.png")]
    #[case(Vec2::new(-0.5, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -1.0), "rasterizer/triangle/orientation/right_8.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_0.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_1.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.5, 0.0), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_2.png")]
    #[case(Vec2::new(0.5, 0.5), Vec2::new(-0.75, 0.0), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_3.png")]
    #[case(Vec2::new(0.25, 0.5), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_4.png")]
    #[case(Vec2::new(0.25, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_5.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_6.png")]
    #[case(Vec2::new(0.5, 1.0), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_7.png")]
    #[case(Vec2::new(0.5, 0.75), Vec2::new(-0.5, 0.25), Vec2::new(-0.25, -0.25), "rasterizer/triangle/orientation/other_8.png")]
    fn triangle_orientation(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.75), Vec2::new(1.0, -1.0), "rasterizer/triangle/thin/00.png")]
    #[case(Vec2::new(-0.75, 1.0), Vec2::new(-1.0, 1.0), Vec2::new(1.0, -1.0), "rasterizer/triangle/thin/01.png")]
    #[case(Vec2::new(1.0, 1.0), Vec2::new(0.75, 1.0), Vec2::new(-1.0, -1.0), "rasterizer/triangle/thin/02.png")]
    #[case(Vec2::new(1.0, 0.75), Vec2::new(1.0, 1.0), Vec2::new(-1.0, -1.0), "rasterizer/triangle/thin/03.png")]
    #[case(Vec2::new(1.0, -0.75), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0), "rasterizer/triangle/thin/04.png")]
    #[case(Vec2::new(1.0, -1.0), Vec2::new(0.75, -1.0), Vec2::new(-1.0, 1.0), "rasterizer/triangle/thin/05.png")]
    #[case(Vec2::new(-0.75, -1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/triangle/thin/06.png")]
    #[case(Vec2::new(-1.0, -1.0), Vec2::new(-1.0, -0.75), Vec2::new(1.0, 1.0), "rasterizer/triangle/thin/07.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(0.0, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/08.png")]
    #[case(Vec2::new(0.25, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/09.png")]
    #[case(Vec2::new(0.0, 1.0), Vec2::new(-0.25, 1.0), Vec2::new(0.0, -1.0), "rasterizer/triangle/thin/10.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/11.png")]
    #[case(Vec2::new(-1.0, 0.25), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/12.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/13.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.0, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/14.png")]
    #[case(Vec2::new(-0.25, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/15.png")]
    #[case(Vec2::new(0.0, -1.0), Vec2::new(0.25, -1.0), Vec2::new(0.0, 1.0), "rasterizer/triangle/thin/16.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.0), "rasterizer/triangle/thin/17.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, -0.25), Vec2::new(1.0, 0.25), "rasterizer/triangle/thin/18.png")]
    #[case(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.25), "rasterizer/triangle/thin/19.png")]
    fn triangle_thin(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec2::new(0.600891411, 0.600891411), Vec2::new(-0.600891411, -0.600891411), Vec2::new(0.600891411, -0.600891411), "rasterizer/triangle/fract/01.png")]
    #[case(Vec2::new(-0.600891411, 0.600891411), Vec2::new(-0.600891411, -0.600891411), Vec2::new(0.600891411, 0.600891411), "rasterizer/triangle/fract/02.png")]
    fn triangle_fract(#[case] v0: Vec2, #[case] v1: Vec2, #[case] v2: Vec2, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_256x256_albedo(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_00.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.75), "rasterizer/interpolation/depth/large_01.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.5), "rasterizer/interpolation/depth/large_02.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.25), "rasterizer/interpolation/depth/large_03.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 0.0), "rasterizer/interpolation/depth/large_04.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.25), "rasterizer/interpolation/depth/large_05.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.5), "rasterizer/interpolation/depth/large_06.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -0.75), "rasterizer/interpolation/depth/large_07.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer/interpolation/depth/large_08.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_09.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_10.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_11.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_12.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.25), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_13.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.5), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_14.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -0.75), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_15.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_16.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_17.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_18.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_19.png")]
    #[case(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_20.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.25), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_21.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.5), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_22.png")]
    #[case(Vec3::new(-1.0, 1.0, -0.75), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_23.png")]
    #[case(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), "rasterizer/interpolation/depth/large_24.png")]
    #[case(Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, -1.0), "rasterizer/interpolation/depth/large_25.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_00.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_01.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_02.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_03.png")]
    #[case(Vec3::new(-0.75, -0.75, 0.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_04.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.25), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_05.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.5), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_06.png")]
    #[case(Vec3::new(-0.75, -0.75, -0.75), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_07.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_08.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_09.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_10.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_11.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_12.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.25), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_13.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.5), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_14.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -0.75), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_15.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, -1.0), Vec3::new(0.5, 0.75, 1.0), "rasterizer/interpolation/depth/tilted_16.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.75), "rasterizer/interpolation/depth/tilted_17.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.5), "rasterizer/interpolation/depth/tilted_18.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.25), "rasterizer/interpolation/depth/tilted_19.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, 0.0), "rasterizer/interpolation/depth/tilted_20.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.25), "rasterizer/interpolation/depth/tilted_21.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.5), "rasterizer/interpolation/depth/tilted_22.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -0.75), "rasterizer/interpolation/depth/tilted_23.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 1.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer/interpolation/depth/tilted_24.png")]
    #[case(Vec3::new(-0.75, -0.75, 1.0), Vec3::new(0.25, -0.25, 0.0), Vec3::new(0.5, 0.75, -1.0), "rasterizer/interpolation/depth/tilted_25.png")]
    fn depth_interpolation(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            projection: Mat44::orthographic(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            ..Default::default()
        };
        assert_depth_against_reference(&render_to_64x64_depth(&command), filename);
    }

    #[rstest]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_0.png")]
    #[case(Vec3::new(-1.75, -1.75, -3.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_1.png")]
    #[case(Vec3::new(-3.75, -3.75, -7.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_2.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(1.75, -1.75, -3.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_3.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(3.75, -3.75, -7.5), Vec3::new(0.0, 0.75, -1.5), "rasterizer/interpolation/color/simple_4.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 1.75, -3.5), "rasterizer/interpolation/color/simple_5.png")]
    #[case(Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 3.75, -7.5), "rasterizer/interpolation/color/simple_6.png")]
    fn color_interpolation_simple(#[case] v0: Vec3, #[case] v1: Vec3, #[case] v2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[v0, v1, v2],
            colors: &[Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)],
            projection: Mat44::perspective(0.1, 10.0, std::f32::consts::PI / 3.0, 1.),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 0.5, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_0.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.5, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_1.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.5, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_2.png"
    )]
    #[case(
        Vec4::new(1.0, 0.2, 0.1, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_3.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.2, 1.0, 0.1, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_4.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.2, 0.1, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_5.png"
    )]
    #[case(
        Vec4::new(1.0, 0.2, 0.1, 1.0),
        Vec4::new(0.2, 1.0, 0.1, 1.0),
        Vec4::new(0.2, 0.1, 1.0, 1.0),
        "rasterizer/interpolation/color/mix_6.png"
    )]
    fn color_interpolation_mix(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(-0.75, -0.75, -1.5), Vec3::new(0.75, -0.75, -1.5), Vec3::new(0.0, 0.75, -1.5)],
            colors: &[c0, c1, c2],
            projection: Mat44::perspective(0.1, 10.0, std::f32::consts::PI / 3.0, 1.),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(Viewport::new(0, 0, 64, 64), "rasterizer/viewport/0.png")]
    #[case(Viewport::new(0, 0, 32, 32), "rasterizer/viewport/1.png")]
    #[case(Viewport::new(32, 0, 64, 32), "rasterizer/viewport/2.png")]
    #[case(Viewport::new(0, 32, 32, 64), "rasterizer/viewport/3.png")]
    #[case(Viewport::new(32, 32, 64, 64), "rasterizer/viewport/4.png")]
    #[case(Viewport::new(0, 0, 32, 64), "rasterizer/viewport/5.png")]
    #[case(Viewport::new(32, 0, 64, 64), "rasterizer/viewport/6.png")]
    #[case(Viewport::new(0, 0, 64, 32), "rasterizer/viewport/7.png")]
    #[case(Viewport::new(0, 32, 64, 64), "rasterizer/viewport/8.png")]
    fn viewport(#[case] v: Viewport, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(v);
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_albedo_against_reference(&color_buffer.as_flat_buffer(), filename);
    }

    #[rstest]
    #[case(256, 256, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/256x256_00.png")]
    #[case(256, 256, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/256x256_01.png")]
    #[case(256, 256, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/256x256_02.png")]
    #[case(256, 256, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/256x256_03.png")]
    #[case(256, 256, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/256x256_04.png")]
    #[case(256, 256, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/256x256_05.png")]
    #[case(256, 256, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/256x256_06.png")]
    #[case(256, 256, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/256x256_07.png")]
    #[case(256, 256, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/256x256_08.png")]
    #[case(256, 256, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/256x256_09.png")]
    #[case(256, 256, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/256x256_10.png")]
    #[case(141, 79, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/141x79_00.png")]
    #[case(141, 79, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/141x79_01.png")]
    #[case(141, 79, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/141x79_02.png")]
    #[case(141, 79, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/141x79_03.png")]
    #[case(141, 79, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/141x79_04.png")]
    #[case(141, 79, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/141x79_05.png")]
    #[case(141, 79, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/141x79_06.png")]
    #[case(141, 79, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/141x79_07.png")]
    #[case(141, 79, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/141x79_08.png")]
    #[case(141, 79, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/141x79_09.png")]
    #[case(141, 79, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/141x79_10.png")]
    #[case(65, 65, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/65x65_00.png")]
    #[case(65, 65, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/65x65_01.png")]
    #[case(65, 65, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/65x65_02.png")]
    #[case(65, 65, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/65x65_03.png")]
    #[case(65, 65, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/65x65_04.png")]
    #[case(65, 65, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/65x65_05.png")]
    #[case(65, 65, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/65x65_06.png")]
    #[case(65, 65, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/65x65_07.png")]
    #[case(65, 65, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/65x65_08.png")]
    #[case(65, 65, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/65x65_09.png")]
    #[case(65, 65, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/65x65_10.png")]
    #[case(1024, 1024, Vec2::new(0.0, 0.5), Vec2::new(-0.5, -0.5), Vec2::new(0.5, -0.5), "rasterizer/tiling/1024x1024_00.png")]
    #[case(1024, 1024, Vec2::new(-0.5, 0.75), Vec2::new(-0.75, -0.75), Vec2::new(-0.25, -0.75), "rasterizer/tiling/1024x1024_01.png")]
    #[case(1024, 1024, Vec2::new(0.5, 0.75), Vec2::new(0.25, -0.75), Vec2::new(0.75, -0.75), "rasterizer/tiling/1024x1024_02.png")]
    #[case(1024, 1024, Vec2::new(-0.75, 0.75), Vec2::new(-0.75, 0.25), Vec2::new(0.75, 0.5), "rasterizer/tiling/1024x1024_03.png")]
    #[case(1024, 1024, Vec2::new(-0.75, -0.25), Vec2::new(-0.75, -0.75), Vec2::new(0.75, -0.5), "rasterizer/tiling/1024x1024_04.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.0), "rasterizer/tiling/1024x1024_05.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 0.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/1024x1024_06.png")]
    #[case(1024, 1024, Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -2.0), Vec2::new(1.0, 1.0), "rasterizer/tiling/1024x1024_07.png")]
    #[case(1024, 1024, Vec2::new(1.0, 1.25), Vec2::new(-1.0, 0.0), Vec2::new(1.0, -1.25), "rasterizer/tiling/1024x1024_08.png")]
    #[case(1024, 1024, Vec2::new(0.25, 0.75), Vec2::new(-0.75, -0.25), Vec2::new(-0.25, -0.25), "rasterizer/tiling/1024x1024_09.png")]
    #[case(1024, 1024, Vec2::new(-0.85, 0.75), Vec2::new(0.65, -0.35), Vec2::new(0.85, -0.25), "rasterizer/tiling/1024x1024_10.png")]
    fn tiling(
        #[case] width: u16,
        #[case] height: u16,
        #[case] v0: Vec2,
        #[case] v1: Vec2,
        #[case] v2: Vec2,
        #[case] filename: &str,
    ) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(v0.x, v0.y, 0.0), Vec3::new(v1.x, v1.y, 0.0), Vec3::new(v2.x, v2.y, 0.0)],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(width, height);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, width, height));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_albedo_against_reference(&color_buffer.as_flat_buffer(), filename);
    }

    #[rstest]
    #[case(
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_0.png"
    )]
    #[case(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        "rasterizer/interpolation/normal/simple_1.png"
    )]
    #[case(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        "rasterizer/interpolation/normal/simple_2.png"
    )]
    #[case(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -1.0),
        "rasterizer/interpolation/normal/simple_3.png"
    )]
    #[case(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        "rasterizer/interpolation/normal/simple_4.png"
    )]
    #[case(
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        "rasterizer/interpolation/normal/simple_5.png"
    )]
    #[case(
        Vec3::new(0.3, 0.3, 0.9),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_6.png"
    )]
    #[case(
        Vec3::new(-0.5, 0.5, 0.8),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        "rasterizer/interpolation/normal/simple_7.png"
    )]
    #[case(
        Vec3::new(-0.5, 0.7, 0.6),
        Vec3::new(0.2, -0.4, 0.9),
        Vec3::new(0.9, 0.2, -0.3),
        "rasterizer/interpolation/normal/simple_8.png"
    )]
    fn normal_interpolation_simple(#[case] n0: Vec3, #[case] n1: Vec3, #[case] n2: Vec3, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            normals: &[n0, n1, n2],
            ..Default::default()
        };
        let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        let mut normal_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
        normal_buffer.fill(0);
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        framebuffer.normal_buffer = Some(&mut normal_buffer);
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(Viewport::new(0, 0, 64, 64));
        rasterizer.commit(&command);
        rasterizer.draw(&mut framebuffer);
        assert_normals_against_reference(&normal_buffer.as_flat_buffer(), filename);
    }

    fn checkerboard_rgb_texture_32x32() -> std::sync::Arc<Texture> {
        let width = 32;
        let height = 32;
        let mut texels = vec![0u8; width * height * 3];
        for y in 0..width {
            for x in 0..height {
                let offset = ((y * width + x) * 3) as usize;
                if (x + y) % 2 == 0 {
                    texels[offset] = (x * 8) as u8; // R
                    texels[offset + 1] = (y * 8) as u8; // G
                    texels[offset + 2] = ((62 - x - y) * 4) as u8; // B
                } else {
                    texels[offset] = 127;
                    texels[offset + 1] = 127;
                    texels[offset + 2] = 127;
                }
            }
        }
        let source =
            TextureSource { texels: &texels, width: width as u32, height: height as u32, format: TextureFormat::RGB };
        Texture::new(&source)
    }

    #[rstest]
    #[case(&[Vec3::new(-0.5, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0),
             Vec3::new(0.5, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_0.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_1.png"
    )]
    #[case(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
             Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_2.png"
    )]
    #[case(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
             Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0),],
           &[Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0),
             Vec2::new(1.0, 1.0), Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0),],
        "rasterizer/texturing/nearest_3.png"
    )]
    #[case(&[Vec3::new(-1.0, 0.5, 0.0), Vec3::new(-1.0, -0.5, 0.0), Vec3::new(1.0, 0.5, 0.0),
             Vec3::new(1.0, 0.5, 0.0), Vec3::new(-1.0, -0.5, 0.0), Vec3::new(1.0, -0.5, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_4.png"
    )]
    #[case(&[Vec3::new(-0.5, 1.0, 0.0), Vec3::new(-0.5, -1.0, 0.0), Vec3::new(0.5, 1.0, 0.0),
             Vec3::new(0.5, 1.0, 0.0), Vec3::new(-0.5, -1.0, 0.0), Vec3::new(0.5, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0),
             Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0),],
        "rasterizer/texturing/nearest_5.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 2.0), Vec2::new(2.0, 0.0),
             Vec2::new(2.0, 0.0), Vec2::new(0.0, 2.0), Vec2::new(2.0, 2.0),],
        "rasterizer/texturing/nearest_6.png"
    )]
    #[case(&[Vec3::new(-1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
             Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0),],
           &[Vec2::new(-0.5, -0.5), Vec2::new(-0.5, 1.5), Vec2::new(1.5, -0.5),
             Vec2::new(1.5, -0.5), Vec2::new(-0.5, 1.5), Vec2::new(1.5, 1.5),],
        "rasterizer/texturing/nearest_7.png"
    )]
    fn texturing_nearest(#[case] world_positions: &[Vec3], #[case] tex_coords: &[Vec2], #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions,
            tex_coords,
            texture: Some(checkerboard_rgb_texture_32x32()),
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0)],
        "rasterizer/texturing/mip_selection_00.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -0.6), Vec2::new(0.6, 1.0)],
        "rasterizer/texturing/mip_selection_01.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, -0.4), Vec2::new(0.4, 1.0)],
        "rasterizer/texturing/mip_selection_02.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(0.0, 1.0)],
        "rasterizer/texturing/mip_selection_03.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.2), Vec2::new(-0.2, 1.0)],
        "rasterizer/texturing/mip_selection_04.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.3), Vec2::new(-0.3, 1.0)],
        "rasterizer/texturing/mip_selection_05.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.5), Vec2::new(-0.5, 1.0)],
        "rasterizer/texturing/mip_selection_06.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.6), Vec2::new(-0.6, 1.0)],
        "rasterizer/texturing/mip_selection_07.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.65), Vec2::new(-0.65, 1.0)],
        "rasterizer/texturing/mip_selection_08.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.75), Vec2::new(-0.75, 1.0)],
        "rasterizer/texturing/mip_selection_09.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.87), Vec2::new(-0.87, 1.0)],
        "rasterizer/texturing/mip_selection_10.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.93), Vec2::new(-0.93, 1.0)],
        "rasterizer/texturing/mip_selection_11.png"
    )]
    #[case(&[Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 0.96), Vec2::new(-0.96, 1.0)],
        "rasterizer/texturing/mip_selection_12.png"
    )]
    fn texturing_mip_selection(#[case] positions: &[Vec2], #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![255u8; 64 * 64],
            width: 64,
            height: 64,
            format: TextureFormat::Grayscale,
        });
        let command = RasterizationCommand {
            world_positions: &[
                Vec3::new(positions[0].x, positions[0].y, 0.0),
                Vec3::new(positions[1].x, positions[1].y, 0.0),
                Vec3::new(positions[2].x, positions[2].y, 0.0),
            ],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            sampling_filter: SamplerFilter::DebugMip,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_24.png"
    )]
    fn alpha_blend_simple(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            colors: &[c0, c1, c2],
            alpha_blending: true,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_00.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 0.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_wbg_01.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 0.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_wbg_02.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 0.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_wbg_03.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 0.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_wbg_04.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 0.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_wbg_05.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 0.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_wbg_06.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 0.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_wbg_07.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 0.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_wbg_08.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 0.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_wbg_09.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/vert_simple_wbg_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/vert_simple_wbg_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/vert_simple_wbg_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/vert_simple_wbg_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/vert_simple_wbg_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/vert_simple_wbg_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/vert_simple_wbg_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/vert_simple_wbg_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/vert_simple_wbg_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_21.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_22.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        "rasterizer/alpha_blend/vert_simple_wbg_23.png"
    )]
    #[case(
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
        "rasterizer/alpha_blend/vert_simple_wbg_24.png"
    )]
    fn alpha_blend_simple_wbg(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            colors: &[c0, c1, c2],
            alpha_blending: true,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_red_solid_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_red_solid_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_red_solid_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_red_solid_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_red_solid_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_red_solid_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_red_solid_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_red_solid_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_red_solid_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_red_solid_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_red_solid_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_red_solid_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_red_solid_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_red_solid_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_red_solid_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_red_solid_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_red_solid_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_red_solid_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_red_solid_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_red_solid_24.png"
    )]
    fn alpha_blend_tex_red_solid(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![255u8, 0u8, 0u8],
            width: 1,
            height: 1,
            format: TextureFormat::RGB,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: true,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_mixed_solid_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_mixed_solid_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_mixed_solid_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_mixed_solid_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_mixed_solid_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_mixed_solid_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_mixed_solid_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_mixed_solid_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_mixed_solid_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_mixed_solid_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_mixed_solid_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_mixed_solid_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_mixed_solid_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_mixed_solid_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_mixed_solid_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_mixed_solid_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_mixed_solid_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_mixed_solid_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_mixed_solid_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_mixed_solid_24.png"
    )]
    fn alpha_blend_tex_mixed_solid(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![0x0Bu8, 0xDAu8, 0x51u8],
            width: 1,
            height: 1,
            format: TextureFormat::RGB,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: true,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }

    #[rstest]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_00.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        Vec4::new(1.0, 1.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_purple_half_01.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        Vec4::new(1.0, 1.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_purple_half_02.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        Vec4::new(1.0, 1.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_purple_half_03.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        Vec4::new(1.0, 1.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_purple_half_04.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        Vec4::new(1.0, 1.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_purple_half_05.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        Vec4::new(1.0, 1.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_purple_half_06.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        Vec4::new(1.0, 1.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_purple_half_07.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        Vec4::new(1.0, 1.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_purple_half_08.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        Vec4::new(1.0, 1.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_purple_half_09.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_10.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_11.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.9),
        Vec4::new(0.0, 1.0, 0.0, 0.9),
        Vec4::new(0.0, 0.0, 1.0, 0.9),
        "rasterizer/alpha_blend/tex_purple_half_12.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.8),
        Vec4::new(0.0, 1.0, 0.0, 0.8),
        Vec4::new(0.0, 0.0, 1.0, 0.8),
        "rasterizer/alpha_blend/tex_purple_half_13.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.7),
        Vec4::new(0.0, 1.0, 0.0, 0.7),
        Vec4::new(0.0, 0.0, 1.0, 0.7),
        "rasterizer/alpha_blend/tex_purple_half_14.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.6),
        Vec4::new(0.0, 1.0, 0.0, 0.6),
        Vec4::new(0.0, 0.0, 1.0, 0.6),
        "rasterizer/alpha_blend/tex_purple_half_15.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.5),
        Vec4::new(0.0, 1.0, 0.0, 0.5),
        Vec4::new(0.0, 0.0, 1.0, 0.5),
        "rasterizer/alpha_blend/tex_purple_half_16.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.4),
        Vec4::new(0.0, 1.0, 0.0, 0.4),
        Vec4::new(0.0, 0.0, 1.0, 0.4),
        "rasterizer/alpha_blend/tex_purple_half_17.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.3),
        Vec4::new(0.0, 1.0, 0.0, 0.3),
        Vec4::new(0.0, 0.0, 1.0, 0.3),
        "rasterizer/alpha_blend/tex_purple_half_18.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.2),
        Vec4::new(0.0, 1.0, 0.0, 0.2),
        Vec4::new(0.0, 0.0, 1.0, 0.2),
        "rasterizer/alpha_blend/tex_purple_half_19.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.1),
        Vec4::new(0.0, 1.0, 0.0, 0.1),
        Vec4::new(0.0, 0.0, 1.0, 0.1),
        "rasterizer/alpha_blend/tex_purple_half_20.png"
    )]
    #[case(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_21.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_22.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        "rasterizer/alpha_blend/tex_purple_half_23.png"
    )]
    #[case(
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 0.0),
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        "rasterizer/alpha_blend/tex_purple_half_24.png"
    )]
    fn alpha_blend_tex_purple_half(#[case] c0: Vec4, #[case] c1: Vec4, #[case] c2: Vec4, #[case] filename: &str) {
        let texture = Texture::new(&TextureSource {
            texels: &vec![0x93u8, 0x70u8, 0xDBu8, 0x7Fu8],
            width: 1,
            height: 1,
            format: TextureFormat::RGBA,
        });
        let command = RasterizationCommand {
            world_positions: &[Vec3::new(0.0, 0.5, 0.0), Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, -0.5, 0.0)],
            tex_coords: &[Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)],
            texture: Some(texture),
            colors: &[c0, c1, c2],
            alpha_blending: true,
            ..Default::default()
        };
        assert_albedo_against_reference(&render_to_64x64_albedo_wbg(&command), filename);
    }
}

#[cfg(test)]
mod tests_binning {
    use super::*;

    #[test]
    fn binning() {
        struct BinningTC {
            v0: Vec2,
            v1: Vec2,
            v2: Vec2,
            mask: u32,
        }
        let test_cases = vec![
            // 1 tile
            BinningTC { mask: 0b0001, v0: Vec2::new(-0.9, 0.9), v1: Vec2::new(-0.9, 0.8), v2: Vec2::new(-0.8, 0.9) },
            BinningTC { mask: 0b0010, v0: Vec2::new(0.8, 0.9), v1: Vec2::new(0.8, 0.8), v2: Vec2::new(0.9, 0.9) },
            BinningTC { mask: 0b0100, v0: Vec2::new(-0.9, -0.8), v1: Vec2::new(-0.9, -0.9), v2: Vec2::new(-0.8, -0.8) },
            BinningTC { mask: 0b1000, v0: Vec2::new(0.8, -0.8), v1: Vec2::new(0.8, -0.9), v2: Vec2::new(0.9, -0.8) },
            // 2 tiles
            BinningTC { mask: 0b0011, v0: Vec2::new(-0.9, 0.9), v1: Vec2::new(-0.9, 0.8), v2: Vec2::new(0.8, 0.9) },
            BinningTC { mask: 0b0101, v0: Vec2::new(-0.9, 0.9), v1: Vec2::new(-0.9, -0.8), v2: Vec2::new(-0.8, 0.9) },
            BinningTC { mask: 0b1010, v0: Vec2::new(0.8, 0.9), v1: Vec2::new(0.8, -0.8), v2: Vec2::new(0.9, 0.9) },
            BinningTC { mask: 0b1100, v0: Vec2::new(-0.9, -0.8), v1: Vec2::new(-0.9, -0.9), v2: Vec2::new(0.8, -0.8) },
            // 3 tiles
            BinningTC { mask: 0b0111, v0: Vec2::new(-0.9, 0.9), v1: Vec2::new(-0.9, -0.8), v2: Vec2::new(0.1, 0.9) },
            BinningTC { mask: 0b0111, v0: Vec2::new(-2.0, 2.0), v1: Vec2::new(-2.0, -2.0), v2: Vec2::new(2.0, 2.0) },
        ];
        for tc in test_cases {
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 120, 100));
            rasterizer.commit(&RasterizationCommand {
                world_positions: &[
                    Vec3::new(tc.v0.x, tc.v0.y, 0.0),
                    Vec3::new(tc.v1.x, tc.v1.y, 0.0),
                    Vec3::new(tc.v2.x, tc.v2.y, 0.0),
                ],
                ..Default::default()
            });
            let mask = ((!rasterizer.tiles[0].triangles.is_empty()) as u32) << 0
                | ((!rasterizer.tiles[1].triangles.is_empty()) as u32) << 1
                | ((!rasterizer.tiles[2].triangles.is_empty()) as u32) << 2
                | ((!rasterizer.tiles[3].triangles.is_empty()) as u32) << 3;
            assert_eq!(mask, tc.mask);
        }
    }
}

#[cfg(test)]
mod tests_watertight {
    use super::*;
    use image::{ImageBuffer, Rgba};
    use std::path::Path;

    fn save_image<P: AsRef<Path>>(path: &P, image: &Buffer<u32>) {
        let raw_rgba: Vec<u8> = image
            .elems
            .iter()
            .flat_map(|&pixel| {
                let bytes = pixel.to_le_bytes();
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            })
            .collect();
        let img1: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(image.width as u32, image.height as u32, raw_rgba).unwrap();
        img1.save(path).unwrap();
    }

    #[test]
    fn fullscreen_quad() {
        // v0--v2v3|
        // |   //  |
        // |  //   |
        // | //    |
        // v1v4---v5
        let wp1 = vec![
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ];
        // v0v5--v4
        // | \\   |
        // |  \\  |
        // |   \\ |
        // v1--v2v3
        let wp2 = vec![
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let colors = vec![
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
            Vec4::new(0.0, 1.0, 0.0, 1.0),
        ];
        let mut rasterizer = Rasterizer::new();
        for dim in 1..=512 {
            for wp in [&wp1, &wp2] {
                let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(dim as u16, dim as u16);
                color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
                rasterizer.setup(Viewport::new(0, 0, dim as u16, dim as u16));
                rasterizer.commit(&RasterizationCommand { world_positions: wp, colors: &colors, ..Default::default() });
                rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });
                let flat = color_buffer.as_flat_buffer();
                let tight = flat.elems.iter().all(|&x| {
                    let c = RGBA::from_u32(x);
                    c.r > 0 || c.g > 0 || c.b > 0
                });
                if !tight {
                    save_image(
                        &Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join(format!("tests/fullscreen_quad_{0}x{0}.actual.png", dim)),
                        &flat,
                    );
                }
                assert!(tight);
            }
        }
    }
}

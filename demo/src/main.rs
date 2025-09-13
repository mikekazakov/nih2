extern crate sdl3;

use image::RgbaImage;
use nih::math::*;
use nih::render::*;
use nih::util::*;
use parking_lot::ReentrantMutex;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use nih::render::rgba::RGBA;
use nih::util::profiler::Profiler;
use once_cell::sync::Lazy;
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Mod};
use sdl3::pixels::PixelFormatEnum;
use sdl3::rect::Rect;
use sdl3::surface::Surface;

mod io;

static PROFILER: Lazy<ReentrantMutex<Profiler>> = Lazy::new(|| ReentrantMutex::new(Profiler::new()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayMode {
    Color,
    Depth,
    Normal,
}

struct State {
    color_buffer: TiledBuffer<u32, 64, 64>,
    depth_buffer: TiledBuffer<u16, 64, 64>,
    normal_buffer: TiledBuffer<u32, 64, 64>,
    rasterizer: Rasterizer,
    rasterizer_stats: RasterizerStatistics,
    mesh: MeshData,
    mesh2: MeshData,
    meshes: HashMap<String, MeshData>,
    textures: HashMap<String, std::sync::Arc<Texture>>,
    texture_filtering: SamplerFilter,
    display_mode: DisplayMode,
    overlay_tiles: bool,
    timestamp: Instant,
    t: Duration,
    dt: Duration,
    last_printout: Instant,
}

impl Default for State {
    fn default() -> Self {
        State {
            color_buffer: TiledBuffer::<u32, 64, 64>::new(1, 1),
            depth_buffer: TiledBuffer::<u16, 64, 64>::new(1, 1),
            normal_buffer: TiledBuffer::<u32, 64, 64>::new(1, 1),
            rasterizer: Rasterizer::new(),
            rasterizer_stats: RasterizerStatistics::default(),
            mesh: MeshData::default(),
            mesh2: MeshData::default(),
            meshes: HashMap::new(),
            textures: HashMap::new(),
            texture_filtering: SamplerFilter::Bilinear,
            display_mode: DisplayMode::Color,
            overlay_tiles: false,
            timestamp: Instant::now(),
            t: Duration::from_secs(0),
            dt: Duration::from_secs(0),
            last_printout: Instant::now(),
        }
    }
}

fn blit_to_window(buffer: &mut Buffer<u32>, window: &sdl3::video::Window, event_pump: &sdl3::EventPump) {
    let width = buffer.width as u32;
    let height = buffer.height as u32;
    let pitch = (buffer.stride * 4) as u32;
    let buffer_surface =
        Surface::from_data(buffer.as_u8_slice_mut(), width, height, pitch, PixelFormatEnum::ABGR8888.into()).unwrap();

    let mut windows_surface = window.surface(&event_pump).unwrap();
    assert_eq!(windows_surface.width(), width);
    assert_eq!(windows_surface.height(), height);
    let rect = Rect::new(0, 0, width, height);
    buffer_surface.blit(rect, &mut windows_surface, rect).unwrap();
    windows_surface.finish().unwrap();
}

fn blit_depth_to_window(buffer: &Buffer<u16>, window: &sdl3::video::Window, event_pump: &sdl3::EventPump) {
    // let mut max = 0;
    // let mut min = 65535;
    // buffer.elems.iter().for_each(|&x| {
    //     if x > max && x != 65535 {
    //         max = x;
    //     }
    //     if x < min {
    //         min = x;
    //     }
    // });
    // let delta = (max - min) as u32;

    let width = buffer.width as u32;
    let height = buffer.height as u32;
    let mut buffer_surface = Surface::new(width, height, PixelFormatEnum::ABGR8888.into()).unwrap();
    let pitch = buffer_surface.pitch() as usize;
    buffer_surface.with_lock_mut(|pixels: &mut [u8]| {
        for y in 0..buffer.height {
            for x in 0..buffer.width {
                let offset = y as usize * pitch + x as usize * 4;
                let depth = buffer.at(x, y);
                if depth == 65535 {
                    // 255u8
                    pixels[offset + 0] = 255; // R
                    pixels[offset + 1] = 200; // G
                    pixels[offset + 2] = 255; // B
                } else {
                    // let gray = (((depth - min) as u32 * 255) / (delta)) as u8;
                    let gray = (((depth) as u32 * 255) / (65534)) as u8;
                    pixels[offset + 0] = gray; // R
                    pixels[offset + 1] = gray; // G
                    pixels[offset + 2] = gray; // B
                };
                pixels[offset + 3] = 255; // A
            }
        }
    });

    let mut windows_surface = window.surface(&event_pump).unwrap();
    assert_eq!(windows_surface.width(), width);
    assert_eq!(windows_surface.height(), height);
    let rect = Rect::new(0, 0, width, height);
    buffer_surface.blit(rect, &mut windows_surface, rect).unwrap();
    windows_surface.finish().unwrap();
}

fn blit_normals_to_window(buffer: &Buffer<u32>, window: &sdl3::video::Window, event_pump: &sdl3::EventPump) {
    let width = buffer.width as u32;
    let height = buffer.height as u32;
    let mut buffer_surface = Surface::new(width, height, PixelFormatEnum::ABGR8888.into()).unwrap();
    let pitch = buffer_surface.pitch() as usize;
    buffer_surface.with_lock_mut(|pixels: &mut [u8]| {
        for y in 0..buffer.height {
            for x in 0..buffer.width {
                let offset = y as usize * pitch + x as usize * 4;
                let n = buffer.at(x, y);
                pixels[offset + 0] = (n & 0xFF) as u8; // R
                pixels[offset + 1] = ((n & 0xFF00) >> 8) as u8; // G
                pixels[offset + 2] = ((n & 0xFF0000) >> 16) as u8; // B
                pixels[offset + 3] = 255; // A
            }
        }
    });

    let mut windows_surface = window.surface(&event_pump).unwrap();
    assert_eq!(windows_surface.width(), width);
    assert_eq!(windows_surface.height(), height);
    let rect = Rect::new(0, 0, width, height);
    buffer_surface.blit(rect, &mut windows_surface, rect).unwrap();
    windows_surface.finish().unwrap();
}

fn render(state: &mut State) {
    let profiler = PROFILER.lock();
    // buf.fill(RGBA::new((tick % 256) as u8, 255, 0, 255));

    // (64, 224, 208)
    state.color_buffer.fill(RGBA::new(64, 224, 208, 255).to_u32());
    state.depth_buffer.fill(u16::MAX);
    state.normal_buffer.fill(RGBA::new(127, 255, 127, 255).to_u32());

    let viewport = Viewport { xmin: 0, ymin: 0, xmax: state.color_buffer.width(), ymax: state.color_buffer.height() };
    let rasterizer = &mut state.rasterizer;
    rasterizer.setup(viewport);
    // rasterizer.set_debug_coloring(true);

    let texture = Texture::new(&TextureSource {
        texels: &[127u8, 255u8, 255u8, 127u8],
        width: 2,
        height: 2,
        format: TextureFormat::Grayscale,
    });

    // let lines = vec![
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(0.5, 0.0, 0.0), //
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(1.0, 1.0, 0.0),
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(-1.0, -1.0, 0.0),
    // ];

    // if false {
    //     let mut mesh_lines = Vec::<nih::math::Vec3>::default();
    //     let num = state.mesh.positions.len() / 3;
    //     // let num = 1;
    //     for i in 0..num {
    //         mesh_lines.push(state.mesh.positions[i * 3 + 0]);
    //         mesh_lines.push(state.mesh.positions[i * 3 + 1]);
    //         mesh_lines.push(state.mesh.positions[i * 3 + 1]);
    //         mesh_lines.push(state.mesh.positions[i * 3 + 2]);
    //         mesh_lines.push(state.mesh.positions[i * 3 + 2]);
    //         mesh_lines.push(state.mesh.positions[i * 3 + 0]);
    //     }
    //
    //     // let lines = nih::math::sphere_to_aa_lines(32);
    //
    //     let mut cmd = DrawLinesCommand::default();
    //     // cmd.lines = &lines;
    //     // // cmd.color = Vec4::new(1.0, 1.0, 0.0, 1.0);
    //     // cmd.color = Vec4::new(1.0, 1.0, 0.0, 0.6);
    //     // cmd.model = Mat34::rotate_yz(tick as f32 / 377.0)
    //     //     * Mat34::rotate_xy(tick as f32 / 177.0)
    //     //     * Mat34::rotate_zx(tick as f32 / 100.0)
    //     //     * Mat34::scale_uniform(0.5);
    //
    //     cmd.lines = &mesh_lines;
    //     // cmd.color = Vec4::new(1.0, 1.0, 0.0, 1.0);
    //     cmd.color = Vec4::new(1.0, 1.0, 0.0, 0.6);
    //     cmd.model = Mat34::rotate_zx(tick as f32 / 100.0)
    //         * Mat34::translate(Vec3::new(0.0, 0.0, 0.0))
    //         * Mat34::scale_uniform(0.5);
    //
    //     let mut framebuffer = Framebuffer::default();
    //     framebuffer.color_buffer = Some(&mut state.color_buffer);
    //     draw_lines(&mut framebuffer, &viewport, &cmd);
    //
    //     let aabb_lines = aabb_to_lines(state.mesh.aabb);
    //     cmd.lines = &aabb_lines;
    //     draw_lines(&mut framebuffer, &viewport, &cmd);
    // }

    {
        // pub struct RasterizationCommand<'a> {
        //     pub world_positions: &'a [Vec3],
        //     pub normals: &'a [Vec3],    // TODO: support deriving the normals?
        //     pub tex_coords: &'a [Vec2], // empty if absent
        //     pub colors: &'a [Vec4],     // empty if absent
        //     pub indices: &'a [u32],
        //     pub model: Mat34,
        //     pub view: Mat44,
        //     pub projection: Mat44,
        // }
        // let mut rasterizer = Rasterizer::new();
        // rasterizer.setup(viewport);

        fn idx_to_color_hash(x: usize) -> Vec4 {
            // Mix the bits using a few bitwise operations and multiplications
            // x ^= x >> 16;
            // x = x.wrapping_mul(0x85ebca6b);
            // x ^= x >> 13;
            // x = x.wrapping_mul(0xc2b2ae35);
            // x ^= x >> 16;
            // x | 0xFF
            if x % 4 == 0 {
                Vec4::new(1.0, 1.0, 1.0, 1.0)
            } else if x % 4 == 1 {
                Vec4::new(1.0, 0.0, 0.0, 1.0)
            } else if x % 4 == 2 {
                Vec4::new(0.0, 1.0, 0.0, 1.0)
            } else {
                Vec4::new(0.0, 0.0, 1.0, 1.0)
            }
        }

        let mut colors1 = Vec::<Vec4>::default();
        for i in 0..state.mesh.positions.len() {
            colors1.push(idx_to_color_hash(i));
        }
        let mut colors2 = Vec::<Vec4>::default();
        for i in 0..state.mesh2.positions.len() {
            colors2.push(idx_to_color_hash(i));
        }

        let mut cmd = RasterizationCommand::default();
        // cmd.view = Mat44::translate(Vec3::new(0.0, 0.0, -4.0));
        cmd.projection =
            Mat44::perspective(1.0, 20.0, std::f32::consts::PI / 3.0, viewport.xmax as f32 / viewport.ymax as f32);
        cmd.culling = CullMode::CW;
        cmd.sampling_filter = state.texture_filtering;

        // {
        //     cmd.world_positions = &state.mesh.positions;
        //     cmd.normals = &state.mesh.normals;
        //     cmd.tex_coords = &state.mesh.tex_coords;
        //     cmd.colors = &colors1;
        //     cmd.indices = &state.mesh.indices;
        //     cmd.model = Mat34::rotate_yz(state.t.as_secs_f32() / 3.77)
        //         * Mat34::rotate_xy(state.t.as_secs_f32() / 1.77)
        //         * Mat34::rotate_zx(state.t.as_secs_f32() / 1.10)
        //         * Mat34::scale_uniform(1.5);
        //     {
        //         let _profile_commit_scope = profiler::ProfileScope::new("commit", &profiler);
        //         rasterizer.commit(&cmd);
        //     }
        // }

        if false {
            let mut colors = Vec::<Vec4>::default();
            for i in 0..state.mesh2.positions.len() {
                colors.push(idx_to_color_hash(i));
            }
            cmd.world_positions = &state.mesh2.positions;
            cmd.normals = &state.mesh2.normals;
            // cmd.normals = &[];
            cmd.tex_coords = &state.mesh2.tex_coords;
            cmd.texture = Some(texture.clone());
            // cmd.colors = &colors2;
            cmd.indices = &state.mesh2.indices;
            cmd.model = Mat34::translate(Vec3::new(0.0, -3.0, -10.0))
                * Mat34::rotate_zx(state.t.as_secs_f32() / 1.10)
                * Mat34::scale_uniform(2.0);
            {
                let _profile_commit_scope = profiler::ProfileScope::new("commit", &profiler);
                rasterizer.commit(&cmd);
                //
                // cmd.model = Mat34::translate(Vec3::new(-4.0, -3.0, -10.0))
                //     * Mat34::rotate_zx(state.t.as_secs_f32() / 1.20)
                //     * Mat34::scale_uniform(2.0);
                // rasterizer.commit(&cmd);
                //
                // cmd.model = Mat34::translate(Vec3::new(4.0, -3.0, -10.0))
                //     * Mat34::rotate_zx(state.t.as_secs_f32() / 1.30)
                //     * Mat34::scale_uniform(2.0);
                // rasterizer.commit(&cmd);
            }
        }
        {
            let mesh = state.meshes.get("Teapot3").unwrap();
            cmd.world_positions = &mesh.positions;
            cmd.normals = &mesh.normals;
            cmd.tex_coords = &mesh.tex_coords;
            // cmd.texture = Some(texture.clone());
            cmd.texture = Some(state.textures.get("Teapot3").unwrap().clone());
            cmd.indices = &mesh.indices;
            cmd.model = Mat34::translate(Vec3::new(0.0, -3.0, -10.0))
                * Mat34::rotate_zx(state.t.as_secs_f32() / 1.10)
                * Mat34::scale_uniform(0.08);
            let _profile_commit_scope = profiler::ProfileScope::new("commit", &profiler);
            rasterizer.commit(&cmd);
        }

        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut state.color_buffer);
        framebuffer.depth_buffer = Some(&mut state.depth_buffer);
        framebuffer.normal_buffer = Some(&mut state.normal_buffer);
        {
            let _profile_draw_scope = profiler::ProfileScope::new("draw", &profiler);
            rasterizer.draw(&mut framebuffer);
        }
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;

    let mut window = video_subsystem
        .window("rust-sdl3 demo: Window", 1280, 720)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    let mut state = State::default();
    state.mesh = io::load_obj(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/Lamp2.obj"));
    state.mesh2 = io::load_obj(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/Teapot.obj"));
    state
        .meshes
        .insert("Teapot2".to_string(), io::load_obj(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/Teapot2.obj")));
    state
        .meshes
        .insert("Teapot3".to_string(), io::load_obj(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/Teapot3.obj")));
    state
        .textures
        .insert("Teapot3".to_string(), io::load_texture(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/Teapot3.jpg")));
    // let reference_image: RgbaImage = image::open(reference_path).unwrap().into_rgba8();

    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;

    'running: loop {
        let profiler = PROFILER.lock();
        let _root_profile_scope = profiler::ProfileScope::new("frame", &profiler);

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                Event::KeyDown { keycode: Some(Keycode::_1), keymod: Mod::LGUIMOD, .. } => {
                    state.display_mode = DisplayMode::Color;
                }
                Event::KeyDown { keycode: Some(Keycode::_2), keymod: Mod::LGUIMOD, .. } => {
                    state.display_mode = DisplayMode::Depth;
                }
                Event::KeyDown { keycode: Some(Keycode::_3), keymod: Mod::LGUIMOD, .. } => {
                    state.display_mode = DisplayMode::Normal;
                }
                Event::KeyDown { keycode: Some(Keycode::_0), keymod: Mod::LGUIMOD, .. } => {
                    state.overlay_tiles = !state.overlay_tiles;
                }
                Event::KeyDown { keycode: Some(Keycode::T), keymod: Mod::LGUIMOD, .. } => {
                    state.texture_filtering = match state.texture_filtering {
                        SamplerFilter::Nearest => SamplerFilter::Bilinear,
                        SamplerFilter::Bilinear => SamplerFilter::Nearest,
                        SamplerFilter::Trilinear => SamplerFilter::Nearest,
                    };
                }
                _ => {}
            }
        }

        {
            // Update the framebuffer size if required
            let size = window.size();
            if state.color_buffer.width() != size.0 as u16 || state.color_buffer.height() != size.1 as u16 {
                state.color_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
                state.depth_buffer = TiledBuffer::<u16, 64, 64>::new(size.0 as u16, size.1 as u16);
                state.normal_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            }

            state.dt = state.timestamp.elapsed();
            state.t += state.dt;
            state.timestamp = Instant::now();
            render(&mut state);

            state.rasterizer_stats = state.rasterizer.statistics().smoothed(5, state.rasterizer_stats);

            if (state.timestamp - state.last_printout).as_secs() > 2 {
                state.last_printout = state.timestamp;
                profiler.print();

                let title = format!(
                    "({}x{})px, {} tiles, tri_comm: {}, tri_sched: {}, tri_binn: {}, rast_frags: {}, FPS: {:.0}",
                    size.0,
                    size.1,
                    state.color_buffer.tiles_x() * state.color_buffer.tiles_y(),
                    state.rasterizer_stats.committed_triangles,
                    state.rasterizer_stats.scheduled_triangles,
                    state.rasterizer_stats.binned_triangles,
                    state.rasterizer_stats.fragments_drawn,
                    1.0 / state.dt.as_secs_f32()
                );
                window.set_title(&title).map_err(|e| e.to_string())?;
            }
        }

        {
            let _blit_profile_scope = profiler::ProfileScope::new("blit to window", &profiler);
            // TODO: this is stupid
            if state.display_mode == DisplayMode::Color {
                let mut flat = state.color_buffer.as_flat_buffer();
                if state.overlay_tiles {
                    overlay_tiles(&mut flat);
                }
                blit_to_window(&mut flat, &window, &event_pump);
            } else if state.display_mode == DisplayMode::Depth {
                blit_depth_to_window(&state.depth_buffer.as_flat_buffer(), &window, &event_pump);
            } else if state.display_mode == DisplayMode::Normal {
                let mut flat = state.normal_buffer.as_flat_buffer();
                if state.overlay_tiles {
                    overlay_tiles(&mut flat);
                }
                blit_normals_to_window(&mut flat, &window, &event_pump);
            }
        }
    }

    Ok(())
}

fn overlay_tiles(buffer: &mut Buffer<u32>) {
    for y in 0..buffer.height {
        for x in 0..buffer.width {
            if (x % 128 <= 64 && y % 128 <= 64) || (x % 128 > 64 && y % 128 > 64) {
                let mut c = RGBA::from_u32(buffer.at(x, y));
                c.r = ((c.r as u16) * 7 / 8) as u8;
                c.g = ((c.g as u16) * 7 / 8) as u8;
                c.b = ((c.b as u16) * 7 / 8) as u8;
                *buffer.at_mut(x, y) = c.to_u32();
            }
        }
    }
}

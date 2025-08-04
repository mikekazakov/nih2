extern crate sdl3;

use nih::math::*;
use nih::render::*;

use nih::render::rgba::RGBA;
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Mod};
use sdl3::pixels::PixelFormatEnum;
use sdl3::rect::Rect;
use sdl3::surface::Surface;

mod io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisplayMode {
    Color,
    Depth,
}

struct State {
    color_buffer: Buffer<u32>,
    depth_buffer: Buffer<u16>,
    mesh: MeshData,
    tick: i32,
    display_mode: DisplayMode,
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
    let width = buffer.width as u32;
    let height = buffer.height as u32;
    let mut buffer_surface = Surface::new(width, height, PixelFormatEnum::ABGR8888.into()).unwrap();
    let pitch = buffer_surface.pitch() as usize;
    buffer_surface.with_lock_mut(|pixels: &mut [u8]| {
        for y in 0..buffer.height {
            for x in 0..buffer.width {
                let depth = buffer.at(x, y);
                let gray = ((depth as u32 * 255) / 65535) as u8;
                let offset = y as usize * pitch + x as usize * 4;
                pixels[offset + 0] = gray; // B
                pixels[offset + 1] = gray; // G
                pixels[offset + 2] = gray; // R
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
    // buf.fill(RGBA::new((tick % 256) as u8, 255, 0, 255));
    let tick = state.tick;

    state.color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
    state.depth_buffer.fill(u16::MAX);

    let viewport =
        Viewport { xmin: 0, ymin: 0, xmax: state.color_buffer.width as u16, ymax: state.color_buffer.height as u16 };
    // let lines = vec![
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(0.5, 0.0, 0.0), //
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(1.0, 1.0, 0.0),
    //     Vec3::new(0.0, 0.0, 0.0),
    //     Vec3::new(-1.0, -1.0, 0.0),
    // ];

    if false {
        let mut mesh_lines = Vec::<nih::math::Vec3>::default();
        let num = state.mesh.positions.len() / 3;
        // let num = 1;
        for i in 0..num {
            mesh_lines.push(state.mesh.positions[i * 3 + 0]);
            mesh_lines.push(state.mesh.positions[i * 3 + 1]);
            mesh_lines.push(state.mesh.positions[i * 3 + 1]);
            mesh_lines.push(state.mesh.positions[i * 3 + 2]);
            mesh_lines.push(state.mesh.positions[i * 3 + 2]);
            mesh_lines.push(state.mesh.positions[i * 3 + 0]);
        }

        // let lines = nih::math::sphere_to_aa_lines(32);

        let mut cmd = DrawLinesCommand::default();
        // cmd.lines = &lines;
        // // cmd.color = Vec4::new(1.0, 1.0, 0.0, 1.0);
        // cmd.color = Vec4::new(1.0, 1.0, 0.0, 0.6);
        // cmd.model = Mat34::rotate_yz(tick as f32 / 377.0)
        //     * Mat34::rotate_xy(tick as f32 / 177.0)
        //     * Mat34::rotate_zx(tick as f32 / 100.0)
        //     * Mat34::scale_uniform(0.5);

        cmd.lines = &mesh_lines;
        // cmd.color = Vec4::new(1.0, 1.0, 0.0, 1.0);
        cmd.color = Vec4::new(1.0, 1.0, 0.0, 0.6);
        cmd.model = Mat34::rotate_zx(tick as f32 / 100.0)
            * Mat34::translate(Vec3::new(0.0, 0.0, 0.0))
            * Mat34::scale_uniform(0.5);

        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut state.color_buffer);
        draw_lines(&mut framebuffer, &viewport, &cmd);

        let aabb_lines = aabb_to_lines(state.mesh.aabb);
        cmd.lines = &aabb_lines;
        draw_lines(&mut framebuffer, &viewport, &cmd);
    }

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
        let mut rasterizer = Rasterizer::new();
        rasterizer.setup(viewport);

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

        let mut colors = Vec::<Vec4>::default();
        for i in 0..state.mesh.positions.len() {
            colors.push(idx_to_color_hash(i));
        }

        let mut cmd = RasterizationCommand::default();
        cmd.world_positions = &state.mesh.positions;
        cmd.normals = &state.mesh.normals;
        cmd.tex_coords = &state.mesh.tex_coords;
        cmd.colors =  /*&state.mesh.colors*/&colors;
        cmd.indices = &state.mesh.indices;
        // model: Mat34::translate(Vec3::new(0.0, -0.8, 0.0)),
        // model: Mat34::identity(),
        cmd.model = Mat34::rotate_yz(tick as f32 / 377.0)
            * Mat34::rotate_xy(tick as f32 / 177.0)
            * Mat34::rotate_zx(tick as f32 / 100.0)
            * Mat34::scale_uniform(1.5);
        // cmd.view = Mat44::identity();
        // cmd.view = Mat44::translate(Vec3::new(0.0, 0.0, 1.0));
        // cmd.view = Mat44::scale_uniform(100.0);
        cmd.view = Mat44::translate(Vec3::new(0.0, 0.0, -4.0));
        // cmd.model = Mat34::scale_uniform(100.0);
        cmd.projection =
            Mat44::perspective(1.0, 20.0, std::f32::consts::PI / 3.0, viewport.xmax as f32 / viewport.ymax as f32);
        cmd.culling = CullMode::CW;
        // color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        // color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        // };
        rasterizer.commit(&cmd);

        // Z: [-1, 1]
        // near -> -1
        // far  -> +1
        // pub fn perspective(near: f32, far: f32, fov_y: f32, aspect_ratio: f32) -> Mat44 {

        // g_Scene.camera_projection = Mat44::perspective(1.f, 40.f, std::numbers::pi / 3.f, f32(g_Width) / f32(g_Height) );

        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut state.color_buffer);
        framebuffer.depth_buffer = Some(&mut state.depth_buffer);
        rasterizer.draw(&mut framebuffer);
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;

    let mut window = video_subsystem
        .window("rust-sdl3 demo: Window", 600, 600)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // let mesh = load_obj("/Users/migun/Documents/gfx/Obj/lamp2.obj");
    let mesh = io::load_obj("/Users/migun/Documents/gfx/Obj2/Lamp2.obj");

    // let mut buf = ;
    // buf.fill(RGBA::new(0, 0, 0, 255));
    let mut state = State {
        color_buffer: Buffer::<u32>::new(window.size().0 as u16, window.size().1 as u16), //
        depth_buffer: Buffer::<u16>::new(window.size().0 as u16, window.size().1 as u16), //
        mesh: mesh,
        tick: 0,
        display_mode: DisplayMode::Color,
    };

    // let (models, materials) =
    //     tobj::load_obj("/Users/migun/Documents/gfx/Obj/teapot-2.obj", &tobj::LoadOptions::default())
    //         .expect("Failed to OBJ load file");

    // let aa = std::fs::read_to_string("/Users/migun/Documents/gfx/Obj/lamp2.obj")?;

    // // let input = BufReader::new(File::open("/Users/migun/Documents/gfx/Obj/teapot-2.obj").unwrap());
    // let input = BufReader::new(File::open("/Users/migun/Documents/gfx/Obj/lamp2.obj").unwrap());
    // // input.re
    // let model: Obj = load_obj(input).unwrap();
    // // let model: Obj = load_obj(aa).unwrap();

    // let model = wavefront_obj::obj::parse(aa)?;

    //model.objects[0].
    //
    //

    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                Event::KeyDown { keycode: Some(Keycode::_1), keymod: Mod::LGUIMOD, .. } => {
                    state.display_mode = DisplayMode::Color;
                }
                Event::KeyDown { keycode: Some(Keycode::_2), keymod: Mod::LGUIMOD, .. } => {
                    state.display_mode = DisplayMode::Depth;
                }
                _ => {}
            }
        }

        {
            let position = window.position();
            let size = window.size();
            let title =
                format!("Window - pos({}x{}), size({}x{}): {}", position.0, position.1, size.0, size.1, state.tick);
            window.set_title(&title).map_err(|e| e.to_string())?;

            state.tick += 1;
            render(&mut state);
        }

        if state.display_mode == DisplayMode::Color {
            blit_to_window(&mut state.color_buffer, &window, &event_pump);
        } else if state.display_mode == DisplayMode::Depth {
            blit_depth_to_window(&state.depth_buffer, &window, &event_pump);
        }
    }

    Ok(())
}

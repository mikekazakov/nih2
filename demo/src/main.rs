extern crate sdl3;

use nih::math::*;
use nih::render::*;

use nih::render::rgba::RGBA;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::PixelFormatEnum;
use sdl3::rect::Rect;
use sdl3::surface::Surface;

mod io;

struct State {
    color_buffer: Buffer<u32>,
    mesh: MeshData,
    tick: i32,
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

fn render(state: &mut State) {
    // buf.fill(RGBA::new((tick % 256) as u8, 255, 0, 255));
    let tick = state.tick;

    state.color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());

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
    cmd.model =
        Mat34::rotate_zx(tick as f32 / 100.0) * Mat34::translate(Vec3::new(0.0, 0.0, 0.0)) * Mat34::scale_uniform(0.5);

    let mut framebuffer = Framebuffer { color_buffer: Some(&mut state.color_buffer) };
    draw_lines(&mut framebuffer, &viewport, &cmd);

    let aabb_lines = aabb_to_lines(state.mesh.aabb);
    cmd.lines = &aabb_lines;
    draw_lines(&mut framebuffer, &viewport, &cmd);
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
        mesh: mesh,
        tick: 0,
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

        blit_to_window(&mut state.color_buffer, &window, &event_pump);
    }

    Ok(())
}

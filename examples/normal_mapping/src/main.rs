use nih::math::*;
use nih::render::*;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::PixelFormat;
use sdl3::surface::Surface;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init SDL and Window
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window(
            "Normal Mapping Example | Space - pause, N - show normals, M - apply normal map, Left/Right - move light, Esc to close",
            1280,
            720,
        )
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // Load the textures
    let albedo_texture = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/albedo.jpg"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..3].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGB, texels: &texels })
    };
    let normal_map = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/normals.png"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..3].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGB, texels: &texels })
    };

    // Allocate the buffers and the rasterizer
    let world_positions = [
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];
    let tex_coords = [
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
    ];
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut normal_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut t = 0.0;
    let mut apply_normal_map: bool = true;
    let mut light_dir_neg = -(Vec3::new(-1.0, -1.0, -1.0).normalized());
    let view_dir_neg: Vec3 = -Vec3::new(0.0, 0.0, -1.0);
    let mut show_normals: bool = false;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    loop {
        // Poll for SDL events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => return Ok(()),
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => paused = !paused,
                Event::KeyDown { keycode: Some(Keycode::M), .. } => apply_normal_map = !apply_normal_map,
                Event::KeyDown { keycode: Some(Keycode::N), .. } => show_normals = !show_normals,
                Event::KeyDown { keycode: Some(Keycode::Left), .. } => {
                    light_dir_neg = Mat34::rotate_zx(-0.05) * light_dir_neg
                }
                Event::KeyDown { keycode: Some(Keycode::Right), .. } => {
                    light_dir_neg = Mat34::rotate_zx(0.05) * light_dir_neg
                }
                _ => {}
            }
        }

        // Animate
        if !paused {
            t += (std::time::Instant::now() - last).as_secs_f32();
        }
        last = std::time::Instant::now();

        // Init the rasterizer
        let size = window.size();
        if color_buffer.width() != size.0 as u16 || color_buffer.height() != size.1 as u16 {
            color_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            normal_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            depth_buffer = TiledBuffer::<u16, 64, 64>::new(size.0 as u16, size.1 as u16);
        }
        color_buffer.fill(RGBA::new(64, 224, 208, 255).to_u32());
        normal_buffer.fill(RGBA::new(127, 255, 127, 255).to_u32());
        depth_buffer.fill(u16::MAX);
        rasterizer.setup(Viewport::new(0, 0, size.0 as u16, size.1 as u16));

        // Commit the draw commands
        rasterizer.commit(&RasterizationCommand {
            world_positions: &world_positions,
            tex_coords: &tex_coords,
            texture: Some(albedo_texture.clone()),
            sampling_filter: SamplerFilter::Bilinear,
            normal_map: if apply_normal_map {
                Some(normal_map.clone())
            } else {
                None
            },
            projection: Mat44::perspective(1.0, 20.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32),
            model: Mat34::translate(Vec3::new(0.0, 0.0, -9.0))
                * Mat34::rotate_yz(-0.9)
                * Mat34::rotate_xy(t * 0.5)
                * Mat34::scale_uniform(6.0),
            ..Default::default()
        });

        // Render into the framebuffer
        rasterizer.draw(&mut Framebuffer {
            color_buffer: Some(&mut color_buffer),
            normal_buffer: Some(&mut normal_buffer),
            depth_buffer: Some(&mut depth_buffer),
        });

        // Apply basic lighting
        let half: Vec3 = (view_dir_neg + light_dir_neg).normalized(); // cheat and use a uniform view direction
        for y in 0..size.1 as u16 {
            for x in 0..size.0 as u16 {
                if depth_buffer.at(x, y) < u16::MAX {
                    let normal_rgba: RGBA = RGBA::from_u32(normal_buffer.at(x, y));
                    let normal: Vec3 = (Vec3::new(normal_rgba.r as f32, normal_rgba.g as f32, normal_rgba.b as f32)
                        - Vec3::new(127.0, 127.0, 127.0))
                        / 128.0;
                    let ambient: f32 = 0.4;
                    let diffuse: f32 = 0.6 * dot(normal, light_dir_neg).max(0.0);
                    let specular: f32 = 0.2 * dot(normal, half).max(0.0).powi(10);
                    let color_rgba: RGBA = RGBA::from_u32(color_buffer.at(x, y));
                    let color_vec: Vec3 = Vec3::new(color_rgba.r as f32, color_rgba.g as f32, color_rgba.b as f32);
                    let color_lit: Vec3 = (color_vec * (diffuse + specular + ambient)).clamped(0.0, 255.0);
                    let final_color: RGBA = RGBA::new(color_lit.x as u8, color_lit.y as u8, color_lit.z as u8, 255);
                    *color_buffer.at_mut(x, y) = final_color.to_u32();
                }
            }
        }

        // Blit the framebuffer to the window
        let mut flat = if show_normals {
            let mut n = normal_buffer.as_flat_buffer();
            n.elems.iter_mut().for_each(|v| *v |= 0xFF000000u32);
            n
        } else {
            color_buffer.as_flat_buffer()
        };
        let mut windows_surface = window.surface(&event_pump)?;
        Surface::from_data(flat.as_u8_slice_mut(), size.0, size.1, size.0 * 4, PixelFormat::ABGR8888.into())
            .unwrap()
            .blit(None, &mut windows_surface, None)?;
        windows_surface.finish()?;
    }
}

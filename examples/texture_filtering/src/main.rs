use nih::math::*;
use nih::render::*;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::PixelFormatEnum;
use sdl3::surface::Surface;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init SDL and Window
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("Texture Filtering Example | Space to pause, Esc to close", 1280, 720)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // Load the texture
    let texture = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/texture.jpg"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..3].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGB, texels: &texels })
    };

    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut t = 0.0;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    loop {
        // Poll for SDL events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => return Ok(()),
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => paused = !paused,
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
        }
        color_buffer.fill(RGBA::new(64, 224, 208, 255).to_u32());
        rasterizer.setup(Viewport::new(0, 0, size.0 as u16, size.1 as u16));

        // Commit the commands
        let world_positions = vec![
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
        ];
        let tex_coords = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];
        let mut cmd = RasterizationCommand::default();
        cmd.world_positions = &world_positions;
        cmd.tex_coords = &tex_coords;
        cmd.texture = Some(texture.clone());
        cmd.projection = Mat44::perspective(1.0, 20.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32);
        cmd.model = Mat34::translate(Vec3::new(-2.02, 0.0, -8.0 + (t * 0.5).cos() * 7.0));
        cmd.sampling_filter = SamplerFilter::Nearest;
        rasterizer.commit(&cmd);
        cmd.model = Mat34::translate(Vec3::new(0.0, 0.0, -8.0 + (t * 0.5).cos() * 7.0));
        cmd.sampling_filter = SamplerFilter::Bilinear;
        rasterizer.commit(&cmd);
        cmd.model = Mat34::translate(Vec3::new(2.02, 0.0, -8.0 + (t * 0.5).cos() * 7.0));
        cmd.sampling_filter = SamplerFilter::Trilinear;
        rasterizer.commit(&cmd);

        // Render into the framebuffer
        let mut framebuffer = Framebuffer::default();
        framebuffer.color_buffer = Some(&mut color_buffer);
        rasterizer.draw(&mut framebuffer);

        // Blit the framebuffer to the window
        let mut flat = color_buffer.as_flat_buffer();
        let mut windows_surface = window.surface(&event_pump)?;
        Surface::from_data(flat.as_u8_slice_mut(), size.0, size.1, size.0 * 4, PixelFormatEnum::ABGR8888.into())
            .unwrap()
            .blit(None, &mut windows_surface, None)?;
        windows_surface.finish()?;
    }
}

use nih::math::*;
use nih::render::*;
use rand::Rng;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::PixelFormat;
use sdl3::surface::Surface;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the per-particle data
    #[derive(Clone, Copy, Debug, Default)]
    struct Particle {
        color: Vec4,
        color_dt: Vec4,
        pos: Vec3,
        pos_dt: Vec3,
        rot_scale: Vec2,
        rot_scale_dt: Vec2,
    }

    // Init SDL and Window
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("Particles Example | Space to pause, Esc to close", 1280, 720)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // Load the texture
    let texture = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/star.png"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..4].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGBA, texels: &texels })
    };

    // Initialize the particle storage
    const MAX_PARTICLES: usize = 1000;
    let mut particles: Vec<Particle> = vec![Particle::default(); MAX_PARTICLES];
    let mut particles_vertices: Vec<Vec3> = vec![Vec3::default(); MAX_PARTICLES * 6];
    let mut particles_colors: Vec<Vec4> = vec![Vec4::default(); MAX_PARTICLES * 6];
    let particles_tex_coords = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
    ]
    .repeat(MAX_PARTICLES);
    let base_positions = [
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];

    // Initialize the rest of the state
    let mut rand_gen = rand::rng();
    let mut rand = |min: f32, max: f32| -> f32 { rand_gen.random_range(min..max) };
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut dt: f32;
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
        dt = if paused {
            0.0
        } else {
            (std::time::Instant::now() - last).as_secs_f32()
        };
        last = std::time::Instant::now();

        // Init the rasterizer
        let size = window.size();
        if color_buffer.width() != size.0 as u16 || color_buffer.height() != size.1 as u16 {
            color_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
        }
        color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
        rasterizer.setup(Viewport::new(0, 0, size.0 as u16, size.1 as u16));

        // Emit / purge
        particles.iter_mut().for_each(|p| {
            if p.color.w <= 0.0 || p.rot_scale.y <= 0.0 {
                p.color = Vec4::new(rand(0.7, 1.0), rand(0.7, 1.0), rand(0.7, 1.0), rand(0.8, 1.0));
                p.color_dt = Vec4::new(0.0, 0.0, 0.0, rand(-1.0, -0.3));
                p.pos = Vec3::new(rand(-1.0, 1.0), -4.0, -8.0);
                p.pos_dt = Vec3::new(rand(-1.0, 1.0), rand(0.5, 5.0), rand(-1.0, 1.0));
                p.rot_scale = Vec2::new(rand(0.0, 6.0), rand(0.5, 1.0));
                p.rot_scale_dt = Vec2::new(rand(-6.0, 6.0), rand(-0.5, 0.0));
            }
        });

        // Update
        particles.iter_mut().for_each(|p| {
            p.pos += p.pos_dt * dt;
            p.color += p.color_dt * dt;
            p.rot_scale += p.rot_scale_dt * dt;
        });

        // Sort by Z
        particles.sort_by(|a, b| a.pos.z.partial_cmp(&b.pos.z).unwrap());

        // Update per-vertex data
        particles.iter().enumerate().for_each(|(i, p)| {
            let m = Mat34::translate(p.pos) * Mat34::rotate_xy(p.rot_scale.x) * Mat34::scale_uniform(p.rot_scale.y);
            for j in 0..6 {
                particles_vertices[i * 6 + j] = m * base_positions[j];
                particles_colors[i * 6 + j] = p.color;
            }
        });

        // Commit the draw command
        rasterizer.commit(&RasterizationCommand {
            world_positions: &particles_vertices,
            tex_coords: &particles_tex_coords,
            colors: &particles_colors,
            texture: Some(texture.clone()),
            sampling_filter: SamplerFilter::Bilinear,
            alpha_blending: AlphaBlendingMode::Additive,
            alpha_test: 2u8,
            projection: Mat44::perspective(1.0, 20.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32),
            ..Default::default()
        });

        // Render into the framebuffer
        rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Default::default() });

        // Blit the framebuffer to the window
        let mut flat = color_buffer.as_flat_buffer();
        let mut windows_surface = window.surface(&event_pump)?;
        Surface::from_data(flat.as_u8_slice_mut(), size.0, size.1, size.0 * 4, PixelFormat::ABGR8888.into())
            .unwrap()
            .blit(None, &mut windows_surface, None)?;
        windows_surface.finish()?;
    }
}

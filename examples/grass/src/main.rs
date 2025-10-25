use nih::math::*;
use nih::render::*;
use noise::{NoiseFn, Perlin, Seedable};
use rand::{Rng, SeedableRng, rngs::SmallRng};
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
            "Grass Example | Space - pause, N - show normals, W - draw wireframe, L - apply lighting, Esc - close",
            1280,
            720,
        )
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    // Load the textures
    let grass_texture = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/grass.png"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..4].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGBA, texels: &texels })
    };
    let ground_texture = {
        let image = image::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("res/ground.jpg"))
            .unwrap()
            .into_rgba8();
        let width = image.width();
        let height = image.height();
        let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..3].iter().copied()).collect();
        Texture::new(&TextureSource { width, height, format: TextureFormat::RGB, texels: &texels })
    };

    let quad_positions = [
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];
    let quad_tex_coords = [
        Vec2::new(0.0, 0.1),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.1),
        Vec2::new(1.0, 0.1),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
    ];

    // Random entropy sources
    let mut rand_gen = SmallRng::seed_from_u64(7);
    let mut rand = |min: f32, max: f32| -> f32 { rand_gen.random_range(min..max) };
    let perlin = Perlin::default().set_seed(2);

    // Generate 17x17=289 bushes
    struct Bush {
        model_mat1: Mat34,
        model_mat2: Mat34,
    }
    let mut bush_positions: Vec<Vec3> = Vec::<Vec3>::new(); // 12 per bush
    let mut bush_normals: Vec<Vec3> = Vec::<Vec3>::new(); // 12 per bush
    let mut bush_tex_coords: Vec<Vec2> = Vec::<Vec2>::new(); // 12 per bush
    let mut bushes = Vec::<Bush>::new();
    for i in -8..9 {
        for j in -8..9 {
            let position: Vec3 = Vec3::new(i as f32 * 5.0 + rand(-2.0, 2.0), 0.0, j as f32 * 5.0 + rand(-2.0, 2.0));
            let rotation: f32 = rand(0.0, 3.14);
            let scale: f32 = rand(3.0, 6.0);
            let model_mat1: Mat34 = Mat34::translate(position)
                * Mat34::rotate_zx(rotation)
                * Mat34::scale_uniform(scale)
                * Mat34::translate(Vec3::new(0.0, 1.0, 0.0));
            let model_mat2: Mat34 = Mat34::translate(position)
                * Mat34::rotate_zx(rotation + 1.57)
                * Mat34::scale_uniform(scale)
                * Mat34::translate(Vec3::new(0.0, 1.0, 0.0));
            bushes.push(Bush { model_mat1, model_mat2 });
            bush_positions.extend(quad_positions.iter().map(|&p| model_mat1 * p));
            bush_positions.extend(quad_positions.iter().map(|&p| model_mat2 * p));
            bush_normals.extend(std::iter::repeat(Vec3::new(0.0, 1.0, 0.0)).take(12));
            bush_tex_coords.extend_from_slice(&quad_tex_coords);
            bush_tex_coords.extend_from_slice(&quad_tex_coords);
        }
    }

    // Allocate the buffers and the rasterizer
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut normal_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut t = 0.0;
    let mut apply_lighting: bool = true;
    let light_dir_neg = -(Vec3::new(-1.0, -1.0, -1.0).normalized());
    let mut show_normals: bool = false;
    let mut show_wireframe: bool = false;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    loop {
        // Poll for SDL events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => return Ok(()),
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => paused = !paused,
                Event::KeyDown { keycode: Some(Keycode::N), .. } => show_normals = !show_normals,
                Event::KeyDown { keycode: Some(Keycode::W), .. } => show_wireframe = !show_wireframe,
                Event::KeyDown { keycode: Some(Keycode::L), .. } => apply_lighting = !apply_lighting,
                _ => {}
            }
        }

        // Update time
        if !paused {
            t += (std::time::Instant::now() - last).as_secs_f32();
        }
        last = std::time::Instant::now();

        // Set up the wind
        let wind_speed: f32 = 100.0; // how fast the wind moves, world units per second
        let wind_strength: f32 = 1.5; // how much the wind offsets the vertices, world units
        let wind_direction: Vec2 = Vec2::new(-1.0, 0.7).normalized(); // where the wind blows
        let wind_offset: Vec2 = wind_speed * t * wind_direction; // perlin sampling offset for the wind

        // Animate the bushes
        for bush_idx in 0..bushes.len() {
            let bush = &mut bushes[bush_idx];
            let vert_idx = bush_idx * 12;

            // Calculate the wind displacement for 4 top vertices
            let wind_displacement = |pos_idx: usize| -> Vec3 {
                let noise: f32 = perlin.get([
                    ((bush_positions[pos_idx].x - wind_offset.x) * 0.01) as f64,
                    ((bush_positions[pos_idx].z - wind_offset.y) * 0.01) as f64,
                ]) as f32;
                let offset2: Vec2 = wind_strength * wind_direction * noise;
                Vec3::new(offset2.x, 0.0, offset2.y)
            };
            let offset0: Vec3 = wind_displacement(vert_idx + 1); // First top vertex of the first billboard
            let offset2: Vec3 = wind_displacement(vert_idx + 5); // Second top vertex of the first billboard
            let offset6: Vec3 = wind_displacement(vert_idx + 7); // First top vertex of the second billboard
            let offset8: Vec3 = wind_displacement(vert_idx + 11); // Second top vertex of the second billboard

            // Update the world positions of the 4 top vertices
            bush_positions[vert_idx + 0] = bush.model_mat1 * quad_positions[0] + offset0;
            bush_positions[vert_idx + 2] = bush.model_mat1 * quad_positions[2] + offset2;
            bush_positions[vert_idx + 3] = bush_positions[vert_idx + 2];
            bush_positions[vert_idx + 6] = bush.model_mat2 * quad_positions[0] + offset6;
            bush_positions[vert_idx + 8] = bush.model_mat2 * quad_positions[2] + offset8;
            bush_positions[vert_idx + 9] = bush_positions[vert_idx + 8];

            // Tilt the normals of the 4 top vertices
            let norm_scale: f32 = 0.5;
            bush_normals[vert_idx + 0] = Vec3::new(offset0.x * norm_scale, 1.0, offset0.y * norm_scale).normalized();
            bush_normals[vert_idx + 2] = Vec3::new(offset2.x * norm_scale, 1.0, offset2.y * norm_scale).normalized();
            bush_normals[vert_idx + 3] = bush_normals[vert_idx + 2];
            bush_normals[vert_idx + 6] = Vec3::new(offset6.x * norm_scale, 1.0, offset6.y * norm_scale).normalized();
            bush_normals[vert_idx + 8] = Vec3::new(offset8.x * norm_scale, 1.0, offset8.y * norm_scale).normalized();
            bush_normals[vert_idx + 9] = bush_normals[vert_idx + 8];
        }

        // Init the rasterizer
        let size = window.size();
        if color_buffer.width() != size.0 as u16 || color_buffer.height() != size.1 as u16 {
            color_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            normal_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            depth_buffer = TiledBuffer::<u16, 64, 64>::new(size.0 as u16, size.1 as u16);
            rasterizer.setup(Viewport::new(0, 0, size.0 as u16, size.1 as u16));
        }
        color_buffer.fill(RGBA::new(102, 204, 255, 255).to_u32());
        normal_buffer.fill(RGBA::new(127, 255, 127, 255).to_u32());
        depth_buffer.fill(u16::MAX);
        rasterizer.reset();
        rasterizer.set_draw_wireframe(show_wireframe);

        // Commit the draw commands
        let projection = Mat44::perspective(1.0, 100.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32);
        let view: Mat44 = Mat44::translate(Vec3::new(0.0, -2.0, -35.0)) * Mat44::rotate_yz(0.5);

        // Draw the ground plane
        rasterizer.commit(&RasterizationCommand {
            world_positions: &quad_positions,
            tex_coords: &quad_tex_coords,
            texture: Some(ground_texture.clone()),
            sampling_filter: SamplerFilter::Bilinear,
            projection,
            view,
            model: Mat34::translate(Vec3::new(0.0, 0.0, 0.0)) * Mat34::rotate_yz(-1.57) * Mat34::scale_uniform(50.0),
            ..Default::default()
        });

        // Draw the bushes
        rasterizer.commit(&RasterizationCommand {
            world_positions: &bush_positions,
            tex_coords: &bush_tex_coords,
            normals: &bush_normals,
            texture: Some(grass_texture.clone()),
            alpha_test: 127u8,
            alpha_blending: AlphaBlendingMode::Normal,
            sampling_filter: SamplerFilter::Bilinear,
            projection,
            view,
            ..Default::default()
        });

        // Render into the framebuffer
        let mut framebuffer = Framebuffer {
            color_buffer: Some(&mut color_buffer),
            normal_buffer: Some(&mut normal_buffer),
            depth_buffer: Some(&mut depth_buffer),
        };
        rasterizer.draw(&mut framebuffer);

        // Apply basic lighting
        if apply_lighting {
            framebuffer.for_each_tile_mut_parallel(move |tile| {
                let depth_tile = tile.depth_buffer.as_mut().unwrap();
                let color_tile = tile.color_buffer.as_mut().unwrap();
                let normal_tile = tile.normal_buffer.as_mut().unwrap();
                for y in 0..depth_tile.height as usize {
                    for x in 0..depth_tile.width as usize {
                        if depth_tile.at_unchecked(x, y) == u16::MAX {
                            continue;
                        }
                        let normal: Vec3 = decode_normal_from_color(RGBA::from_u32(normal_tile.at_unchecked(x, y)));
                        let ambient: f32 = 0.6;
                        let diffuse: f32 = 0.6 * dot(normal, light_dir_neg).max(0.0);
                        let color_rgba: RGBA = RGBA::from_u32(color_tile.at_unchecked(x, y));
                        let color_vec: Vec3 = Vec3::new(color_rgba.r as f32, color_rgba.g as f32, color_rgba.b as f32);
                        let color_lit: Vec3 = (color_vec * (diffuse + ambient)).min(255.0);
                        let final_color: RGBA = RGBA::new(color_lit.x as u8, color_lit.y as u8, color_lit.z as u8, 255);
                        *color_tile.get_unchecked(x, y) = final_color.to_u32();
                    }
                }
            });
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

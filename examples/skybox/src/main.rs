mod hosek_wilkie_sky;
mod reinhard_tone_mapper;

use crate::hosek_wilkie_sky::HosekWilkieSky;
use crate::reinhard_tone_mapper::ReinhardToneMapper;
use nih::math::simd::F32x4;
use nih::math::*;
use nih::render::*;
use noise::{NoiseFn, Seedable};
use rand::{Rng, SeedableRng};
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::pixels::PixelFormat;
use sdl3::surface::Surface;
use std::sync::Arc;

#[derive(PartialEq)]
enum Face {
    XNeg,
    XPos,
    YNeg,
    YPos,
    ZNeg,
    ZPos,
}

fn camera_to_mat34(orientation: Quat, position: Vec3) -> Mat34 {
    let r: Mat33 = orientation.as_mat33();
    let r_inv: Mat33 = r.transpose();
    let t_inv: Vec3 = -(r_inv * position);
    Mat34([
        r_inv.0[0], r_inv.0[1], r_inv.0[2], t_inv.x, //
        r_inv.0[3], r_inv.0[4], r_inv.0[5], t_inv.y, //
        r_inv.0[6], r_inv.0[7], r_inv.0[8], t_inv.z, //
    ])
}

fn build_face(sky: &HosekWilkieSky, face: Face, sun_dir: Vec3) -> Arc<Texture> {
    let width = 512;
    let height = 512;
    let tone_mapper = ReinhardToneMapper::new(0.5, 14.0);

    let mut texels: Vec<u8> = Vec::<u8>::new();
    texels.resize(width * height * 3, 127);
    let height_max = if face == Face::YPos { height } else { height / 2 };
    
    let sun_zenith_color: Vec3 = Vec3::new(58.0, 55.0, 29.0);
    let sun_horizon_color: Vec3 = Vec3::new(60.0, 57.0, 27.0);
    let sun_base_size: f32 = 0.055;
    let sun_size: f32 = sun_base_size + (1.0 - sun_dir.y * sun_dir.y).sqrt() * sun_base_size * 0.25;
    let sun_size_inv: f32 = 1.0 / sun_size;
    let sun_color: Vec3 = lerp(sun_horizon_color, sun_zenith_color, sun_dir.y.abs());

    let mut theta_cos_row: Vec<f32> = vec![0.0; width];
    let mut gamma_cos_row: Vec<f32> = vec![0.0; width];
    let mut gamma_row: Vec<f32> = vec![0.0; width];
    let mut r_row: Vec<f32> = vec![0.0; width];
    let mut g_row: Vec<f32> = vec![0.0; width];
    let mut b_row: Vec<f32> = vec![0.0; width];

    // Set up the initial direction vector for y=0/x=0, depending on the face.
    // TODO: not actually precisely -1.0/+1.0?..
    let mut dir_row: Vec3 = match face {
        Face::XNeg => Vec3::new(-1.0, 1.0, 1.0),
        Face::XPos => Vec3::new(1.0, 1.0, -1.0),
        Face::YNeg => Vec3::new(1.0, -1.0, -1.0),
        Face::YPos => Vec3::new(-1.0, 1.0, 1.0),
        Face::ZNeg => Vec3::new(-1.0, 1.0, -1.0),
        Face::ZPos => Vec3::new(1.0, 1.0, 1.0),
    };
    // Set up the direction increment for each row.
    let dir_dy: Vec3 = match face {
        Face::XNeg => Vec3::new(0.0, -2.0 / (height as f32), 0.0),
        Face::XPos => Vec3::new(0.0, -2.0 / (height as f32), 0.0),
        Face::YNeg => Vec3::new(0.0, 0.0, 2.0 / (height as f32)),
        Face::YPos => Vec3::new(0.0, 0.0, -2.0 / (height as f32)),
        Face::ZNeg => Vec3::new(0.0, -2.0 / (height as f32), 0.0),
        Face::ZPos => Vec3::new(0.0, -2.0 / (height as f32), 0.0),
    };
    // Set up the direction increment for each column.
    let dir_dx: Vec3 = match face {
        Face::XNeg => Vec3::new(0.0, 0.0, -2.0 / (width as f32)),
        Face::XPos => Vec3::new(0.0, 0.0, 2.0 / (width as f32)),
        Face::YNeg => Vec3::new(-2.0 / (width as f32), 0.0, 0.0),
        Face::YPos => Vec3::new(2.0 / (width as f32), 0.0, 0.0),
        Face::ZNeg => Vec3::new(2.0 / (width as f32), 0.0, 0.0),
        Face::ZPos => Vec3::new(-2.0 / (width as f32), 0.0, 0.0),
    };
    let dir_dx_x_4: F32x4 = F32x4::splat(dir_dx.x) * F32x4::splat(4.0);
    let dir_dx_y_4: F32x4 = F32x4::splat(dir_dx.y) * F32x4::splat(4.0);
    let dir_dx_z_4: F32x4 = F32x4::splat(dir_dx.z) * F32x4::splat(4.0);
    let dir_offset_x_4: F32x4 = F32x4::load([dir_dx.x, dir_dx.x * 2.0, dir_dx.x * 3.0, dir_dx.x * 4.0]);
    let dir_offset_y_4: F32x4 = F32x4::load([dir_dx.y, dir_dx.y * 2.0, dir_dx.y * 3.0, dir_dx.y * 4.0]);
    let dir_offset_z_4: F32x4 = F32x4::load([dir_dx.z, dir_dx.z * 2.0, dir_dx.z * 3.0, dir_dx.z * 4.0]);
    let sun_dir_x_4: F32x4 = F32x4::splat(sun_dir.x);
    let sun_dir_y_4: F32x4 = F32x4::splat(sun_dir.y);
    let sun_dir_z_4: F32x4 = F32x4::splat(sun_dir.z);
    for y in 0..height_max {
        // Calculate gamma, theta_cos, gamma_cos for each texel in the row.
        let mut vec_x_4: F32x4 = F32x4::splat(dir_row.x) + dir_offset_x_4;
        let mut vec_y_4: F32x4 = F32x4::splat(dir_row.y) + dir_offset_y_4;
        let mut vec_z_4: F32x4 = F32x4::splat(dir_row.z) + dir_offset_z_4;
        for x in (0..width).step_by(4) {
            // normalize the components of the direction vector
            let recip_len_sqrt: F32x4 = (vec_x_4 * vec_x_4 + vec_y_4 * vec_y_4 + vec_z_4 * vec_z_4).rsqrt();
            let normalized_vec_x_4: F32x4 = vec_x_4 * recip_len_sqrt;
            let normalized_vec_y_4: F32x4 = vec_y_4 * recip_len_sqrt;
            let normalized_vec_z_4: F32x4 = vec_z_4 * recip_len_sqrt;
            // cos(theta) - cos(angle between the zenith and the view direction)
            let theta_cos_4: F32x4 = normalized_vec_y_4;
            // gamma_cos = dot(dir, sun_dir).clamp(-1.0, 1.0);
            let gamma_cos_4: F32x4 = (normalized_vec_x_4 * sun_dir_x_4 +
                normalized_vec_y_4 * sun_dir_y_4 +
                normalized_vec_z_4 * sun_dir_z_4).min(F32x4::splat(1.0)).max(F32x4::splat(-1.0));
            // gamma - angle between the view direction and the Sun
            let gamma_4: F32x4 = gamma_cos_4.acos();
            theta_cos_4.store_to(unsafe { &mut *(theta_cos_row.as_mut_ptr().add(x) as *mut [f32; 4]) });
            gamma_cos_4.store_to(unsafe { &mut *(gamma_cos_row.as_mut_ptr().add(x) as *mut [f32; 4]) });
            gamma_4.store_to(unsafe { &mut *(gamma_row.as_mut_ptr().add(x) as *mut [f32; 4]) });
            // step the direction vector forward by 4 texels
            // wasteful - de-facto only 1 of the 3 dx regs is non-zero
            vec_x_4 += dir_dx_x_4;
            vec_y_4 += dir_dx_y_4;
            vec_z_4 += dir_dx_z_4;
        }

        // Calculate per-channel radiance values for each texel in the row.
        sky.f_simd_r(&gamma_row, &theta_cos_row, &gamma_cos_row, &mut r_row);
        sky.f_simd_g(&gamma_row, &theta_cos_row, &gamma_cos_row, &mut g_row);
        sky.f_simd_b(&gamma_row, &theta_cos_row, &gamma_cos_row, &mut b_row);

        // Inject 'the Sun' into the sky.
        for x in 0..width {
            let gamma: f32 = gamma_row[x];
            let sun_amount: f32 = (1.0 - gamma * sun_size_inv).clamp(0.0, 1.0);
            if sun_amount > 0.0 {
                let sun_color: Vec3 = sun_color * (sun_amount * sun_amount);
                r_row[x] += sun_color.x;
                g_row[x] += sun_color.y;
                b_row[x] += sun_color.z;
            }
        }

        // Map the radiance values to RGB colors and store them in the texture.
        tone_mapper.map(&r_row, &g_row, &b_row, texels[y * width * 3..y * width * 3 + width * 3].as_mut());

        // Step the direction vector forward by 1 row
        dir_row += dir_dy;
    }

    Texture::new(&TextureSource {
        width: width as u32,
        height: height as u32,
        format: TextureFormat::RGB,
        texels: &texels,
    })
}

fn test_hosek_wilkie_sky() {
    // The reference outputs were copied from the results of running the code from the original paper.
    let sky1: HosekWilkieSky = HosekWilkieSky::new(2.0, Vec3::new(0.0, 0.0, 0.0), std::f32::consts::FRAC_PI_4);
    assert!((sky1.f(0.0, std::f32::consts::FRAC_PI_4.cos(), 0.0f32.cos()) - Vec3::new(8.663214, 11.592292, 16.004868)).length() < 0.01);
    assert!((sky1.f(0.1, std::f32::consts::FRAC_PI_4.cos(), 0.1f32.cos()) - Vec3::new(7.697937, 10.479785, 15.563609)).length() < 0.01);
    assert!((sky1.f(0.1, 0.6f32.cos(), 0.1f32.cos()) - Vec3::new(6.292841, 8.564651, 13.267812)).length() < 0.01);
    let sky2: HosekWilkieSky = HosekWilkieSky::new(3.0, Vec3::new(0.6, 0.2, 0.9), 1.0);
    assert!((sky2.f(0.1, 0.6f32.cos(), 0.1f32.cos()) - Vec3::new(15.872860, 17.629661, 26.922695)).length() < 0.01);
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sanity check first of all
    test_hosek_wilkie_sky();

    // Init SDL and Window
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window(
            "Skybox Example | Space - pause, W - draw wireframe, R/F - turbidity, T/G - albedo.r, Y/H - albedo.g, U/J - albedo.b, Esc - close",
            1280,
            720,
        )
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    let dummy_gray_texture = Texture::new(&TextureSource {
        texels: &vec![100u8; 64 * 64],
        width: 64,
        height: 64,
        format: TextureFormat::Grayscale,
    });
    let mut neg_x_tex = dummy_gray_texture.clone();
    let neg_y_tex = dummy_gray_texture.clone();
    let mut neg_z_tex = dummy_gray_texture.clone();
    let mut pos_x_tex = dummy_gray_texture.clone();
    let mut pos_y_tex = dummy_gray_texture.clone();
    let mut pos_z_tex = dummy_gray_texture.clone();

    let neg_z_positions = [
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    let pos_z_positions = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    let pos_x_positions = [
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ];
    let neg_x_positions = [
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    ];
    let neg_y_positions = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
    ];
    let pos_y_positions = [
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
    ];
    let cubemap_face_tex_coords = [
        Vec2::new(0.001, 0.001),
        Vec2::new(0.001, 0.999),
        Vec2::new(0.999, 0.001),
        Vec2::new(0.999, 0.001),
        Vec2::new(0.001, 0.999),
        Vec2::new(0.999, 0.999),
    ];

    // Allocate the buffers and the rasterizer
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut t = 0.0;
    let mut dt: f32 = 0.0;
    let mut sky_turbidity: f32 = 3.0;
    // let mut sky_turbidity: f32 = 1.0;
    let mut ground_albedo: Vec3 = Vec3::new(0.0, 0.0, 0.5);
    let mut rebuild_skybox: bool = true;
    let mut camera_orientation: Quat = Quat::from_axis_angle(Vec3::new(0.0, 0.0, -1.0), 0.0);
    let camera_position: Vec3 = Vec3::new(0.0, 2.0, 35.0);
    let mut show_wireframe: bool = false;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    let mut faces_build_time: f32 = 0.0;
    let mut faces_build_time_n: u32 = 0;
    loop {
        // Poll for SDL events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => return Ok(()),
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => paused = !paused,
                Event::KeyDown { keycode: Some(Keycode::W), .. } => show_wireframe = !show_wireframe,
                Event::KeyDown { keycode: Some(Keycode::R), .. } => {
                    sky_turbidity = (sky_turbidity + 0.5).min(10.0);
                    println!("turbidity: {}", sky_turbidity);
                    rebuild_skybox = true;
                }
                Event::KeyDown { keycode: Some(Keycode::F), .. } => {
                    sky_turbidity = (sky_turbidity - 0.5).max(1.0);
                    println!("turbidity: {}", sky_turbidity);
                    rebuild_skybox = true;
                }
                Event::KeyDown { keycode: Some(Keycode::T), .. } | Event::KeyDown { keycode: Some(Keycode::G), .. } | Event::KeyDown { keycode: Some(Keycode::Y), .. }
                | Event::KeyDown { keycode: Some(Keycode::H), .. } | Event::KeyDown { keycode: Some(Keycode::U), .. } | Event::KeyDown { keycode: Some(Keycode::J), .. }
                => {
                    if let Event::KeyDown { keycode: Some(Keycode::T), .. } = event {
                        ground_albedo.x += 0.1;
                    }
                    if let Event::KeyDown { keycode: Some(Keycode::G), .. } = event {
                        ground_albedo.x -= 0.1;
                    }
                    if let Event::KeyDown { keycode: Some(Keycode::Y), .. } = event {
                        ground_albedo.y += 0.1;
                    }
                    if let Event::KeyDown { keycode: Some(Keycode::H), .. } = event {
                        ground_albedo.y -= 0.1;
                    }
                    if let Event::KeyDown { keycode: Some(Keycode::U), .. } = event {
                        ground_albedo.z += 0.1;
                    }
                    if let Event::KeyDown { keycode: Some(Keycode::J), .. } = event {
                        ground_albedo.z -= 0.1;
                    }
                    ground_albedo = ground_albedo.clamped(0.0, 1.0);
                    println!("ground albedo: {:.1}, {:.1}, {:.1}", ground_albedo.x, ground_albedo.y, ground_albedo.z);
                    rebuild_skybox = true;
                }
                Event::MouseMotion { xrel, yrel, mousestate, .. } => {
                    if mousestate.left() {
                        let sensitivity: f32 = 0.002;
                        let angle_yaw: f32 = -xrel * sensitivity;
                        let angle_pitch: f32 = -yrel * sensitivity;
                        let yaw: Quat = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), angle_yaw);
                        let pitch: Quat = Quat::from_axis_angle(camera_orientation * Vec3::new(1.0, 0.0, 0.0), angle_pitch);
                        camera_orientation = (yaw * pitch * camera_orientation).normalized();
                    }
                }
                _ => {}
            }
        }

        // Update time
        if !paused {
            t += (std::time::Instant::now() - last).as_secs_f32();
            rebuild_skybox = true;
        }
        dt = (std::time::Instant::now() - last).as_secs_f32();
        last = std::time::Instant::now();
        // println!("FPS: {:.0}", 1.0 / dt);

        if rebuild_skybox {
            let sun_dir: Vec3 = Vec3::new(0.0, (t * 0.1).sin(), -(t * 0.1).cos()).normalized();
            let theta_sun: f32 = sun_dir.y.acos(); // angle from zenith, radians
            let sun_elevation: f32 = (3.14 / 2.0 - theta_sun).max(0.0); // angle from the horizon, radians
            let sky: HosekWilkieSky = HosekWilkieSky::new(sky_turbidity, ground_albedo, sun_elevation);
            let start = std::time::Instant::now();
            neg_x_tex = build_face(&sky, Face::XNeg, sun_dir);
            pos_x_tex = build_face(&sky, Face::XPos, sun_dir);
            pos_y_tex = build_face(&sky, Face::YPos, sun_dir);
            pos_z_tex = build_face(&sky, Face::ZPos, sun_dir);
            neg_z_tex = build_face(&sky, Face::ZNeg, sun_dir);
            let duration = std::time::Instant::now() - start;
            faces_build_time += duration.as_secs_f32();
            faces_build_time_n += 1;
            if faces_build_time_n == 1000 {
                println!("build_face: {:.1}ms", faces_build_time);
                faces_build_time = 0.0;
                faces_build_time_n = 0;
            }
            rebuild_skybox = false;
        }

        // Init the rasterizer
        let size = window.size();
        if color_buffer.width() != size.0 as u16 || color_buffer.height() != size.1 as u16 {
            color_buffer = TiledBuffer::<u32, 64, 64>::new(size.0 as u16, size.1 as u16);
            rasterizer.setup(Viewport::new(0, 0, size.0 as u16, size.1 as u16));
        }
        color_buffer.fill(RGBA::new(102, 204, 255, 255).to_u32());
        rasterizer.reset();
        rasterizer.set_draw_wireframe(show_wireframe);

        // Commit the draw commands
        let projection: Mat44 =
            Mat44::perspective(1.0, 100.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32);
        let view: Mat44 = camera_to_mat34(camera_orientation, camera_position).as_mat44();
        let view_orientation: Mat44 = view.as_mat33().as_mat44();

        // draw the skybox
        let mut commit_face = |pos: &[Vec3; 6], texture: &Arc<Texture>| {
            rasterizer.commit(&RasterizationCommand {
                world_positions: pos,
                tex_coords: &cubemap_face_tex_coords,
                texture: Some(texture.clone()),
                sampling_filter: SamplerFilter::Bilinear,
                projection,
                view: view_orientation,
                model: Mat34::scale_uniform(2.0),
                ..Default::default()
            });
        };
        commit_face(&neg_x_positions, &neg_x_tex);
        commit_face(&pos_x_positions, &pos_x_tex);
        commit_face(&neg_y_positions, &neg_y_tex);
        commit_face(&pos_y_positions, &pos_y_tex);
        commit_face(&neg_z_positions, &neg_z_tex);
        commit_face(&pos_z_positions, &pos_z_tex);
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

mod hosek_wilkie_sky;

use crate::hosek_wilkie_sky::HosekWilkieSky;
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
    fn to_srgb(c: Vec3) -> Vec3 {
        let encode = |x: f32| {
            if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            }
        };
        Vec3 { x: encode(c.x), y: encode(c.y), z: encode(c.z) }
    }

    fn tonemap_reinhard(rgb: Vec3, exposure: f32, white: Option<f32>) -> Vec3 {
        let r = rgb.x * exposure;
        let g = rgb.y * exposure;
        let b = rgb.z * exposure;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let yd = match white {
            None => y / (1.0 + y),
            Some(w) => {
                (y * (1.0 + y / (w * w))) / (1.0 + y) // white point version
            }
        };
        let s = if y > 0.0 { yd / y } else { 1.0 };
        Vec3::new(r * s, g * s, b * s)
    }

    fn linear_to_rgb(c: Vec3) -> Vec3 {
        let exposure: f32 = 1.0;
        let exposed: Vec3 = tonemap_reinhard(c, exposure, Some(10.0));
        let display: Vec3 = to_srgb(exposed);
        display
    }

    let width = 512;
    let height = 512;

    let mut texels: Vec<u8> = Vec::<u8>::new();
    texels.resize(width * height * 3, 127);
    let height_max = if face == Face::YPos { height } else { height / 2 };
    for y in 0..height_max {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let u: f32 = 2.0 * (x as f32 + 0.5) / (width as f32) - 1.0; // [-1, 1
            let v: f32 = 2.0 * ((height - 1 - y) as f32 + 0.5) / (height as f32) - 1.0; // [-1, 1]
            let dir: Vec3 = match face {
                Face::XNeg => Vec3::new(-1.0, v, -u).normalized(),
                Face::XPos => Vec3::new(1.0, v, u).normalized(),
                Face::YNeg => Vec3::new(-u, -1.0, v).normalized(),
                Face::YPos => Vec3::new(u, 1.0, v).normalized(),
                Face::ZNeg => Vec3::new(u, v, -1.0).normalized(),
                Face::ZPos => Vec3::new(-u, v, 1.0).normalized(),
            };

            let theta: f32 = dir.y.acos(); // view angle from zenith
            let cos_gamma = dot(dir, sun_dir);
            let gamma: f32 = cos_gamma.acos(); // angle between view direction and sun
            let mut f = sky.f(theta, gamma);

            let sun_angular_radius = 0.01; // ~0.5 degrees
            if gamma < sun_angular_radius * 2.0 {
                let sun_intensity = 3.0; // multiplier at Sun center
                let falloff = (-(gamma * gamma) / (2.0 * sun_angular_radius * sun_angular_radius)).exp();
                f = f + f * sun_intensity * falloff;
            }

            let c = linear_to_rgb(f);

            texels[idx * 3 + 0] = (c.x * 255.0).clamp(0.0, 255.0) as u8;
            texels[idx * 3 + 1] = (c.y * 255.0).clamp(0.0, 255.0) as u8;
            texels[idx * 3 + 2] = (c.z * 255.0).clamp(0.0, 255.0) as u8;
        }
    }

    Texture::new(&TextureSource {
        width: width as u32,
        height: height as u32,
        format: TextureFormat::RGB,
        texels: &texels,
    })
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init SDL and Window
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window(
            "Grass Example | Space - pause, W - draw wireframe, Esc - close",
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
    let mut rebuild_skybox: bool = true;
    let mut camera_orientation: Quat = Quat::from_axis_angle(Vec3::new(0.0, 0.0, -1.0), 0.0);
    let camera_position: Vec3 = Vec3::new(0.0, 2.0, 35.0);
    let mut show_wireframe: bool = false;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    loop {
        // Poll for SDL events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => return Ok(()),
                Event::KeyDown { keycode: Some(Keycode::Space), .. } => paused = !paused,
                Event::KeyDown { keycode: Some(Keycode::W), .. } => show_wireframe = !show_wireframe,
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
        println!("FPS: {:.0}", 1.0 / dt);

        if rebuild_skybox {
            let sun_dir: Vec3 = Vec3::new(0.0, (t * 0.1).sin(), -(t * 0.1).cos()).normalized();
            let turbidity: f32 = 2.5; // daylight sky
            let theta_sun: f32 = sun_dir.y.acos(); // angle from zenith, radians
            let sun_elevation: f32 = (3.14 / 2.0 - theta_sun).max(0.0); // angle from the horizon, radians
            let ground_albedo: Vec3 = Vec3::new(0.0, 0.0, 1.0);
            let sky: HosekWilkieSky = HosekWilkieSky::new(turbidity, ground_albedo, sun_elevation);
            neg_x_tex = build_face(&sky, Face::XNeg, sun_dir);
            pos_x_tex = build_face(&sky, Face::XPos, sun_dir);
            pos_y_tex = build_face(&sky, Face::YPos, sun_dir);
            pos_z_tex = build_face(&sky, Face::ZPos, sun_dir);
            neg_z_tex = build_face(&sky, Face::ZNeg, sun_dir);
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

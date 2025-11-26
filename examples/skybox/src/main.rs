mod flares;
mod hosek_wilkie_sky;
mod reinhard_tone_mapper;

use crate::flares::{Flare, Flares};
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
use std::path::Path;
use std::sync::Arc;

#[derive(PartialEq, Clone, Copy, Debug, Hash, Eq, PartialOrd, Ord)]
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

fn project_dir_to_face_uv(face: Face, dir: Vec3) -> Option<(f32, f32)> {
    let ax: f32 = dir.x.abs();
    let ay: f32 = dir.y.abs();
    let az: f32 = dir.z.abs();
    match face {
        Face::XPos if ax >= ay && ax >= az => {
            if dir.x <= 0.0 {
                return None;
            }
            let sc: f32 = 1.0 / ax;
            Some((dir.z * sc, dir.y * sc))
        }
        Face::XNeg if ax >= ay && ax >= az => {
            if dir.x >= 0.0 {
                return None;
            }
            let sc: f32 = -1.0 / ax;
            Some((dir.z * sc, -dir.y * sc))
        }
        Face::YPos if ay >= ax && ay >= az => {
            if dir.y <= 0.0 {
                return None;
            }
            let sc: f32 = 1.0 / ay;
            Some((dir.x * sc, dir.z * sc))
        }
        Face::YNeg if ay >= ax && ay >= az => {
            if dir.x >= 0.0 {
                return None;
            }
            let sc: f32 = -1.0 / ay;
            Some((dir.x * sc, -dir.z * sc))
        }
        Face::ZPos if az >= ax && az >= ay => {
            if dir.z <= 0.0 {
                return None;
            }
            let sc: f32 = 1.0 / az;
            Some((-dir.x * sc, dir.y * sc))
        }
        Face::ZNeg if az >= ax && az >= ay => {
            if dir.z >= 0.0 {
                return None;
            }
            let sc: f32 = -1.0 / az;
            Some((-dir.x * sc, -dir.y * sc))
        }
        _ => None,
    }
}

fn project_dir_to_face_xy(face: Face, dir: Vec3, width: f32, height: f32) -> Option<(i32, i32)> {
    if let Some((u, v)) = project_dir_to_face_uv(face, dir) {
        Some((((u * 0.5 + 0.5) * width) as i32, ((v * -0.5 + 0.5) * height) as i32))
    } else {
        None
    }
}

fn compute_sun_edge_16(sun_dir: Vec3, sun_radius: f32, offset_angle_rad: f32) -> [Vec3; 16] {
    // --- Step 1: stable orthonormal basis around sun_dir ---
    let up = if sun_dir.y.abs() < 0.99 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };

    let ex = cross(sun_dir, up).normalized();
    let ey = cross(sun_dir, ex).normalized();

    // --- Step 2: precompute cos/sin(radius) ---
    let cos_r = sun_radius.cos();
    let sin_r = sun_radius.sin();

    // --- Step 3: produce 16 angles around the circle ---
    let mut result = [Vec3::new(0.0, 0.0, 0.0); 16];

    for i in 0..16 {
        // angle = offset + i * 360°/16 = offset + i * 22.5°
        let phi = offset_angle_rad + (i as f32) * (std::f32::consts::TAU / 16.0);
        let phi_cos = phi.cos();
        let phi_sin = phi.sin();

        // spherical rim position
        // R = cos(r)*D + sin(r)*(cos(phi)*ex + sin(phi)*ey)
        let dir = sun_dir * cos_r + ex * (phi_cos * sin_r) + ey * (phi_sin * sin_r);

        result[i] = dir.normalized();
    }

    result
}

fn inject_sun(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    gamma_row: &[f32],
    neg_sun_size_inv: f32,
    sun_color: Vec3,
    x_min: usize,
    x_max: usize,
) {
    debug_assert!(r.len() == g.len() && r.len() == b.len() && r.len() == gamma_row.len());
    debug_assert!(x_min <= x_max);
    debug_assert!(x_max <= r.len());
    debug_assert_eq!(x_min % 4, 0);
    debug_assert_eq!(x_max % 4, 0);

    // Setup the raw pointers
    let mut r_ptr: *mut f32 = unsafe { r.as_mut_ptr().add(x_min) };
    let mut g_ptr: *mut f32 = unsafe { g.as_mut_ptr().add(x_min) };
    let mut b_ptr: *mut f32 = unsafe { b.as_mut_ptr().add(x_min) };
    let mut gamma_ptr: *const f32 = unsafe { gamma_row.as_ptr().add(x_min) };

    // Setup the uniforms
    let neg_sun_size_inv: F32x4 = F32x4::splat(neg_sun_size_inv);
    let one: F32x4 = F32x4::splat(1.0);
    let zero: F32x4 = F32x4::splat(0.0);
    let sun_color_r: F32x4 = F32x4::splat(sun_color.x);
    let sun_color_g: F32x4 = F32x4::splat(sun_color.y);
    let sun_color_b: F32x4 = F32x4::splat(sun_color.z);

    let steps: usize = (x_max - x_min) / 4;
    for _idx in 0..steps {
        // Load the inputs
        let gamma: F32x4 = F32x4::load(unsafe { *(gamma_ptr as *const [f32; 4]) });
        let r: F32x4 = F32x4::load(unsafe { *(r_ptr as *const [f32; 4]) });
        let g: F32x4 = F32x4::load(unsafe { *(g_ptr as *const [f32; 4]) });
        let b: F32x4 = F32x4::load(unsafe { *(b_ptr as *const [f32; 4]) });

        // sun_amount = (1.0 - gamma * sun_size_inv).max(0.0)
        let sun_amount: F32x4 = gamma.fma(neg_sun_size_inv, one).max(zero);

        // sun_color * (sun_amount * sun_amount);
        let sun_amount_2: F32x4 = sun_amount * sun_amount;
        let sun_r: F32x4 = sun_color_r * sun_amount_2;
        let sun_g: F32x4 = sun_color_g * sun_amount_2;
        let sun_b: F32x4 = sun_color_b * sun_amount_2;

        // sky_color += sun_color
        let r_out: F32x4 = r + sun_r;
        let g_out: F32x4 = g + sun_g;
        let b_out: F32x4 = b + sun_b;

        // Write the updated radiance out
        r_out.store_to(unsafe { &mut *(r_ptr as *mut [f32; 4]) });
        g_out.store_to(unsafe { &mut *(g_ptr as *mut [f32; 4]) });
        b_out.store_to(unsafe { &mut *(b_ptr as *mut [f32; 4]) });

        // Advance the input/output pointers
        r_ptr = unsafe { r_ptr.add(4) };
        g_ptr = unsafe { g_ptr.add(4) };
        b_ptr = unsafe { b_ptr.add(4) };
        gamma_ptr = unsafe { gamma_ptr.add(4) };
    }
}

fn build_face(sky: &HosekWilkieSky, face: Face, sun_dir: Vec3) -> Arc<Texture> {
    let width = 512;
    let height = 512;
    let tone_mapper = ReinhardToneMapper::new(0.5, 14.0);

    // Setup the Sun stuff
    let sun_zenith_color: Vec3 = Vec3::new(58.0, 55.0, 29.0);
    let sun_horizon_color: Vec3 = Vec3::new(60.0, 57.0, 27.0);
    let sun_base_size: f32 = 0.055;
    let sun_size: f32 = sun_base_size + (1.0 - sun_dir.y * sun_dir.y).sqrt() * sun_base_size * 0.25;
    let sun_neg_size_inv: f32 = -1.0 / sun_size;
    let sun_color: Vec3 = lerp(sun_horizon_color, sun_zenith_color, sun_dir.y.abs());
    let sun_edge_dirs: [Vec3; 16] = compute_sun_edge_16(sun_dir, sun_size, 3.14 / 32.0);
    let sun_edge_projected: [Option<(i32, i32)>; 16] =
        std::array::from_fn(|i| project_dir_to_face_xy(face, sun_edge_dirs[i], width as f32, height as f32));
    let sun_any_edge_projected: bool = sun_edge_projected.iter().any(|xy| xy.is_some());
    let (sun_min_y, sun_max_y, sun_min_x, sun_max_x): (i32, i32, i32, i32) = if sun_any_edge_projected {
        let mut min_y: i32 = i32::MAX;
        let mut max_y: i32 = i32::MIN;
        let mut min_x: i32 = i32::MAX;
        let mut max_x: i32 = i32::MIN;
        for sun_edge_xy in sun_edge_projected {
            if let Some((sun_edge_x, sun_edge_y)) = sun_edge_xy {
                min_y = min_y.min(sun_edge_y);
                max_y = max_y.max(sun_edge_y);
                min_x = min_x.min(sun_edge_x);
                max_x = max_x.max(sun_edge_x);
            }
        }
        let gap: i32 = 12;
        (min_y - gap, max_y + gap, min_x - gap, max_x + gap)
    } else {
        (0, -1, 0, -1)
    };
    let sun_min_x: usize = (sun_min_x.max(0) as usize) & (!3);
    let sun_max_x: usize = (sun_max_x.min(width as i32) as usize) & (!3);

    // Allocate buffers for intermediate results.
    let mut theta_cos_row: Vec<f32> = vec![0.0; width];
    let mut gamma_cos_row: Vec<f32> = vec![0.0; width];
    let mut gamma_row: Vec<f32> = vec![0.0; width];
    let mut r_row: Vec<f32> = vec![0.0; width];
    let mut g_row: Vec<f32> = vec![0.0; width];
    let mut b_row: Vec<f32> = vec![0.0; width];

    // Allocate the texture buffer.
    let mut texels: Vec<u8> = Vec::<u8>::new();
    texels.resize(width * height * 3, 127);
    let height_max = if face == Face::YPos { height } else { height / 2 };

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
            let gamma_cos_4: F32x4 = (normalized_vec_x_4 * sun_dir_x_4
                + normalized_vec_y_4 * sun_dir_y_4
                + normalized_vec_z_4 * sun_dir_z_4)
                .min(F32x4::splat(1.0))
                .max(F32x4::splat(-1.0));
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
        if sun_any_edge_projected && (y as i32 >= sun_min_y) && (y as i32 <= sun_max_y) {
            inject_sun(
                &mut r_row,
                &mut g_row,
                &mut b_row,
                &gamma_row,
                sun_neg_size_inv,
                sun_color,
                sun_min_x,
                sun_max_x,
            );
        }

        // Map the radiance values to RGB colors and store them in the texture.
        tone_mapper.map(&r_row, &g_row, &b_row, texels[y * width * 3..y * width * 3 + width * 3].as_mut(), y);

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

fn load_tex<P: AsRef<Path>>(path: P) -> Arc<Texture> {
    let image = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("res/").join(path))
        .unwrap()
        .into_rgba8();
    let width = image.width();
    let height = image.height();
    let texels: Vec<u8> = image.pixels().flat_map(|p| p.0[..4].iter().copied()).collect();
    Texture::new(&TextureSource { width, height, format: TextureFormat::RGBA, texels: &texels })
}

fn init_flares() -> Flares {
    let flare_tex1 = load_tex("flare1.png");
    let flare_tex2 = load_tex("flare2.png");
    let flare_tex3 = load_tex("flare3.png");
    let mut flares: Flares = Flares::new();

    let mut flare1: Flare = Flare::default();
    flare1.texture = flare_tex2.clone();
    flare1.scale1 = 0.2;
    flare1.scale0 = 0.1;
    flare1.angle1 = 0.7;
    flare1.angle0 = 0.1;
    flare1.alpha1 = -0.4;
    flare1.alpha0 = 0.6;
    flares.add_flare(flare1);

    let mut flare2: Flare = Flare::default();
    flare2.texture = flare_tex3.clone();
    flare2.scale1 = 0.1;
    flare2.pos1 = 0.5;
    flare2.scale1 = 0.1;
    flare2.scale0 = 0.05;
    flare2.angle1 = -0.3;
    flare2.angle0 = 0.2;
    flare2.alpha1 = -0.3;
    flare2.alpha0 = 1.0;
    flares.add_flare(flare2);

    let mut flare3: Flare = Flare::default();
    flare3.texture = flare_tex1.clone();
    flare3.pos1 = -1.5;
    flare3.scale1 = -0.1;
    flare3.scale0 = 0.1;
    flare3.angle0 = 0.3;
    flares.add_flare(flare3);

    let mut flare4: Flare = Flare::default();
    flare4.texture = flare_tex1.clone();
    flare4.scale1 = 0.1;
    flare4.pos1 = 0.05;
    flare4.scale1 = 0.05;
    flare4.scale0 = 0.02;
    flare4.angle1 = 0.2;
    flare4.angle0 = 0.5;
    flare4.alpha1 = -0.1;
    flare4.alpha0 = 0.8;
    flares.add_flare(flare4);

    let mut flare5: Flare = Flare::default();
    flare5.texture = flare_tex1.clone();
    flare5.scale1 = 0.1;
    flare5.pos1 = -0.5;
    flare5.scale1 = 0.05;
    flare5.scale0 = 0.02;
    flare5.angle1 = -0.1;
    flare5.angle0 = 0.3;
    flare5.alpha1 = -0.1;
    flare5.alpha0 = 0.6;
    flares.add_flare(flare5);

    flares
}

fn test_hosek_wilkie_sky() {
    // The reference outputs were copied from the results of running the code from the original paper.
    let sky1: HosekWilkieSky = HosekWilkieSky::new(2.0, Vec3::new(0.0, 0.0, 0.0), std::f32::consts::FRAC_PI_4);
    assert!(
        (sky1.f(0.0, std::f32::consts::FRAC_PI_4.cos(), 0.0f32.cos()) - Vec3::new(8.663214, 11.592292, 16.004868))
            .length()
            < 0.01
    );
    assert!(
        (sky1.f(0.1, std::f32::consts::FRAC_PI_4.cos(), 0.1f32.cos()) - Vec3::new(7.697937, 10.479785, 15.563609))
            .length()
            < 0.01
    );
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
    let mut viewport: Viewport = Viewport::default();
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(1, 1);
    let mut rasterizer = Rasterizer::new();
    let mut last = std::time::Instant::now();
    let mut t = 0.0;
    let mut dt: f32 = 0.0;
    let mut sun_dir: Vec3 = Vec3::new(0.0, 0.0, -1.0).normalized();
    let mut sky_turbidity: f32 = 3.0;
    let mut ground_albedo: Vec3 = Vec3::new(0.0, 0.0, 0.5);
    let mut rebuild_skybox: bool = true;
    let mut camera_orientation: Quat = Quat::from_axis_angle(Vec3::new(0.0, 0.0, -1.0), 0.0);
    let camera_position: Vec3 = Vec3::new(0.0, 2.0, 35.0);
    let mut show_wireframe: bool = false;
    let mut paused = false;
    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;
    let mut faces_build_time: f32 = 0.0;
    let mut faces_build_time_n: u32 = 0;
    let flares: Flares = init_flares();

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
                Event::KeyDown { keycode: Some(Keycode::T), .. }
                | Event::KeyDown { keycode: Some(Keycode::G), .. }
                | Event::KeyDown { keycode: Some(Keycode::Y), .. }
                | Event::KeyDown { keycode: Some(Keycode::H), .. }
                | Event::KeyDown { keycode: Some(Keycode::U), .. }
                | Event::KeyDown { keycode: Some(Keycode::J), .. } => {
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
                        let pitch: Quat =
                            Quat::from_axis_angle(camera_orientation * Vec3::new(1.0, 0.0, 0.0), angle_pitch);
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
            sun_dir = Vec3::new((t * 0.1).sin() * 0.5, (t * 0.1).sin(), -(t * 0.1).cos()).normalized();
            let theta_sun: f32 = sun_dir.y.acos(); // angle from zenith, radians
            let sun_elevation: f32 = (3.14 / 2.0 - theta_sun).max(0.0); // angle from the horizon, radians
            let sky: HosekWilkieSky = HosekWilkieSky::new(sky_turbidity, ground_albedo, sun_elevation);
            let faces = [Face::YPos, Face::XNeg, Face::XPos, Face::ZPos, Face::ZNeg];
            let start = std::time::Instant::now();
            use rayon::prelude::*;
            let results: Vec<Arc<Texture>> = faces.par_iter().map(|face| build_face(&sky, *face, sun_dir)).collect();
            pos_y_tex = results[0].clone();
            neg_x_tex = results[1].clone();
            pos_x_tex = results[2].clone();
            pos_z_tex = results[3].clone();
            neg_z_tex = results[4].clone();
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
            viewport = Viewport::new(0, 0, size.0 as u16, size.1 as u16);
            rasterizer.setup(viewport);
        }
        color_buffer.fill(RGBA::new(102, 204, 255, 255).to_u32());
        rasterizer.reset();
        rasterizer.set_draw_wireframe(show_wireframe);

        // Commit the draw commands
        let projection: Mat44 =
            Mat44::perspective(1.0, 100.0, std::f32::consts::PI / 3.0, size.0 as f32 / size.1 as f32);
        let view: Mat44 = camera_to_mat34(camera_orientation, camera_position).as_mat44();
        let view_orientation: Mat44 = view.as_mat33().as_mat44();

        // draw the skybox and the flares
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
        flares.commit(viewport, &view_orientation, &projection, sun_dir, &mut rasterizer);
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

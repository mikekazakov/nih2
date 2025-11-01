use criterion::{Bencher, BenchmarkId, Criterion, criterion_group, criterion_main};
use nih::math::*;
use nih::render::*;

// 4884 triangles of 64x64 pixels each, i.e. 64x64x4884/2 ~= 10M pixels
// Laid out front-to-back from Z=1 to Z=-1
fn build_4884_triangles_coords() -> Vec<Vec3> {
    let base_batch = [
        // UL-LL-LR
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        // UR-LL-LR
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        // // UR-UL-LL
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        // // UR-LL-LR
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
    ];
    let batches = 4884 / 4;
    let mut coords = Vec::<Vec3>::new();
    let mut z: f32 = 1.0;
    for i in 0..batches {
        coords.extend(base_batch.iter());
        for j in (coords.len() - 12..coords.len()).step_by(6) {
            z -= 0.0008;
            coords[j + 0].z = z;
            coords[j + 1].z = z;
            coords[j + 2].z = z;
            coords[j + 3].z = z;
            coords[j + 4].z = z;
            coords[j + 5].z = z;
        }
    }
    assert_eq!(coords.len(), 4884 * 3);
    coords
}

fn build_4884_triangles_fixed_colors() -> Vec<Vec4> {
    let base_batch = [Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(1.0, 0.0, 0.0, 1.0)];
    let batches = 4884;
    let mut colors = Vec::<Vec4>::new();
    for i in 0..batches {
        colors.extend(base_batch.iter());
    }
    assert_eq!(colors.len(), 4884 * 3);
    colors
}

fn build_4884_triangles_varying_colors() -> Vec<Vec4> {
    let base_batch = [Vec4::new(1.0, 0.0, 0.0, 1.0), Vec4::new(0.0, 1.0, 0.0, 1.0), Vec4::new(0.0, 0.0, 1.0, 1.0)];
    let batches = 4884;
    let mut colors = Vec::<Vec4>::new();
    for i in 0..batches {
        colors.extend(base_batch.iter());
    }
    assert_eq!(colors.len(), 4884 * 3);
    colors
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
    color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
    let mut rasterizer = Rasterizer::new();
    rasterizer.setup(Viewport::new(0, 0, 64, 64));

    let tris_positions = build_4884_triangles_coords();
    let tris_fixed_colors = build_4884_triangles_fixed_colors();
    let tris_varying_colors = build_4884_triangles_varying_colors();

    let plain = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand { world_positions: &tris_positions, ..Default::default() };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
            std::hint::black_box(color_buffer);
        })
    };
    let fixed_colors = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand {
                world_positions: &tris_positions,
                colors: &tris_fixed_colors,
                ..Default::default()
            };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
            std::hint::black_box(color_buffer);
        })
    };
    let varying_colors = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand {
                world_positions: &tris_positions,
                colors: &tris_varying_colors,
                ..Default::default()
            };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
            std::hint::black_box(color_buffer);
        })
    };
    let alpha_blending = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand {
                world_positions: &tris_positions,
                colors: &tris_varying_colors,
                alpha_blending: AlphaBlendingMode::Normal,
                ..Default::default()
            };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer { color_buffer: Some(&mut color_buffer), ..Framebuffer::default() });
            std::hint::black_box(color_buffer);
        })
    };
    let depth = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(64, 64);
            depth_buffer.fill(u16::MAX);
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand {
                world_positions: &tris_positions,
                colors: &tris_varying_colors,
                alpha_blending: AlphaBlendingMode::Normal,
                ..Default::default()
            };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer {
                color_buffer: Some(&mut color_buffer),
                depth_buffer: Some(&mut depth_buffer),
                ..Framebuffer::default()
            });
            std::hint::black_box(color_buffer);
        })
    };
    let normals = |bencher: &mut Bencher| {
        bencher.iter(|| {
            let mut color_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            color_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut normals_buffer = TiledBuffer::<u32, 64, 64>::new(64, 64);
            normals_buffer.fill(RGBA::new(0, 0, 0, 255).to_u32());
            let mut depth_buffer = TiledBuffer::<u16, 64, 64>::new(64, 64);
            depth_buffer.fill(u16::MAX);
            let mut rasterizer = Rasterizer::new();
            rasterizer.setup(Viewport::new(0, 0, 64, 64));
            let command = RasterizationCommand {
                world_positions: &tris_positions,
                colors: &tris_varying_colors,
                alpha_blending: AlphaBlendingMode::Normal,
                ..Default::default()
            };
            rasterizer.commit(&command);
            rasterizer.draw(&mut Framebuffer {
                color_buffer: Some(&mut color_buffer),
                depth_buffer: Some(&mut depth_buffer),
                normal_buffer: Some(&mut normals_buffer),
                ..Framebuffer::default()
            });
            std::hint::black_box(color_buffer);
        })
    };

    let mut group = c.benchmark_group("Fill 10Mpx");
    group.bench_function(BenchmarkId::new("64x64", "0 plain"), &plain);
    group.bench_function(BenchmarkId::new("64x64", "1 fixed colors"), &fixed_colors);
    group.bench_function(BenchmarkId::new("64x64", "2 varying colors"), &varying_colors);
    group.bench_function(BenchmarkId::new("64x64", "3 alpha blend"), &alpha_blending);
    group.bench_function(BenchmarkId::new("64x64", "4 depth"), &depth);
    group.bench_function(BenchmarkId::new("64x64", "5 normals"), &normals);
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

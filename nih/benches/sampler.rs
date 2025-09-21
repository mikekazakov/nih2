use criterion::{Bencher, BenchmarkId, Criterion, criterion_group, criterion_main};
use nih::render::*;

fn sample_1m(sampler: &Sampler) {
    for y in (0..1000).map(|y| y as f32 * 0.001) {
        for x in (0..1000).map(|x| x as f32 * 0.001) {
            std::hint::black_box(sampler.sample(x, y)); // NB! implicitly applies uv-scale
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let texture_grayscale = Texture::new(&TextureSource {
        texels: &vec![255u8; 1024 * 1024],
        width: 1024,
        height: 1024,
        format: TextureFormat::Grayscale,
    });
    let texture_rgb = Texture::new(&TextureSource {
        texels: &vec![255u8; 1024 * 1024 * 3],
        width: 1024,
        height: 1024,
        format: TextureFormat::RGB,
    });
    let sampler_nearest_grayscale = Sampler::new(&texture_grayscale, SamplerFilter::Nearest, 0.5);
    let sampler_nearest_rgb = Sampler::new(&texture_rgb, SamplerFilter::Nearest, 0.5);
    let sampler_bilinear_grayscale = Sampler::new(&texture_grayscale, SamplerFilter::Bilinear, 0.5);
    let sampler_bilinear_rgb = Sampler::new(&texture_rgb, SamplerFilter::Bilinear, 0.5);
    let sampler_trilinear_grayscale = Sampler::new(&texture_grayscale, SamplerFilter::Trilinear, 0.5);
    let sampler_trilinear_rgb = Sampler::new(&texture_rgb, SamplerFilter::Trilinear, 0.5);
    fn runner(bencher: &mut Bencher, sampler: &Sampler) {
        bencher.iter(|| {
            sample_1m(sampler);
        })
    }
    let mut group = c.benchmark_group("Sample 1M");
    group.bench_with_input(BenchmarkId::new("Nearest", "Gray"), &sampler_nearest_grayscale, runner);
    group.bench_with_input(BenchmarkId::new("Nearest", "RGB"), &sampler_nearest_rgb, runner);
    group.bench_with_input(BenchmarkId::new("Bilinear", "Gray"), &sampler_bilinear_grayscale, runner);
    group.bench_with_input(BenchmarkId::new("Bilinear", "RGB"), &sampler_bilinear_rgb, runner);
    group.bench_with_input(BenchmarkId::new("Trilinear", "Gray"), &sampler_trilinear_grayscale, runner);
    group.bench_with_input(BenchmarkId::new("Trilinear", "RGB"), &sampler_trilinear_rgb, runner);
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

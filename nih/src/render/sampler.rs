use super::{FramebufferTile, RGBA, Texture, Vertex, Viewport};

pub enum SamplerFilter {
    Nearest,
    Bilinear,
    Trilinear,
}

type SampleFunction = fn(*const u8, f32, f32) -> RGBA;

pub struct Sampler {
    texels0: *const u8,
    sample_function: SampleFunction,
}

impl Sampler {
    pub fn new(texture: &std::sync::Arc<Texture>, lod: f32) -> Self {
        let mips = texture.count;
        let mip0_index = ((lod as f32).floor() as i32).clamp(0, mips as i32 - 1);
        let mip0 = &texture.mips[mip0_index as usize];
        let texels0 = unsafe { texture.texels.as_ptr().add(mip0.offset as usize) };
        let sample_function = match (mip0.width) {
            1 => sample::<1>,
            2 => sample::<2>,
            4 => sample::<4>,
            8 => sample::<8>,
            16 => sample::<16>,
            32 => sample::<32>,
            64 => sample::<64>,
            128 => sample::<128>,
            256 => sample::<256>,
            512 => sample::<512>,
            1024 => sample::<1024>,
            _ => {
                panic!("Invalid texture size")
            }
        };
        Sampler { texels0, sample_function }
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBA {
        (self.sample_function)(self.texels0, u, v)
    }
}

fn sample<const SIZE: u16>(texels: *const u8, u: f32, v: f32) -> RGBA {
    let bpp: usize = 1;
    let stride: usize = SIZE as usize * bpp;
    let fwidth: f32 = SIZE as f32;
    let fheight: f32 = SIZE as f32;
    let tx: f32 = (u + 10.0) * fwidth;
    let ty: f32 = (v + 10.0) * fheight;
    let itx: i32 = unsafe { tx.to_int_unchecked() };
    let ity: i32 = unsafe { ty.to_int_unchecked() };
    let x: usize = (itx as usize) & (SIZE as usize - 1);
    let y: usize = (ity as usize) & (SIZE as usize - 1);
    let offset: usize = y * stride + x * bpp;
    let texel: *const u8 = unsafe { texels.add(offset) };
    let c: u8 = unsafe { *texel };
    RGBA::new(c, c, c, 255)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{TextureFormat, TextureSource};

    #[test]
    fn test_sample_nearest_from_1x1_grayscale_texture() {
        let texture =
            Texture::new(&TextureSource { texels: &[42u8], width: 1, height: 1, format: TextureFormat::Grayscale });
        let sampler = Sampler::new(&texture, 0.0);
        assert_eq!(sampler.sample(0.0, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(1.0, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.0, 1.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(1.0, 1.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(-1.0, -1.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.5, 0.5), RGBA::new(42, 42, 42, 255));
    }

    #[test]
    fn test_sample_nearest_from_2x2_grayscale_texture() {
        let texture = Texture::new(&TextureSource {
            texels: &[42u8, 43u8, 44u8, 45u8],
            width: 2,
            height: 2,
            format: TextureFormat::Grayscale,
        });
        let sampler = Sampler::new(&texture, 0.0);
        assert_eq!(sampler.sample(0.1, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.25, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.4, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.6, 0.0), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(0.75, 0.0), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(0.9, 0.0), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(1.1, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(1.9, 0.0), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(-0.1, 0.0), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(-0.75, 0.0), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(-0.1, 0.9), RGBA::new(45, 45, 45, 255));
        assert_eq!(sampler.sample(1.1, 0.9), RGBA::new(44, 44, 44, 255));
        assert_eq!(sampler.sample(0.1, 0.1), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.1, 0.4), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.1, 0.6), RGBA::new(44, 44, 44, 255));
        assert_eq!(sampler.sample(0.1, 0.9), RGBA::new(44, 44, 44, 255));
        assert_eq!(sampler.sample(0.6, 0.1), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(0.6, 0.4), RGBA::new(43, 43, 43, 255));
        assert_eq!(sampler.sample(0.6, 0.6), RGBA::new(45, 45, 45, 255));
        assert_eq!(sampler.sample(0.6, 0.9), RGBA::new(45, 45, 45, 255));
        assert_eq!(sampler.sample(0.1, -0.6), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(0.1, -0.1), RGBA::new(44, 44, 44, 255));
        assert_eq!(sampler.sample(0.1, 1.1), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(1.1, 1.1), RGBA::new(42, 42, 42, 255));
        assert_eq!(sampler.sample(-0.1, -0.1), RGBA::new(45, 45, 45, 255));
    }
}

use super::*;

#[repr(u8)]
pub enum SamplerFilter {
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2,
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
        let sample_function = match texture.format {
            TextureFormat::Grayscale => match mip0.width {
                1 => sample_nearest::<1, { TextureFormat::Grayscale as u8 }>,
                2 => sample_nearest::<2, { TextureFormat::Grayscale as u8 }>,
                4 => sample_nearest::<4, { TextureFormat::Grayscale as u8 }>,
                8 => sample_nearest::<8, { TextureFormat::Grayscale as u8 }>,
                16 => sample_nearest::<16, { TextureFormat::Grayscale as u8 }>,
                32 => sample_nearest::<32, { TextureFormat::Grayscale as u8 }>,
                64 => sample_nearest::<64, { TextureFormat::Grayscale as u8 }>,
                128 => sample_nearest::<128, { TextureFormat::Grayscale as u8 }>,
                256 => sample_nearest::<256, { TextureFormat::Grayscale as u8 }>,
                512 => sample_nearest::<512, { TextureFormat::Grayscale as u8 }>,
                1024 => sample_nearest::<1024, { TextureFormat::Grayscale as u8 }>,
                _ => {
                    panic!("Invalid texture size")
                }
            },
            TextureFormat::RGB => match mip0.width {
                1 => sample_nearest::<1, { TextureFormat::RGB as u8 }>,
                2 => sample_nearest::<2, { TextureFormat::RGB as u8 }>,
                4 => sample_nearest::<4, { TextureFormat::RGB as u8 }>,
                8 => sample_nearest::<8, { TextureFormat::RGB as u8 }>,
                16 => sample_nearest::<16, { TextureFormat::RGB as u8 }>,
                32 => sample_nearest::<32, { TextureFormat::RGB as u8 }>,
                64 => sample_nearest::<64, { TextureFormat::RGB as u8 }>,
                128 => sample_nearest::<128, { TextureFormat::RGB as u8 }>,
                256 => sample_nearest::<256, { TextureFormat::RGB as u8 }>,
                512 => sample_nearest::<512, { TextureFormat::RGB as u8 }>,
                1024 => sample_nearest::<1024, { TextureFormat::RGB as u8 }>,
                _ => {
                    panic!("Invalid texture size")
                }
            },
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

impl Default for Sampler {
    fn default() -> Self {
        Sampler { texels0: std::ptr::null(), sample_function: noop_sample }
    }
}

fn noop_sample(_texels: *const u8, _u: f32, _v: f32) -> RGBA {
    RGBA::new(0, 0, 0, 255)
}

fn sample_nearest<const SIZE: u16, const FORMAT: u8>(texels: *const u8, u: f32, v: f32) -> RGBA {
    let bpp: usize = bytes_per_pixel_u8(FORMAT);
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
    if FORMAT == TextureFormat::Grayscale as u8 {
        let c: u8 = unsafe { *texel };
        return RGBA::new(c, c, c, 255);
    }
    if FORMAT == TextureFormat::RGB as u8 {
        let r: u8 = unsafe { *texel.add(0) };
        let g: u8 = unsafe { *texel.add(1) };
        let b: u8 = unsafe { *texel.add(2) };
        return RGBA::new(r, g, b, 255);
        // Somehow this is slower than the above code:
        // return RGBA::from_u32( unsafe { (texel as *const u32).read_unaligned() } | 0xFF000000 );
    }
    RGBA::new(0, 0, 0, 255)
}

const fn bytes_per_pixel_u8(fmt: u8) -> usize {
    match fmt {
        x if x == TextureFormat::RGBA as u8 => 4,
        x if x == TextureFormat::RGB as u8 => 3,
        x if x == TextureFormat::Grayscale as u8 => 1,
        _ => unreachable!(),
    }
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
        {
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
        {
            let sampler = Sampler::new(&texture, 1.0);
            assert_eq!(sampler.sample(0.0, 0.0), RGBA::new(44, 44, 44, 255));
            assert_eq!(sampler.sample(0.9, 0.9), RGBA::new(44, 44, 44, 255));
            assert_eq!(sampler.sample(5.9, 0.9), RGBA::new(44, 44, 44, 255));
            assert_eq!(sampler.sample(5.9, -3.9), RGBA::new(44, 44, 44, 255));
        }
    }

    #[test]
    fn test_sample_nearest_from_2x2_rgb_texture() {
        // Texel layout (row-major):
        // [ (0,0): red, (1,0): green ]
        // [ (0,1): blue, (1,1): white ]
        // RGB: red=(255,0,0), green=(0,255,0), blue=(0,0,255), white=(255,255,255)
        let texels: [u8; 12] = [
            255, 0, 0, // (0,0) red
            0, 255, 0, // (1,0) green
            0, 0, 255, // (0,1) blue
            255, 255, 255, // (1,1) white
        ];
        let texture = Texture::new(&TextureSource { texels: &texels, width: 2, height: 2, format: TextureFormat::RGB });
        let sampler = Sampler::new(&texture, 0.0);
        // Top-left (should be red)
        assert_eq!(sampler.sample(0.1, 0.1), RGBA::new(255, 0, 0, 255));
        // Top-right (should be green)
        assert_eq!(sampler.sample(0.6, 0.1), RGBA::new(0, 255, 0, 255));
        // Bottom-left (should be blue)
        assert_eq!(sampler.sample(0.1, 0.6), RGBA::new(0, 0, 255, 255));
        // Bottom-right (should be white)
        assert_eq!(sampler.sample(0.6, 0.6), RGBA::new(255, 255, 255, 255));
        // Center (should be red, as nearest wraps to (0,0))
        assert_eq!(sampler.sample(0.25, 0.25), RGBA::new(255, 0, 0, 255));
        // Edges and out-of-bounds (wrapping)
        // u: 1.1 wraps to 0 (red), v: 0.1
        assert_eq!(sampler.sample(1.1, 0.1), RGBA::new(255, 0, 0, 255));
        // u: -0.1 wraps to 1 (green), v: 0.1
        assert_eq!(sampler.sample(-0.1, 0.1), RGBA::new(0, 255, 0, 255));
        // u: 0.1, v: 1.1 wraps to 0 (red)
        assert_eq!(sampler.sample(0.1, 1.1), RGBA::new(255, 0, 0, 255));
        // u: 0.6, v: 1.1 wraps to 0 (green)
        assert_eq!(sampler.sample(0.6, 1.1), RGBA::new(0, 255, 0, 255));
        // u: -0.1, v: -0.1 wraps to (1,1) (white)
        assert_eq!(sampler.sample(-0.1, -0.1), RGBA::new(255, 255, 255, 255));
        // u: 1.1, v: 1.1 wraps to (0,0) (red)
        assert_eq!(sampler.sample(1.1, 1.1), RGBA::new(255, 0, 0, 255));
    }
}

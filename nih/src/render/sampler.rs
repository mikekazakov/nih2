use super::*;

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SamplerFilter {
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2,
}

type SampleFunction = fn(*const u8, f32, f32) -> RGBA;

#[derive(Clone, Copy, Debug)]
pub struct SamplerUVScale {
    // U or V coordinate is first biased by this value...
    pub bias: f32,

    // ... then scaled by this value.
    pub scale: f32,
}

pub struct Sampler {
    texels0: *const u8,
    sample_function: SampleFunction,
    uv_scale: SamplerUVScale,
}

impl Sampler {
    pub fn new(texture: &std::sync::Arc<Texture>, filtering: SamplerFilter, lod: f32) -> Self {
        let mips = texture.count;
        let mip0_index = ((lod as f32).floor() as i32).clamp(0, mips as i32 - 1);
        let mip0 = &texture.mips[mip0_index as usize];
        let texels0 = unsafe { texture.texels.as_ptr().add(mip0.offset as usize) };
        let sample_function = if filtering == SamplerFilter::Nearest {
            match texture.format {
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
            }
        } else if filtering == SamplerFilter::Bilinear {
            match texture.format {
                TextureFormat::Grayscale => match mip0.width {
                    1 => sample_bilinear::<1, { TextureFormat::Grayscale as u8 }>,
                    2 => sample_bilinear::<2, { TextureFormat::Grayscale as u8 }>,
                    4 => sample_bilinear::<4, { TextureFormat::Grayscale as u8 }>,
                    8 => sample_bilinear::<8, { TextureFormat::Grayscale as u8 }>,
                    16 => sample_bilinear::<16, { TextureFormat::Grayscale as u8 }>,
                    32 => sample_bilinear::<32, { TextureFormat::Grayscale as u8 }>,
                    64 => sample_bilinear::<64, { TextureFormat::Grayscale as u8 }>,
                    128 => sample_bilinear::<128, { TextureFormat::Grayscale as u8 }>,
                    256 => sample_bilinear::<256, { TextureFormat::Grayscale as u8 }>,
                    512 => sample_bilinear::<512, { TextureFormat::Grayscale as u8 }>,
                    1024 => sample_bilinear::<1024, { TextureFormat::Grayscale as u8 }>,
                    _ => {
                        panic!("Invalid texture size")
                    }
                },
                TextureFormat::RGB => match mip0.width {
                    1 => sample_bilinear::<1, { TextureFormat::RGB as u8 }>, // TODO: cheat and use nearest
                    2 => sample_bilinear::<2, { TextureFormat::RGB as u8 }>,
                    4 => sample_bilinear::<4, { TextureFormat::RGB as u8 }>,
                    8 => sample_bilinear::<8, { TextureFormat::RGB as u8 }>,
                    16 => sample_bilinear::<16, { TextureFormat::RGB as u8 }>,
                    32 => sample_bilinear::<32, { TextureFormat::RGB as u8 }>,
                    64 => sample_bilinear::<64, { TextureFormat::RGB as u8 }>,
                    128 => sample_bilinear::<128, { TextureFormat::RGB as u8 }>,
                    256 => sample_bilinear::<256, { TextureFormat::RGB as u8 }>,
                    512 => sample_bilinear::<512, { TextureFormat::RGB as u8 }>,
                    1024 => sample_bilinear::<1024, { TextureFormat::RGB as u8 }>,
                    _ => {
                        panic!("Invalid texture size")
                    }
                },
                _ => {
                    panic!("Invalid texture size")
                }
            }
        } else {
            panic!("Invalid filtering")
        };

        let uv_scale = if filtering == SamplerFilter::Nearest {
            match mip0.width {
                1 => SamplerUVScale { bias: 10.0, scale: 1.0 },
                2 => SamplerUVScale { bias: 10.0, scale: 2.0 },
                4 => SamplerUVScale { bias: 10.0, scale: 4.0 },
                8 => SamplerUVScale { bias: 10.0, scale: 8.0 },
                16 => SamplerUVScale { bias: 10.0, scale: 16.0 },
                32 => SamplerUVScale { bias: 10.0, scale: 32.0 },
                64 => SamplerUVScale { bias: 10.0, scale: 64.0 },
                128 => SamplerUVScale { bias: 10.0, scale: 128.0 },
                256 => SamplerUVScale { bias: 10.0, scale: 256.0 },
                512 => SamplerUVScale { bias: 10.0, scale: 512.0 },
                1024 => SamplerUVScale { bias: 10.0, scale: 1024.0 },
                _ => {
                    panic!("Invalid texture size")
                }
            }
        } else if filtering == SamplerFilter::Bilinear {
            match mip0.width {
                1 => SamplerUVScale { bias: 10.0, scale: 1.0 * 256.0 },
                2 => SamplerUVScale { bias: 10.0, scale: 2.0 * 256.0 },
                4 => SamplerUVScale { bias: 10.0, scale: 4.0 * 256.0 },
                8 => SamplerUVScale { bias: 10.0, scale: 8.0 * 256.0 },
                16 => SamplerUVScale { bias: 10.0, scale: 16.0 * 256.0 },
                32 => SamplerUVScale { bias: 10.0, scale: 32.0 * 256.0 },
                64 => SamplerUVScale { bias: 10.0, scale: 64.0 * 256.0 },
                128 => SamplerUVScale { bias: 10.0, scale: 128.0 * 256.0 },
                256 => SamplerUVScale { bias: 10.0, scale: 256.0 * 256.0 },
                512 => SamplerUVScale { bias: 10.0, scale: 512.0 * 256.0 },
                1024 => SamplerUVScale { bias: 10.0, scale: 1024.0 * 256.0 },
                _ => {
                    panic!("Invalid texture size")
                }
            }
        } else {
            panic!("Invalid filtering")
        };

        Sampler { texels0, sample_function, uv_scale }
    }

    pub fn sample_prescaled(&self, u: f32, v: f32) -> RGBA {
        (self.sample_function)(self.texels0, u, v)
    }

    pub fn sample(&self, u: f32, v: f32) -> RGBA {
        let tu = (u + self.uv_scale.bias) * self.uv_scale.scale;
        let tv = (v + self.uv_scale.bias) * self.uv_scale.scale;
        (self.sample_function)(self.texels0, tu, tv)
    }

    pub fn uv_scale(&self) -> SamplerUVScale {
        self.uv_scale
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler { texels0: std::ptr::null(), sample_function: noop_sample, uv_scale: SamplerUVScale::default() }
    }
}

impl Default for SamplerUVScale {
    fn default() -> Self {
        SamplerUVScale { bias: 0.0, scale: 1.0 }
    }
}

fn noop_sample(_texels: *const u8, _u: f32, _v: f32) -> RGBA {
    RGBA::new(0, 0, 0, 255)
}

fn sample_nearest<const SIZE: u16, const FORMAT: u8>(texels: *const u8, u: f32, v: f32) -> RGBA {
    debug_assert!(u >= 0.0 && v >= 1.0);
    let bpp: usize = bytes_per_pixel_u8(FORMAT);
    let stride: usize = SIZE as usize * bpp;
    let itx: i32 = unsafe { u.to_int_unchecked() };
    let ity: i32 = unsafe { v.to_int_unchecked() };
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

fn sample_bilinear<const SIZE: u16, const FORMAT: u8>(texels: *const u8, u: f32, v: f32) -> RGBA {
    debug_assert!(u >= 0.0 && v >= 1.0);
    let bpp: usize = bytes_per_pixel_u8(FORMAT);
    let stride: usize = SIZE as usize * bpp;
    let half: i32 = 127;
    let offset: i32 = 1 << 20; // why so much??
    let itx: i32 = unsafe { u.to_int_unchecked() };
    let ity: i32 = unsafe { v.to_int_unchecked() };
    let bx: u32 = (itx + offset - half) as u32;
    let by: u32 = (ity + offset - half) as u32;
    let wx1: u32 = bx & 255;
    let wx: u32 = 256 - wx1;
    let wy1: u32 = by & 255;
    let wy: u32 = 256 - wy1;
    let wa: u32 = wx * wy;
    let wb: u32 = wx1 * wy;
    let wc: u32 = wx * wy1;
    let wd: u32 = wx1 * wy1;
    let x0: u32 = bx >> 8;
    let x1: u32 = x0 + 1;
    let y0: u32 = by >> 8;
    let y1: u32 = y0 + 1;
    let tx0: u32 = x0 & (SIZE as u32 - 1);
    let tx1: u32 = x1 & (SIZE as u32 - 1);
    let ty0: u32 = y0 & (SIZE as u32 - 1);
    let ty1: u32 = y1 & (SIZE as u32 - 1);
    let offset_a: usize = ty0 as usize * stride + tx0 as usize * bpp;
    let offset_b: usize = ty0 as usize * stride + tx1 as usize * bpp;
    let offset_c: usize = ty1 as usize * stride + tx0 as usize * bpp;
    let offset_d: usize = ty1 as usize * stride + tx1 as usize * bpp;
    if FORMAT == TextureFormat::Grayscale as u8 {
        let a: u8 = unsafe { *texels.add(offset_a) };
        let b: u8 = unsafe { *texels.add(offset_b) };
        let c: u8 = unsafe { *texels.add(offset_c) };
        let d: u8 = unsafe { *texels.add(offset_d) };
        let abcd: u32 = (a as u32) * wa + (b as u32) * wb + (c as u32) * wc + (d as u32) * wd;
        let result: u8 = (abcd >> 16) as u8;
        return RGBA::new(result, result, result, 255);
    }
    if FORMAT == TextureFormat::RGB as u8 {
        let a: u32 = unsafe { (texels.add(offset_a) as *const u32).read_unaligned() };
        let b: u32 = unsafe { (texels.add(offset_b) as *const u32).read_unaligned() };
        let c: u32 = unsafe { (texels.add(offset_c) as *const u32).read_unaligned() };
        let d: u32 = unsafe { (texels.add(offset_d) as *const u32).read_unaligned() };
        let r: u32 = (a & 0xFF) * wa + (b & 0xFF) * wb + (c & 0xFF) * wc + (d & 0xFF) * wd;
        let g: u32 = ((a >> 8) & 0xFF) * wa + ((b >> 8) & 0xFF) * wb + ((c >> 8) & 0xFF) * wc + ((d >> 8) & 0xFF) * wd;
        let b: u32 =
            ((a >> 16) & 0xFF) * wa + ((b >> 16) & 0xFF) * wb + ((c >> 16) & 0xFF) * wc + ((d >> 16) & 0xFF) * wd;
        return RGBA::new((r >> 16) as u8, (g >> 16) as u8, (b >> 16) as u8, 255);
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

    #[macro_export]
    macro_rules! assert_rgba_eq {
        ($left:expr, $right:expr, $tol:expr $(,)?) => {{
            let l = $left;
            let r = $right;
            let tol: i16 = $tol as i16;

            let dr = (l.r as i16 - r.r as i16).abs();
            let dg = (l.g as i16 - r.g as i16).abs();
            let db = (l.b as i16 - r.b as i16).abs();
            let da = (l.a as i16 - r.a as i16).abs();

            if dr > tol || dg > tol || db > tol || da > tol {
                panic!("assertion failed: left != right within tol={}\n  left: {:?}\n right: {:?}", tol, l, r);
            }
        }};
    }

    #[test]
    fn test_sample_nearest_from_1x1_grayscale_texture() {
        let texture =
            Texture::new(&TextureSource { texels: &[42u8], width: 1, height: 1, format: TextureFormat::Grayscale });
        let sampler = Sampler::new(&texture, SamplerFilter::Nearest, 0.0);
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
            let sampler = Sampler::new(&texture, SamplerFilter::Nearest, 0.0);
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
            let sampler = Sampler::new(&texture, SamplerFilter::Nearest, 1.0);
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
        let sampler = Sampler::new(&texture, SamplerFilter::Nearest, 0.0);
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

    #[test]
    fn test_sample_bilinear_from_1x1_grayscale_texture() {
        let texture =
            Texture::new(&TextureSource { texels: &[250u8], width: 1, height: 1, format: TextureFormat::Grayscale });
        let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 0.0);
        assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(250, 250, 250, 255), 1);
        assert_rgba_eq!(sampler.sample(1.0, 0.0), RGBA::new(250, 250, 250, 255), 1);
        assert_rgba_eq!(sampler.sample(0.0, 1.0), RGBA::new(250, 250, 250, 255), 1);
        assert_rgba_eq!(sampler.sample(1.0, 1.0), RGBA::new(250, 250, 250, 255), 1);
        assert_rgba_eq!(sampler.sample(-1.0, -1.0), RGBA::new(250, 250, 250, 255), 1);
        assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(250, 250, 250, 255), 1);
    }

    #[test]
    fn test_sample_bilinear_from_2x2_grayscale_texture_0() {
        let texture = Texture::new(&TextureSource {
            texels: &[255u8, 0u8, 0u8, 0u8],
            width: 2,
            height: 2,
            format: TextureFormat::Grayscale,
        });
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 0.0);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(255, 255, 255, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(1.25, 0.25), RGBA::new(255, 255, 255, 255), 2);
            assert_rgba_eq!(sampler.sample(1.75, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 1.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(1.75, 1.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.0, 0.25), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.25), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(1.0, 0.25), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 0.5), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 1.0), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(1.0, 1.0), RGBA::new(64, 64, 64, 255), 2);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 1.0);
            assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.9, 0.9), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(5.9, 0.9), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(5.9, -3.9), RGBA::new(64, 64, 64, 255), 2);
        }
    }

    #[test]
    fn test_sample_bilinear_from_2x2_grayscale_texture_1() {
        let texture = Texture::new(&TextureSource {
            texels: &[0u8, 0u8, 0u8, 255u8],
            width: 2,
            height: 2,
            format: TextureFormat::Grayscale,
        });
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 0.0);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(255, 255, 255, 255), 2);
            assert_rgba_eq!(sampler.sample(1.25, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(1.75, 0.25), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(0.25, 1.75), RGBA::new(0, 0, 0, 255), 2);
            assert_rgba_eq!(sampler.sample(1.75, 1.75), RGBA::new(255, 255, 255, 255), 2);
            assert_rgba_eq!(sampler.sample(0.0, 0.75), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.75), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(1.0, 0.75), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 0.5), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.75, 1.0), RGBA::new(127, 127, 127, 255), 2);
            assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(1.0, 1.0), RGBA::new(64, 64, 64, 255), 2);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 1.0);
            assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(0.9, 0.9), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(5.9, 0.9), RGBA::new(64, 64, 64, 255), 2);
            assert_rgba_eq!(sampler.sample(5.9, -3.9), RGBA::new(64, 64, 64, 255), 2);
        }
    }

    #[test]
    fn test_sample_bilinear_from_1x1_rgb_texture() {
        let texture = Texture::new(&TextureSource {
            texels: &[250u8, 150u8, 50u8],
            width: 1,
            height: 1,
            format: TextureFormat::RGB,
        });
        let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 0.0);
        assert_rgba_eq!(sampler.sample(0.0, 0.0), RGBA::new(250, 150, 50, 255), 1);
        assert_rgba_eq!(sampler.sample(1.0, 0.0), RGBA::new(250, 150, 50, 255), 1);
        assert_rgba_eq!(sampler.sample(0.0, 1.0), RGBA::new(250, 150, 50, 255), 1);
        assert_rgba_eq!(sampler.sample(1.0, 1.0), RGBA::new(250, 150, 50, 255), 1);
        assert_rgba_eq!(sampler.sample(-1.0, -1.0), RGBA::new(250, 150, 50, 255), 1);
        assert_rgba_eq!(sampler.sample(0.5, 0.5), RGBA::new(250, 150, 50, 255), 1);
    }

    #[test]
    fn test_sample_bilinear_from_2x2_rgb_texture() {
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
        let sampler = Sampler::new(&texture, SamplerFilter::Bilinear, 0.0);
        assert_rgba_eq!(sampler.sample(0.00, 0.00), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.25, 0.00), RGBA::new(127, 0, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.50, 0.00), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.75, 0.00), RGBA::new(127, 255, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(1.00, 0.00), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.00, 0.25), RGBA::new(127, 127, 0, 255), 2);
        assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(255, 0, 0, 255), 2);
        assert_rgba_eq!(sampler.sample(0.50, 0.25), RGBA::new(127, 127, 0, 255), 2);
        assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(0, 255, 0, 255), 2);
        assert_rgba_eq!(sampler.sample(0.00, 0.50), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.25, 0.50), RGBA::new(127, 0, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.75, 0.50), RGBA::new(127, 255, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(1.00, 0.50), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.00, 0.75), RGBA::new(127, 127, 255, 255), 2);
        assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(0, 0, 255, 255), 2);
        assert_rgba_eq!(sampler.sample(0.50, 0.75), RGBA::new(127, 127, 255, 255), 2);
        assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(255, 255, 255, 255), 2);
        assert_rgba_eq!(sampler.sample(1.00, 0.75), RGBA::new(127, 127, 255, 255), 2);
        assert_rgba_eq!(sampler.sample(0.00, 1.00), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.25, 1.00), RGBA::new(127, 0, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.50, 1.00), RGBA::new(127, 127, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(0.75, 1.00), RGBA::new(127, 255, 127, 255), 2);
        assert_rgba_eq!(sampler.sample(1.00, 1.00), RGBA::new(127, 127, 127, 255), 2);
    }
}

use super::*;

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SamplerFilter {
    Nearest = 0,
    Bilinear = 1,
    DebugMip = 2,
    Trilinear = 3,
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
        let mips: u32 = texture.count;
        let lod_rounded: f32 = if lod > 0.0 { lod.round() } else { 0.0 };
        let lod_floored: f32 = if lod > 0.0 { lod.floor() } else { 0.0 };
        let lod_fract: f32 = if lod > 0.0 { lod - lod_floored } else { 0.0 };
        let lod_fract_level: usize = (lod_fract * TRILINEAR_FRACT_LEVELS as f32) as usize;
        debug_assert!(lod_fract_level < TRILINEAR_FRACT_LEVELS as usize);

        let mip0_index = match filtering {
            SamplerFilter::Nearest | SamplerFilter::Bilinear | SamplerFilter::DebugMip => {
                (lod_rounded as i32).clamp(0, mips as i32 - 1)
            }
            SamplerFilter::Trilinear => (lod_floored as i32).clamp(0, mips as i32 - 1),
        };
        let mip0 = &texture.mips[mip0_index as usize];
        let texels0 = unsafe { texture.texels.as_ptr().add(mip0.offset as usize) };
        let log2_size = mip0.width.trailing_zeros() as usize;
        let entry = match filtering {
            SamplerFilter::Nearest => &NEAREST_SAMPLER_TABLE[texture.format as usize][log2_size],
            SamplerFilter::Bilinear => &BILINEAR_SAMPLER_TABLE[texture.format as usize][log2_size],
            SamplerFilter::DebugMip => &DEBUG_SAMPLER_TABLE[texture.format as usize][log2_size],
            SamplerFilter::Trilinear => &TRILINEAR_SAMPLER_TABLE[texture.format as usize][log2_size][lod_fract_level],
        };
        let sample_function = entry.f;
        let uv_scale = SamplerUVScale { bias: entry.b, scale: entry.s };
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
    debug_assert!(u >= 0.0 && v >= 0.0);
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
        return RGBA::from_u32(unsafe { (texel as *const u32).read_unaligned() } | 0xFF000000);
    }
    if FORMAT == TextureFormat::RGBA as u8 {
        return RGBA::from_u32(unsafe { *(texel as *const u32) });
    }
    RGBA::new(0, 0, 0, 255)
}

fn sample_bilinear<const SIZE: u16, const FORMAT: u8>(texels: *const u8, u: f32, v: f32) -> RGBA {
    debug_assert!(u >= 0.0 && v >= 0.0);
    let bpp: usize = bytes_per_pixel_u8(FORMAT);
    let stride: usize = SIZE as usize * bpp;
    let itx: i32 = unsafe { u.to_int_unchecked() };
    let ity: i32 = unsafe { v.to_int_unchecked() };
    let tx: u32 = itx as u32;
    let ty: u32 = ity as u32;
    let wx1: u32 = tx & 255;
    let wx: u32 = 256 - wx1;
    let wy1: u32 = ty & 255;
    let wy: u32 = 256 - wy1;
    let wa: u32 = wx * wy;
    let wb: u32 = wx1 * wy;
    let wc: u32 = wx * wy1;
    let wd: u32 = wx1 * wy1;
    let x0: u32 = tx >> 8;
    let x1: u32 = x0 + 1;
    let y0: u32 = ty >> 8;
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

fn mip_size_sample<const SIZE: u16>(_texels: *const u8, _u: f32, _v: f32) -> RGBA {
    match SIZE {
        1 => RGBA::new(255, 0, 0, 255),       // red
        2 => RGBA::new(0, 255, 0, 255),       // green
        4 => RGBA::new(0, 0, 255, 255),       // blue
        8 => RGBA::new(255, 255, 0, 255),     // yellow
        16 => RGBA::new(255, 0, 255, 255),    // magenta
        32 => RGBA::new(0, 255, 255, 255),    // cyan
        64 => RGBA::new(255, 128, 0, 255),    // orange
        128 => RGBA::new(255, 192, 203, 255), // pink
        256 => RGBA::new(0, 128, 255, 255),   // sky blue
        512 => RGBA::new(128, 255, 0, 255),   // lime
        1024 => RGBA::new(128, 0, 255, 255),  // purple
        2048 => RGBA::new(0, 128, 128, 255),  // teal
        4096 => RGBA::new(255, 0, 128, 255),  // rose
        _ => RGBA::new(0, 0, 0, 255),
    }
}

const TRILINEAR_FRACT_LEVELS: u32 = 16;
const TRILINEAR_FRACT_LEVELS_LOG2: u32 = TRILINEAR_FRACT_LEVELS.ilog2();
fn sample_trilinear<const MIP0_SIZE: u16, const FORMAT: u8, const FRACT: u32>(
    mip0_texels: *const u8,
    u: f32,
    v: f32,
) -> RGBA {
    debug_assert!(u >= 0.0 && v >= 0.0);
    debug_assert!(MIP0_SIZE >= 2);

    // These are all compile-time constants, but the compiler doesn't allow to declare them as const
    let mip0_size: u16 = MIP0_SIZE;
    let mip1_size: u16 = MIP0_SIZE / 2;
    let bpp: usize = bytes_per_pixel_u8(FORMAT);
    let mip0_stride: usize = mip0_size as usize * bpp;
    let mip1_stride: usize = mip1_size as usize * bpp;
    let mip1_texels_offset: isize = (mip0_size as usize * mip0_size as usize * bpp) as isize;

    // Convert input into 24.8 fixed-point biased into positive territory, minus half texel, and scaled by the tex size
    let itx: i32 = unsafe { u.to_int_unchecked() };
    let ity: i32 = unsafe { v.to_int_unchecked() };

    // Extract coordinates, offsets and bilinears weights for the mip0 texels
    let mip0_tx: u32 = itx as u32;
    let mip0_ty: u32 = ity as u32;
    let mip0_wx1: u32 = mip0_tx & 255;
    let mip0_wx: u32 = 256 - mip0_wx1;
    let mip0_wy1: u32 = mip0_ty & 255;
    let mip0_wy: u32 = 256 - mip0_wy1;
    let mip0_wa: u32 = mip0_wx * mip0_wy;
    let mip0_wb: u32 = mip0_wx1 * mip0_wy;
    let mip0_wc: u32 = mip0_wx * mip0_wy1;
    let mip0_wd: u32 = mip0_wx1 * mip0_wy1;
    let mip0_x0: u32 = mip0_tx >> 8;
    let mip0_x1: u32 = mip0_x0 + 1;
    let mip0_y0: u32 = mip0_ty >> 8;
    let mip0_y1: u32 = mip0_y0 + 1;
    let mip0_tx0: u32 = mip0_x0 & (mip0_size as u32 - 1);
    let mip0_tx1: u32 = mip0_x1 & (mip0_size as u32 - 1);
    let mip0_ty0: u32 = mip0_y0 & (mip0_size as u32 - 1);
    let mip0_ty1: u32 = mip0_y1 & (mip0_size as u32 - 1);
    let mip0_offset_a: usize = mip0_ty0 as usize * mip0_stride + mip0_tx0 as usize * bpp;
    let mip0_offset_b: usize = mip0_ty0 as usize * mip0_stride + mip0_tx1 as usize * bpp;
    let mip0_offset_c: usize = mip0_ty1 as usize * mip0_stride + mip0_tx0 as usize * bpp;
    let mip0_offset_d: usize = mip0_ty1 as usize * mip0_stride + mip0_tx1 as usize * bpp;

    // Extract coordinates, offsets and bilinears weights for the mip1 texels
    let mip1_tx: u32 = (itx as u32 - 127) >> 1;
    let mip1_ty: u32 = (ity as u32 - 127) >> 1;
    let mip1_wx1: u32 = mip1_tx & 255;
    let mip1_wx: u32 = 256 - mip1_wx1;
    let mip1_wy1: u32 = mip1_ty & 255;
    let mip1_wy: u32 = 256 - mip1_wy1;
    let mip1_wa: u32 = mip1_wx * mip1_wy;
    let mip1_wb: u32 = mip1_wx1 * mip1_wy;
    let mip1_wc: u32 = mip1_wx * mip1_wy1;
    let mip1_wd: u32 = mip1_wx1 * mip1_wy1;
    let mip1_x0: u32 = mip1_tx >> 8;
    let mip1_x1: u32 = mip1_x0 + 1;
    let mip1_y0: u32 = mip1_ty >> 8;
    let mip1_y1: u32 = mip1_y0 + 1;
    let mip1_tx0: u32 = mip1_x0 & (mip1_size as u32 - 1);
    let mip1_tx1: u32 = mip1_x1 & (mip1_size as u32 - 1);
    let mip1_ty0: u32 = mip1_y0 & (mip1_size as u32 - 1);
    let mip1_ty1: u32 = mip1_y1 & (mip1_size as u32 - 1);
    let mip1_offset_a: usize = mip1_ty0 as usize * mip1_stride + mip1_tx0 as usize * bpp;
    let mip1_offset_b: usize = mip1_ty0 as usize * mip1_stride + mip1_tx1 as usize * bpp;
    let mip1_offset_c: usize = mip1_ty1 as usize * mip1_stride + mip1_tx0 as usize * bpp;
    let mip1_offset_d: usize = mip1_ty1 as usize * mip1_stride + mip1_tx1 as usize * bpp;
    let mip1_texels: *const u8 = unsafe { mip0_texels.offset(mip1_texels_offset) };
    if FORMAT == TextureFormat::Grayscale as u8 {
        // Fetch the texels
        let mip0_a: u8 = unsafe { *mip0_texels.add(mip0_offset_a) };
        let mip0_b: u8 = unsafe { *mip0_texels.add(mip0_offset_b) };
        let mip0_c: u8 = unsafe { *mip0_texels.add(mip0_offset_c) };
        let mip0_d: u8 = unsafe { *mip0_texels.add(mip0_offset_d) };
        let mip1_a: u8 = unsafe { *mip1_texels.add(mip1_offset_a) };
        let mip1_b: u8 = unsafe { *mip1_texels.add(mip1_offset_b) };
        let mip1_c: u8 = unsafe { *mip1_texels.add(mip1_offset_c) };
        let mip1_d: u8 = unsafe { *mip1_texels.add(mip1_offset_d) };

        // Perform the bilinear interpolations
        let mip0_abcd: u32 = (mip0_a as u32) * mip0_wa
            + (mip0_b as u32) * mip0_wb
            + (mip0_c as u32) * mip0_wc
            + (mip0_d as u32) * mip0_wd;
        let mip1_abcd: u32 = (mip1_a as u32) * mip1_wa
            + (mip1_b as u32) * mip1_wb
            + (mip1_c as u32) * mip1_wc
            + (mip1_d as u32) * mip1_wd;

        // Perform the linear interpolation
        let result: u8 = ((mip0_abcd * (TRILINEAR_FRACT_LEVELS - FRACT) + mip1_abcd * FRACT)
            >> (16 + TRILINEAR_FRACT_LEVELS_LOG2)) as u8;
        return RGBA::new(result, result, result, 255);
    }
    if FORMAT == TextureFormat::RGB as u8 {
        // Fetch the texels
        let mip0_a: u32 = unsafe { (mip0_texels.add(mip0_offset_a) as *const u32).read_unaligned() };
        let mip0_b: u32 = unsafe { (mip0_texels.add(mip0_offset_b) as *const u32).read_unaligned() };
        let mip0_c: u32 = unsafe { (mip0_texels.add(mip0_offset_c) as *const u32).read_unaligned() };
        let mip0_d: u32 = unsafe { (mip0_texels.add(mip0_offset_d) as *const u32).read_unaligned() };
        let mip1_a: u32 = unsafe { (mip1_texels.add(mip1_offset_a) as *const u32).read_unaligned() };
        let mip1_b: u32 = unsafe { (mip1_texels.add(mip1_offset_b) as *const u32).read_unaligned() };
        let mip1_c: u32 = unsafe { (mip1_texels.add(mip1_offset_c) as *const u32).read_unaligned() };
        let mip1_d: u32 = unsafe { (mip1_texels.add(mip1_offset_d) as *const u32).read_unaligned() };

        // Perform the bilinear interpolations
        let mip0_r: u32 = (mip0_a & 0xFF) * mip0_wa
            + (mip0_b & 0xFF) * mip0_wb
            + (mip0_c & 0xFF) * mip0_wc
            + (mip0_d & 0xFF) * mip0_wd;
        let mip0_g: u32 = ((mip0_a >> 8) & 0xFF) * mip0_wa
            + ((mip0_b >> 8) & 0xFF) * mip0_wb
            + ((mip0_c >> 8) & 0xFF) * mip0_wc
            + ((mip0_d >> 8) & 0xFF) * mip0_wd;
        let mip0_b: u32 = ((mip0_a >> 16) & 0xFF) * mip0_wa
            + ((mip0_b >> 16) & 0xFF) * mip0_wb
            + ((mip0_c >> 16) & 0xFF) * mip0_wc
            + ((mip0_d >> 16) & 0xFF) * mip0_wd;
        let mip1_r: u32 = (mip1_a & 0xFF) * mip1_wa
            + (mip1_b & 0xFF) * mip1_wb
            + (mip1_c & 0xFF) * mip1_wc
            + (mip1_d & 0xFF) * mip1_wd;
        let mip1_g: u32 = ((mip1_a >> 8) & 0xFF) * mip1_wa
            + ((mip1_b >> 8) & 0xFF) * mip1_wb
            + ((mip1_c >> 8) & 0xFF) * mip1_wc
            + ((mip1_d >> 8) & 0xFF) * mip1_wd;
        let mip1_b: u32 = ((mip1_a >> 16) & 0xFF) * mip1_wa
            + ((mip1_b >> 16) & 0xFF) * mip1_wb
            + ((mip1_c >> 16) & 0xFF) * mip1_wc
            + ((mip1_d >> 16) & 0xFF) * mip1_wd;

        // Perform the linear interpolations
        let r: u32 = mip0_r * (TRILINEAR_FRACT_LEVELS - FRACT) + mip1_r * FRACT;
        let g: u32 = mip0_g * (TRILINEAR_FRACT_LEVELS - FRACT) + mip1_g * FRACT;
        let b: u32 = mip0_b * (TRILINEAR_FRACT_LEVELS - FRACT) + mip1_b * FRACT;
        const SHIFT: u32 = 16 + TRILINEAR_FRACT_LEVELS_LOG2;
        return RGBA::new((r >> SHIFT) as u8, (g >> SHIFT) as u8, (b >> SHIFT) as u8, 255);
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

const MAX_LOG2_SIZE: usize = 10; // up to 1024
const FORMATS: usize = 3; // Grayscale, RGB, RGBA

#[derive(Debug, Copy, Clone)]
struct SamplerEntry {
    // Sampling function
    f: SampleFunction,

    // Coordinate bias: x' = (x + b) * s
    b: f32,

    // Coordinate scale: x' = (x + b) * s
    s: f32,
}

static NEAREST_SAMPLER_TABLE: [[SamplerEntry; MAX_LOG2_SIZE + 1]; FORMATS] = {
    let mut table = [[SamplerEntry { f: noop_sample, b: 0.0, s: 1.0 }; MAX_LOG2_SIZE + 1]; FORMATS];
    const TF_GRS: u8 = TextureFormat::Grayscale as u8;
    const TF_RGB: u8 = TextureFormat::RGB as u8;
    const TF_RGBA: u8 = TextureFormat::RGBA as u8;
    type SA = SamplerEntry;
    let grs = &mut table[TextureFormat::Grayscale as usize];
    grs[0] = SA { f: sample_nearest::<1, TF_GRS>, b: 10.0, s: 1.0 };
    grs[1] = SA { f: sample_nearest::<2, TF_GRS>, b: 10.0, s: 2.0 };
    grs[2] = SA { f: sample_nearest::<4, TF_GRS>, b: 10.0, s: 4.0 };
    grs[3] = SA { f: sample_nearest::<8, TF_GRS>, b: 10.0, s: 8.0 };
    grs[4] = SA { f: sample_nearest::<16, TF_GRS>, b: 10.0, s: 16.0 };
    grs[5] = SA { f: sample_nearest::<32, TF_GRS>, b: 10.0, s: 32.0 };
    grs[6] = SA { f: sample_nearest::<64, TF_GRS>, b: 10.0, s: 64.0 };
    grs[7] = SA { f: sample_nearest::<128, TF_GRS>, b: 10.0, s: 128.0 };
    grs[8] = SA { f: sample_nearest::<256, TF_GRS>, b: 10.0, s: 256.0 };
    grs[9] = SA { f: sample_nearest::<512, TF_GRS>, b: 10.0, s: 512.0 };
    grs[10] = SA { f: sample_nearest::<1024, TF_GRS>, b: 10.0, s: 1024.0 };
    let rgb = &mut table[TextureFormat::RGB as usize];
    rgb[0] = SA { f: sample_nearest::<1, TF_RGB>, b: 10.0, s: 1.0 };
    rgb[1] = SA { f: sample_nearest::<2, TF_RGB>, b: 10.0, s: 2.0 };
    rgb[2] = SA { f: sample_nearest::<4, TF_RGB>, b: 10.0, s: 4.0 };
    rgb[3] = SA { f: sample_nearest::<8, TF_RGB>, b: 10.0, s: 8.0 };
    rgb[4] = SA { f: sample_nearest::<16, TF_RGB>, b: 10.0, s: 16.0 };
    rgb[5] = SA { f: sample_nearest::<32, TF_RGB>, b: 10.0, s: 32.0 };
    rgb[6] = SA { f: sample_nearest::<64, TF_RGB>, b: 10.0, s: 64.0 };
    rgb[7] = SA { f: sample_nearest::<128, TF_RGB>, b: 10.0, s: 128.0 };
    rgb[8] = SA { f: sample_nearest::<256, TF_RGB>, b: 10.0, s: 256.0 };
    rgb[9] = SA { f: sample_nearest::<512, TF_RGB>, b: 10.0, s: 512.0 };
    rgb[10] = SA { f: sample_nearest::<1024, TF_RGB>, b: 10.0, s: 1024.0 };
    let rgba = &mut table[TextureFormat::RGBA as usize];
    rgba[0] = SA { f: sample_nearest::<1, TF_RGBA>, b: 10.0, s: 1.0 };
    rgba[1] = SA { f: sample_nearest::<2, TF_RGBA>, b: 10.0, s: 2.0 };
    rgba[2] = SA { f: sample_nearest::<4, TF_RGBA>, b: 10.0, s: 4.0 };
    rgba[3] = SA { f: sample_nearest::<8, TF_RGBA>, b: 10.0, s: 8.0 };
    rgba[4] = SA { f: sample_nearest::<16, TF_RGBA>, b: 10.0, s: 16.0 };
    rgba[5] = SA { f: sample_nearest::<32, TF_RGBA>, b: 10.0, s: 32.0 };
    rgba[6] = SA { f: sample_nearest::<64, TF_RGBA>, b: 10.0, s: 64.0 };
    rgba[7] = SA { f: sample_nearest::<128, TF_RGBA>, b: 10.0, s: 128.0 };
    rgba[8] = SA { f: sample_nearest::<256, TF_RGBA>, b: 10.0, s: 256.0 };
    rgba[9] = SA { f: sample_nearest::<512, TF_RGBA>, b: 10.0, s: 512.0 };
    rgba[10] = SA { f: sample_nearest::<1024, TF_RGBA>, b: 10.0, s: 1024.0 };
    table
};

static BILINEAR_SAMPLER_TABLE: [[SamplerEntry; MAX_LOG2_SIZE + 1]; FORMATS] = {
    let mut table = [[SamplerEntry { f: noop_sample, b: 0.0, s: 1.0 }; MAX_LOG2_SIZE + 1]; FORMATS];
    const GRAYSCALE: u8 = TextureFormat::Grayscale as u8;
    const RGB: u8 = TextureFormat::RGB as u8;
    type SA = SamplerEntry;
    let grs = &mut table[TextureFormat::Grayscale as usize];
    grs[0] = SA { f: sample_bilinear::<1, GRAYSCALE>, b: 10.0 - 127.0 / (1.0 * 256.0), s: 1.0 * 256.0 };
    grs[1] = SA { f: sample_bilinear::<2, GRAYSCALE>, b: 10.0 - 127.0 / (2.0 * 256.0), s: 2.0 * 256.0 };
    grs[2] = SA { f: sample_bilinear::<4, GRAYSCALE>, b: 10.0 - 127.0 / (4.0 * 256.0), s: 4.0 * 256.0 };
    grs[3] = SA { f: sample_bilinear::<8, GRAYSCALE>, b: 10.0 - 127.0 / (8.0 * 256.0), s: 8.0 * 256.0 };
    grs[4] = SA { f: sample_bilinear::<16, GRAYSCALE>, b: 10.0 - 127.0 / (16.0 * 256.0), s: 16.0 * 256.0 };
    grs[5] = SA { f: sample_bilinear::<32, GRAYSCALE>, b: 10.0 - 127.0 / (32.0 * 256.0), s: 32.0 * 256.0 };
    grs[6] = SA { f: sample_bilinear::<64, GRAYSCALE>, b: 10.0 - 127.0 / (64.0 * 256.0), s: 64.0 * 256.0 };
    grs[7] = SA { f: sample_bilinear::<128, GRAYSCALE>, b: 10.0 - 127.0 / (128.0 * 256.0), s: 128.0 * 256.0 };
    grs[8] = SA { f: sample_bilinear::<256, GRAYSCALE>, b: 10.0 - 127.0 / (256.0 * 256.0), s: 256.0 * 256.0 };
    grs[9] = SA { f: sample_bilinear::<512, GRAYSCALE>, b: 10.0 - 127.0 / (512.0 * 256.0), s: 512.0 * 256.0 };
    grs[10] = SA { f: sample_bilinear::<1024, GRAYSCALE>, b: 10.0 - 127.0 / (1024.0 * 256.0), s: 1024.0 * 256.0 };
    let rgb = &mut table[TextureFormat::RGB as usize];
    rgb[0] = SA { f: sample_bilinear::<1, RGB>, b: 10.0 - 127.0 / (1.0 * 256.0), s: 1.0 * 256.0 };
    rgb[1] = SA { f: sample_bilinear::<2, RGB>, b: 10.0 - 127.0 / (2.0 * 256.0), s: 2.0 * 256.0 };
    rgb[2] = SA { f: sample_bilinear::<4, RGB>, b: 10.0 - 127.0 / (4.0 * 256.0), s: 4.0 * 256.0 };
    rgb[3] = SA { f: sample_bilinear::<8, RGB>, b: 10.0 - 127.0 / (8.0 * 256.0), s: 8.0 * 256.0 };
    rgb[4] = SA { f: sample_bilinear::<16, RGB>, b: 10.0 - 127.0 / (16.0 * 256.0), s: 16.0 * 256.0 };
    rgb[5] = SA { f: sample_bilinear::<32, RGB>, b: 10.0 - 127.0 / (32.0 * 256.0), s: 32.0 * 256.0 };
    rgb[6] = SA { f: sample_bilinear::<64, RGB>, b: 10.0 - 127.0 / (64.0 * 256.0), s: 64.0 * 256.0 };
    rgb[7] = SA { f: sample_bilinear::<128, RGB>, b: 10.0 - 127.0 / (128.0 * 256.0), s: 128.0 * 256.0 };
    rgb[8] = SA { f: sample_bilinear::<256, RGB>, b: 10.0 - 127.0 / (256.0 * 256.0), s: 256.0 * 256.0 };
    rgb[9] = SA { f: sample_bilinear::<512, RGB>, b: 10.0 - 127.0 / (512.0 * 256.0), s: 512.0 * 256.0 };
    rgb[10] = SA { f: sample_bilinear::<1024, RGB>, b: 10.0 - 127.0 / (1024.0 * 256.0), s: 1024.0 * 256.0 };
    table
};

static DEBUG_SAMPLER_TABLE: [[SamplerEntry; MAX_LOG2_SIZE + 1]; FORMATS] = {
    let mut table = [[SamplerEntry { f: noop_sample, b: 0.0, s: 1.0 }; MAX_LOG2_SIZE + 1]; FORMATS];
    type SA = SamplerEntry;
    let grs = &mut table[TextureFormat::Grayscale as usize];
    grs[0] = SA { f: mip_size_sample::<1>, b: 0.0, s: 1.0 };
    grs[1] = SA { f: mip_size_sample::<2>, b: 0.0, s: 1.0 };
    grs[2] = SA { f: mip_size_sample::<4>, b: 0.0, s: 1.0 };
    grs[3] = SA { f: mip_size_sample::<8>, b: 0.0, s: 1.0 };
    grs[4] = SA { f: mip_size_sample::<16>, b: 0.0, s: 1.0 };
    grs[5] = SA { f: mip_size_sample::<32>, b: 0.0, s: 1.0 };
    grs[6] = SA { f: mip_size_sample::<64>, b: 0.0, s: 1.0 };
    grs[7] = SA { f: mip_size_sample::<128>, b: 0.0, s: 1.0 };
    grs[8] = SA { f: mip_size_sample::<256>, b: 0.0, s: 1.0 };
    grs[9] = SA { f: mip_size_sample::<512>, b: 0.0, s: 1.0 };
    grs[10] = SA { f: mip_size_sample::<1024>, b: 0.0, s: 1.0 };
    table[TextureFormat::RGB as usize] = table[TextureFormat::Grayscale as usize];
    table
};

// Helper to repeat over fract levels
macro_rules! for_each_fract {
    ($mac:ident, $arr:ident[$idx:expr], $size:expr, $format:expr) => {
        $mac!($arr[$idx], $size, $format, 0);
        $mac!($arr[$idx], $size, $format, 1);
        $mac!($arr[$idx], $size, $format, 2);
        $mac!($arr[$idx], $size, $format, 3);
        $mac!($arr[$idx], $size, $format, 4);
        $mac!($arr[$idx], $size, $format, 5);
        $mac!($arr[$idx], $size, $format, 6);
        $mac!($arr[$idx], $size, $format, 7);
        $mac!($arr[$idx], $size, $format, 8);
        $mac!($arr[$idx], $size, $format, 9);
        $mac!($arr[$idx], $size, $format, 10);
        $mac!($arr[$idx], $size, $format, 11);
        $mac!($arr[$idx], $size, $format, 12);
        $mac!($arr[$idx], $size, $format, 13);
        $mac!($arr[$idx], $size, $format, 14);
        $mac!($arr[$idx], $size, $format, 15);
    };
}

// Specialization of fill_trilinear_entries for each FRACT value
macro_rules! fill_trilinear_entry {
    ($arr:ident[$idx:expr], $size:expr, $format:expr, $fract:expr) => {
        $arr[$idx][$fract] = SA {
            f: sample_trilinear::<$size, $format, $fract>,
            b: 10.0 - 127.0 / ($size as f32 * 256.0),
            s: $size as f32 * 256.0,
        };
    };
}

static TRILINEAR_SAMPLER_TABLE: [[[SamplerEntry; TRILINEAR_FRACT_LEVELS as usize]; MAX_LOG2_SIZE + 1]; FORMATS] = {
    let mut table = [[[SamplerEntry { f: noop_sample, b: 0.0, s: 1.0 }; TRILINEAR_FRACT_LEVELS as usize];
        MAX_LOG2_SIZE + 1]; FORMATS];
    const GRAYSCALE: u8 = TextureFormat::Grayscale as u8;
    const RGB: u8 = TextureFormat::RGB as u8;
    type SA = SamplerEntry;
    let grs = &mut table[GRAYSCALE as usize];

    // Sometimes Rust is really obnoxious: "cannot use `for` loop on `std::ops::Range<usize>` in statics"
    let mut i: usize = 0;
    while i < 16 {
        grs[0][i] = SA { f: sample_nearest::<1, GRAYSCALE>, b: 10.0, s: 1.0 };
        i += 1
    }
    for_each_fract!(fill_trilinear_entry, grs[1], 2, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[2], 4, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[3], 8, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[4], 16, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[5], 32, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[6], 64, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[7], 128, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[8], 256, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[9], 512, GRAYSCALE);
    for_each_fract!(fill_trilinear_entry, grs[10], 1024, GRAYSCALE);

    let rgb = &mut table[RGB as usize];
    i = 0;
    while i < 16 {
        rgb[0][i] = SA { f: sample_nearest::<1, RGB>, b: 10.0, s: 1.0 };
        i += 1
    }
    for_each_fract!(fill_trilinear_entry, rgb[1], 2, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[2], 4, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[3], 8, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[4], 16, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[5], 32, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[6], 64, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[7], 128, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[8], 256, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[9], 512, RGB);
    for_each_fract!(fill_trilinear_entry, rgb[10], 1024, RGB);

    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{TextureFormat, TextureSource};
    use std::sync::Arc;

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

    #[test]
    fn test_sample_trilinear_from_2x2_grayscale_texture() {
        // Mip 0:
        // 1.0 0.0
        // 0.0 0.0
        // Mip 1:
        // 0.5
        let texels = vec![255u8, 0u8, 0u8, 0u8, 127u8, 0u8, 0u8, 0u8];
        let mut mips: [Mip; MAX_MIP_LEVELS] = Default::default();
        mips[0] = Mip { width: 2, height: 2, offset: 0 };
        mips[1] = Mip { width: 1, height: 1, offset: 4 };
        let texture = Arc::new(Texture { texels, count: 2, mips: mips, format: TextureFormat::Grayscale });
        let e: i16 = 3;
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.0);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(255, 255, 255, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(0, 0, 0, 255), e);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(0, 0, 0, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(0, 0, 0, 255), e);
            assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(64, 64, 64, 255), e);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.1);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(242, 242, 242, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(10, 10, 10, 255), e); // !
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(10, 10, 10, 255), e); // !
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(10, 10, 10, 255), e); // !
            assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(69, 69, 69, 255), e);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.5);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(192, 192, 192, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(64, 64, 64, 255), e);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(64, 64, 64, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(64, 64, 64, 255), e);
            assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(96, 96, 96, 255), e);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.9);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(140, 140, 140, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(114, 114, 114, 255), e);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(114, 114, 114, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(114, 114, 114, 255), e);
            assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(120, 120, 120, 255), e);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 1.0);
            assert_rgba_eq!(sampler.sample(0.25, 0.25), RGBA::new(127, 127, 127, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.25), RGBA::new(127, 127, 127, 255), e);
            assert_rgba_eq!(sampler.sample(0.25, 0.75), RGBA::new(127, 127, 127, 255), e);
            assert_rgba_eq!(sampler.sample(0.75, 0.75), RGBA::new(127, 127, 127, 255), e);
            assert_rgba_eq!(sampler.sample(0.50, 0.50), RGBA::new(127, 127, 127, 255), e);
        }
    }

    #[test]
    fn test_sample_trilinear_from_4x4_grayscale_texture() {
        // Mip 0:
        // 1.0 0.0 1.0 0.0
        // 0.0 1.0 0.0 1.0
        // 1.0 0.0 1.0 0.0
        // 0.0 1.0 0.0 1.0
        // Mip 1:
        // 1.0 0.0
        // 0.0 1.0
        // Mip 2:
        // 0.0
        let texels = vec![
            255u8, 0u8, 255u8, 0u8, 0u8, 255u8, 0u8, 255u8, 255u8, 0u8, 255u8, 0u8, 0u8, 255u8, 0u8, 255u8, //
            255u8, 0u8, 0u8, 255u8, //
            0u8, 0u8, 0u8, 0u8,
        ];
        let mut mips: [Mip; MAX_MIP_LEVELS] = Default::default();
        mips[0] = Mip { width: 4, height: 4, offset: 0 };
        mips[1] = Mip { width: 2, height: 2, offset: 16 };
        mips[1] = Mip { width: 1, height: 1, offset: 20 };
        let texture = Arc::new(Texture { texels, count: 3, mips: mips, format: TextureFormat::Grayscale });
        let e: i16 = 5;
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.0);
            assert_rgba_eq!(sampler.sample(0.125, 0.125), RGBA::new(255, 255, 255, 255), e);
            assert_rgba_eq!(sampler.sample(0.250, 0.125), RGBA::new(127, 127, 127, 255), e);
            assert_rgba_eq!(sampler.sample(0.375, 0.125), RGBA::new(0, 0, 0, 255), e);
        }
        {
            let sampler = Sampler::new(&texture, SamplerFilter::Trilinear, 0.9);
            assert_rgba_eq!(sampler.sample(0.000, 0.250), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.125, 0.250), RGBA::new(185, 185, 185, 255), e);
            assert_rgba_eq!(sampler.sample(0.250, 0.250), RGBA::new(242, 242, 242, 255), e);
            assert_rgba_eq!(sampler.sample(0.375, 0.250), RGBA::new(187, 187, 187, 255), e); // should be 203 !!
            assert_rgba_eq!(sampler.sample(0.500, 0.250), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.625, 0.250), RGBA::new(70, 70, 70, 255), e);
            assert_rgba_eq!(sampler.sample(0.750, 0.250), RGBA::new(13, 13, 13, 255), e);
            assert_rgba_eq!(sampler.sample(0.875, 0.250), RGBA::new(70, 70, 70, 255), e);
            assert_rgba_eq!(sampler.sample(1.000, 0.250), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.000, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.125, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.250, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.375, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.500, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.625, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.750, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.875, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(1.000, 0.500), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.000, 0.750), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.125, 0.750), RGBA::new(70, 70, 70, 255), e);
            assert_rgba_eq!(sampler.sample(0.250, 0.750), RGBA::new(13, 13, 13, 255), e);
            assert_rgba_eq!(sampler.sample(0.375, 0.750), RGBA::new(70, 70, 70, 255), e);
            assert_rgba_eq!(sampler.sample(0.500, 0.750), RGBA::new(126, 126, 126, 255), e);
            assert_rgba_eq!(sampler.sample(0.625, 0.750), RGBA::new(187, 187, 187, 255), e); // should be 203 !!
            assert_rgba_eq!(sampler.sample(0.750, 0.750), RGBA::new(242, 242, 242, 255), e);
            assert_rgba_eq!(sampler.sample(0.875, 0.750), RGBA::new(185, 185, 185, 255), e);
            assert_rgba_eq!(sampler.sample(1.000, 0.750), RGBA::new(126, 126, 126, 255), e);
        }
    }
}

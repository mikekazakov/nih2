use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Grayscale,
    RGB,
    RGBA,
}

pub struct TextureSource<'a> {
    pub texels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

const MAX_MIP_LEVELS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct Mip {
    pub width: u16,
    pub height: u16,
    pub offset: u32,
}

impl Default for Mip {
    fn default() -> Self {
        Self { width: 0, height: 0, offset: 0 }
    }
}

pub struct Texture {
    pub texels: Vec<u8>,
    pub count: u32,
    pub mips: [Mip; MAX_MIP_LEVELS],
    pub format: TextureFormat,
}

impl Texture {
    pub fn new(source: &TextureSource) -> Arc<Self> {
        let bpp = bytes_per_pixel(source.format);
        assert!(source.height > 0);
        assert!(source.width > 0);
        assert!(source.height.is_power_of_two());
        assert!(source.width.is_power_of_two());
        assert_eq!(source.height, source.width);
        assert_eq!(source.texels.len(), source.height as usize * source.width as usize * bpp);

        // Compute mip count
        let mut dim = source.width;
        let mut mip_count = 1;
        while dim > 1 && mip_count < MAX_MIP_LEVELS {
            dim >>= 1;
            mip_count += 1;
        }

        // Compute total memory required and mip infos
        let mut total_size = 0 as usize;
        let mut mips: [Mip; MAX_MIP_LEVELS] = Default::default();
        dim = source.width;
        for level in 0..mip_count {
            let mip_size = (dim * dim) as usize * bpp;
            mips[level] = Mip { width: dim as u16, height: dim as u16, offset: total_size as u32 };
            total_size += mip_size;
            dim >>= 1;
        }

        // Allocate texels
        let mut texel_data = vec![0u8; total_size];

        // Copy base level
        texel_data[..source.texels.len()].copy_from_slice(&source.texels);

        // Generate mip levels
        for level in 1..mip_count {
            let src_mip = mips[level - 1];
            let src = texel_data[src_mip.offset as usize
                ..src_mip.offset as usize + src_mip.width as usize * src_mip.height as usize * bpp]
                .to_vec();

            let dst_mip = &mut mips[level];

            let dst = &mut texel_data[dst_mip.offset as usize
                ..dst_mip.offset as usize + dst_mip.width as usize * dst_mip.height as usize * bpp];

            for y in 0..dst_mip.height as usize {
                for x in 0..dst_mip.width as usize {
                    let mut sum = vec![0u32; bpp];
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let src_x = x * 2 + dx;
                            let src_y = y * 2 + dy;
                            let p_offset = (src_y * src_mip.width as usize + src_x) * bpp;
                            for i in 0..bpp {
                                sum[i] += src[p_offset + i] as u32;
                            }
                        }
                    }

                    let dst_offset = (y * dst_mip.width as usize + x) * bpp;
                    for i in 0..bpp {
                        dst[dst_offset + i] = ((sum[i] + 2) / 4) as u8;
                    }
                }
            }
        }

        Arc::new(Texture { mips, count: mip_count as u32, format: source.format, texels: texel_data })
    }
}

fn bytes_per_pixel(fmt: TextureFormat) -> usize {
    match fmt {
        TextureFormat::RGBA => 4,
        TextureFormat::RGB => 3,
        TextureFormat::Grayscale => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_grayscale_1x1() {
        let texel = [42u8];
        let source = TextureSource { texels: &texel, width: 1, height: 1, format: TextureFormat::Grayscale };
        let texture = Texture::new(&source);
        assert_eq!(texture.count, 1);
        assert_eq!(texture.mips[0].width, 1);
        assert_eq!(texture.mips[0].height, 1);
        assert_eq!(texture.mips[0].offset, 0);
        assert_eq!(texture.texels, vec![42u8]);
    }

    #[test]
    fn bake_rgb_1x1() {
        let texel = [10u8, 20u8, 30u8];
        let source = TextureSource { texels: &texel, width: 1, height: 1, format: TextureFormat::RGB };
        let texture = Texture::new(&source);
        assert_eq!(texture.count, 1);
        assert_eq!(texture.mips[0].width, 1);
        assert_eq!(texture.mips[0].height, 1);
        assert_eq!(texture.mips[0].offset, 0);
        assert_eq!(texture.texels, vec![10u8, 20u8, 30u8]);
    }

    #[test]
    fn bake_grayscale_2x2() {
        let texels = [10u8, 20u8, 30u8, 40u8];
        let source = TextureSource { texels: &texels, width: 2, height: 2, format: TextureFormat::Grayscale };
        let texture = Texture::new(&source);
        assert_eq!(texture.count, 2);
        assert_eq!(texture.mips[0].width, 2);
        assert_eq!(texture.mips[0].height, 2);
        assert_eq!(texture.mips[0].offset, 0);
        assert_eq!(texture.mips[1].width, 1);
        assert_eq!(texture.mips[1].height, 1);
        assert_eq!(texture.mips[1].offset, 4);
        assert_eq!(texture.texels, vec![10u8, 20u8, 30u8, 40u8, 25u8]);
    }

    #[test]
    fn bake_rgb_2x2() {
        let texels = [10u8, 20u8, 30u8, 40u8, 50u8, 60u8, 70u8, 80u8, 90u8, 100u8, 110u8, 120u8];
        let source = TextureSource { texels: &texels, width: 2, height: 2, format: TextureFormat::RGB };
        let texture = Texture::new(&source);
        assert_eq!(texture.count, 2);
        assert_eq!(texture.mips[0].width, 2);
        assert_eq!(texture.mips[0].height, 2);
        assert_eq!(texture.mips[0].offset, 0);
        assert_eq!(texture.mips[1].width, 1);
        assert_eq!(texture.mips[1].height, 1);
        assert_eq!(texture.mips[1].offset, 12);
        // Average of the 4 texels per channel:
        // R = (10+40+70+100)/4 = 55
        // G = (20+50+80+110)/4 = 65
        // B = (30+60+90+120)/4 = 75
        let expected_texels =
            [10u8, 20u8, 30u8, 40u8, 50u8, 60u8, 70u8, 80u8, 90u8, 100u8, 110u8, 120u8, 55u8, 65u8, 75u8];
        assert_eq!(texture.texels, expected_texels);
    }

    #[test]
    fn bake_rgb_4x4() {
        let texels: Vec<u8> = (0u8..48u8).collect();
        let source = TextureSource { texels: &texels, width: 4, height: 4, format: TextureFormat::RGB };
        let texture = Texture::new(&source);
        assert_eq!(texture.count, 3);

        assert_eq!(texture.mips[0].width, 4);
        assert_eq!(texture.mips[0].height, 4);
        assert_eq!(texture.mips[0].offset, 0);
        assert_eq!(texture.texels[0..48], texels);

        assert_eq!(texture.mips[1].width, 2);
        assert_eq!(texture.mips[1].height, 2);
        assert_eq!(texture.mips[1].offset, 48);
        assert_eq!(texture.texels[48..60], [8, 9, 10, 14, 15, 16, 32, 33, 34, 38, 39, 40]);

        assert_eq!(texture.mips[2].width, 1);
        assert_eq!(texture.mips[2].height, 1);
        assert_eq!(texture.mips[2].offset, 60);
        assert_eq!(texture.texels[60..63], [23u8, 24u8, 25u8]);
    }
}

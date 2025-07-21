use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Zeroable, Pod)]
pub struct RGBA {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RGBA {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_u32(&self) -> u32 {
        (self.r as u32) | ((self.g as u32) << 8) | ((self.b as u32) << 16) | ((self.a as u32) << 24)
    }

    pub fn from_u32(packed: u32) -> Self {
        Self {
            r: (packed & 0xFF) as u8,
            g: ((packed >> 8) & 0xFF) as u8,
            b: ((packed >> 16) & 0xFF) as u8,
            a: ((packed >> 24) & 0xFF) as u8,
        }
    }
}

pub struct ColorBuffer {
    /// Width of usable pixels in the buffer
    pub width: usize,

    /// Height of usable pixels in the buffer
    pub height: usize,

    /// Number of pixels between the rows
    pub stride: usize,

    /// The actual pixels in the buffer
    pub pixels: Vec<u32>,
}

impl ColorBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let stride = width;
        let pixels = vec![0u32; stride * height];
        Self { width, height, stride, pixels }
    }

    pub fn at(&self, x: usize, y: usize) -> u32 {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        self.pixels[y * self.stride + x]
    }

    pub fn at_mut(&mut self, x: usize, y: usize) -> &mut u32 {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        &mut self.pixels[y * self.stride + x]
    }

    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.pixels)
    }

    pub fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.pixels)
    }

    pub fn as_u32_slice(&self) -> &[u32] {
        &self.pixels
    }

    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        &mut self.pixels
    }

    pub fn as_rgba_slice(&self) -> &[RGBA] {
        bytemuck::cast_slice(&self.pixels)
    }

    pub fn as_rgba_slice_mut(&mut self) -> &mut [RGBA] {
        bytemuck::cast_slice_mut(&mut self.pixels)
    }

    pub fn fill(&mut self, color: RGBA) {
        let raw = color.to_u32();
        for pixel in self.pixels.iter_mut() {
            *pixel = raw;
        }
    }
}

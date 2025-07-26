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
        // (self.r as u32) | ((self.g as u32) << 8) | ((self.b as u32) << 16) | ((self.a as u32) << 24)
        bytemuck::cast(*self)
    }

    pub fn from_u32(packed: u32) -> Self {
        // Self {
        //     r: (packed & 0xFF) as u8,
        //     g: ((packed >> 8) & 0xFF) as u8,
        //     b: ((packed >> 16) & 0xFF) as u8,
        //     a: ((packed >> 24) & 0xFF) as u8,
        // }
        bytemuck::cast(packed)
    }
}

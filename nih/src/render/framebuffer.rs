use super::*;

pub struct Framebuffer<'a> {
    pub color_buffer: Option<&'a mut Buffer<u32>>,
}

impl Framebuffer<'_> {
    pub fn width(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.width;
        }
        return 0;
    }

    pub fn height(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.height;
        }
        return 0;
    }
}

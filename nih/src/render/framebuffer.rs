use super::*;

pub struct Framebuffer<'a> {
    pub color_buffer: Option<&'a mut Buffer<u32>>,
    pub depth_buffer: Option<&'a mut Buffer<u16>>,
}

impl Default for Framebuffer<'_> {
    fn default() -> Self {
        Self { color_buffer: None, depth_buffer: None }
    }
}

impl Framebuffer<'_> {
    pub fn width(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.width;
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.width;
        }
        return 0;
    }

    pub fn height(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.height;
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.height;
        }
        return 0;
    }
}

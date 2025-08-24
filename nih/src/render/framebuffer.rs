use super::*;

pub struct Framebuffer<'a> {
    pub color_buffer: Option<&'a mut TiledBuffer<u32, 64, 64>>,
    pub depth_buffer: Option<&'a mut TiledBuffer<u16, 64, 64>>,
}

pub struct FramebufferTile {
    pub color_buffer: Option<TiledBufferTileMut<u32, 64, 64>>,
    pub depth_buffer: Option<TiledBufferTileMut<u16, 64, 64>>,
}

impl Default for Framebuffer<'_> {
    fn default() -> Self {
        Self { color_buffer: None, depth_buffer: None }
    }
}

impl Framebuffer<'_> {
    pub const TILE_WITH: u16 = 64;
    pub const TILE_HEIGHT: u16 = 64;

    pub fn width(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.width();
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.width();
        }
        return 0;
    }

    pub fn height(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.height();
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.height();
        }
        return 0;
    }

    pub fn tile(&mut self, x: u16, y: u16) -> FramebufferTile {
        FramebufferTile {
            color_buffer: if let Some(buffer) = self.color_buffer.as_mut() {
                Some(buffer.tile_mut(x, y))
            } else {
                None
            },
            depth_buffer: if let Some(buffer) = self.depth_buffer.as_mut() {
                Some(buffer.tile_mut(x, y))
            } else {
                None
            },
        }
    }
}

impl FramebufferTile {
    pub const TILE_WITH: u16 = 64;
    pub const TILE_HEIGHT: u16 = 64;

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

    pub fn origin_x(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.origin_x;
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.origin_x;
        }
        return 0;
    }

    pub fn origin_y(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.origin_y;
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.origin_y;
        }
        return 0;
    }
}

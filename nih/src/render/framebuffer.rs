use super::*;

pub struct Framebuffer<'a> {
    pub color_buffer: Option<&'a mut TiledBuffer<u32, 64, 64>>,
    pub depth_buffer: Option<&'a mut TiledBuffer<u16, 64, 64>>,

    // NB! Normals might be not normalized!
    pub normal_buffer: Option<&'a mut TiledBuffer<u32, 64, 64>>,
}

pub struct FramebufferTile {
    pub color_buffer: Option<TiledBufferTileMut<u32, 64, 64>>,
    pub depth_buffer: Option<TiledBufferTileMut<u16, 64, 64>>,
    pub normal_buffer: Option<TiledBufferTileMut<u32, 64, 64>>,
}

impl Default for Framebuffer<'_> {
    fn default() -> Self {
        Self { color_buffer: None, depth_buffer: None, normal_buffer: None }
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

    pub fn tiles_x(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.tiles_x();
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.tiles_x();
        }
        return 0;
    }

    pub fn tiles_y(&self) -> u16 {
        if let Some(buffer) = &self.color_buffer {
            return buffer.tiles_y();
        }
        if let Some(buffer) = &self.depth_buffer {
            return buffer.tiles_y();
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
            normal_buffer: if let Some(buffer) = self.normal_buffer.as_mut() {
                Some(buffer.tile_mut(x, y))
            } else {
                None
            },
        }
    }

    pub fn for_each_tile_mut_parallel<F>(&mut self, f: F)
    where
        F: Fn(&mut FramebufferTile) + Send + Sync + 'static,
    {
        let tiles_x: u16 = self.tiles_x();
        let tiles_y: u16 = self.tiles_y();
        if tiles_x > 1 || tiles_y > 1 {
            let mut tiles: Vec<FramebufferTile> = Vec::<FramebufferTile>::new();
            for y in 0..tiles_y {
                for x in 0..tiles_x {
                    tiles.push(self.tile(x, y));
                }
            }
            use rayon::prelude::*;
            tiles.par_iter_mut().for_each(|tile| {
                f(tile);
            });
        } else {
            let mut tile: FramebufferTile = self.tile(0, 0);
            f(&mut tile);
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

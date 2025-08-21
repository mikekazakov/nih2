use bytemuck::{Pod, Zeroable};

pub struct Buffer<T> {
    /// Width of usable elements in the buffer
    pub width: u16,

    /// Height of usable elements in the buffer
    pub height: u16,

    /// Number of elements between the rows
    pub stride: u16,

    /// The actual elements in the buffer
    pub elems: Vec<T>,
}

pub struct BufferTile<'a, T> {
    /// X offset of the tile inside the buffer, in elements
    pub origin_x: u16,

    /// Y offset of the tile inside the buffer, in elements
    pub origin_y: u16,

    /// Width of the tile
    pub width: u16,

    /// Height of the tile
    pub height: u16,

    /// Number of elements between the rows
    pub stride: u16,

    /// The actual elements in the tile of the buffer
    pub data: &'a mut [T],
}

impl<T: Copy + Zeroable + Pod> Buffer<T> {
    pub fn new(width: u16, height: u16) -> Self {
        let stride = width;
        let elems = vec![T::zeroed(); (stride as usize) * (height as usize)];
        Self { width, height, stride, elems }
    }

    pub fn at(&self, x: u16, y: u16) -> T {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        self.elems[(y as usize) * (self.stride as usize) + (x as usize)]
    }

    pub fn at_mut(&mut self, x: u16, y: u16) -> &mut T {
        assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        &mut self.elems[(y as usize) * (self.stride as usize) + (x as usize)]
    }

    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.elems)
    }

    pub fn as_u8_slice_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.elems)
    }

    pub fn as_u32_slice(&self) -> &[T] {
        &self.elems
    }

    pub fn as_u32_slice_mut(&mut self) -> &mut [T] {
        &mut self.elems
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.elems
    }

    pub fn fill(&mut self, with: T) {
        // let raw = color.to_u32();
        for elem in self.elems.iter_mut() {
            *elem = with;
        }
    }

    pub fn split_into_tiles<'a>(&'a mut self, tile_width: u16, tile_height: u16) -> Vec<BufferTile<'a, T>> {
        assert!(tile_width > 0 && tile_height > 0);
        let mut tiles = Vec::new();

        let rows = (self.height + tile_height - 1) / tile_height;
        let cols = (self.width + tile_width - 1) / tile_width;

        for row in 0..rows {
            for col in 0..cols {
                let y = row * tile_height;
                let x = col * tile_width;

                let tile_ptr = self.elems.as_mut_ptr();
                let tile_data: &mut [T];
                unsafe {
                    // This builds a flat mutable slice covering all rows of the tile, with stride matching the parent buffer.
                    tile_data = std::slice::from_raw_parts_mut(
                        tile_ptr.add((y * self.stride + x) as usize),
                        (self.stride * tile_height) as usize,
                    );
                }
                tiles.push(BufferTile {
                    origin_x: x,
                    origin_y: y,
                    width: tile_width.min(self.width - x),
                    height: tile_height.min(self.height - y),
                    stride: self.stride,
                    data: tile_data,
                });
            }
        }

        tiles
    }
}

impl<'a, T> BufferTile<'a, T> {
    pub fn at(&self, x: u16, y: u16) -> &T {
        assert!(x < self.width && y < self.height);
        &self.data[(y as usize) * (self.stride as usize) + (x as usize)]
    }

    pub fn at_mut(&mut self, x: u16, y: u16) -> &mut T {
        assert!(x < self.width && y < self.height);
        &mut self.data[(y as usize) * (self.stride as usize) + (x as usize)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_mut_basic() {
        let mut buffer = Buffer::<u32>::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                *buffer.at_mut(x, y) = (y * 4 + x) as u32;
            }
        }

        {
            let mut tiles = buffer.split_into_tiles(2, 2);
            *tiles[0].at_mut(0, 0) = 42;
            assert_eq!(*tiles[0].at(0, 0), 42);
            *tiles[1].at_mut(0, 0) = 33;
            assert_eq!(*tiles[1].at(0, 0), 33);
        }
        assert_eq!(buffer.at(0, 0), 42);
        assert_eq!(buffer.at(2, 0), 33);
    }

    #[test]
    fn test_tile_mut_threads() {
        use rayon::prelude::*;
        let mut buffer = Buffer::<u32>::new(4, 4);
        let mut tiles = buffer.split_into_tiles(2, 2);
        tiles.par_iter_mut().for_each(|tile| {
            for y in 0..tile.height {
                for x in 0..tile.width {
                    *tile.at_mut(x, y) = 123;
                }
            }
        });
        assert_eq!(buffer.at(0, 0), 123);
        assert_eq!(buffer.at(3, 3), 123);
    }

    #[test]
    fn test_tile_clamping() {
        let mut buffer = Buffer::<u32>::new(4, 3);
        let tiles = buffer.split_into_tiles(5, 10);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].width, 4);
        assert_eq!(tiles[0].height, 3);
        assert_eq!(tiles[0].stride, 4);
    }
}

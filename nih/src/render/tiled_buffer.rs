use crate::render::Buffer;
use bytemuck::{Pod, Zeroable};

pub struct TiledBufferTile<T, const W: usize, const H: usize> {
    /// X offset of the tile inside the buffer, in elements
    pub origin_x: u16,

    /// Y offset of the tile inside the buffer, in elements
    pub origin_y: u16,

    /// Logical width of the tile, 0 < width <= W
    pub width: u16,

    /// Logical height of the tile, 0 < height <= H
    pub height: u16,

    /// Pointer to the first element of the tile
    pub ptr: *const T,
}

pub struct TiledBufferTileMut<T, const W: usize, const H: usize> {
    /// X offset of the tile inside the buffer, in elements
    pub origin_x: u16,

    /// Y offset of the tile inside the buffer, in elements
    pub origin_y: u16,

    /// Logical width of the tile, 0 < width <= W
    pub width: u16,

    /// Logical height of the tile, 0 < height <= H
    pub height: u16,

    /// Pointer to the first element of the tile
    pub ptr: *mut T,
}

const _: [(); 1] = [(); (size_of::<TiledBufferTile<u16, 64, 64>>() == 16) as usize];
const _: [(); 1] = [(); (size_of::<TiledBufferTileMut<u16, 64, 64>>() == 16) as usize];
unsafe impl<T, const W: usize, const H: usize> Send for TiledBufferTile<T, W, H> {}
unsafe impl<T, const W: usize, const H: usize> Sync for TiledBufferTile<T, W, H> {}
unsafe impl<T, const W: usize, const H: usize> Send for TiledBufferTileMut<T, W, H> {}
unsafe impl<T, const W: usize, const H: usize> Sync for TiledBufferTileMut<T, W, H> {}

impl<T: Copy + Clone, const W: usize, const H: usize> TiledBufferTile<T, W, H> {
    pub const WIDTH: usize = W;
    pub const HEIGHT: usize = H;
    pub const STRIDE: usize = W * std::mem::size_of::<T>();

    /// Returns the element at (x, y) with bounds checking.
    /// Panics if (x, y) is out of the tile’s logical bounds.
    pub fn get(&self, x: usize, y: usize) -> T {
        if x >= self.width as usize || y >= self.height as usize {
            panic!(
                "TiledBufferTile index out of bounds: ({}, {}) not in (0..{}, 0..{})",
                x, y, self.width, self.height
            );
        }

        // safe because bounds were checked
        unsafe { *self.ptr.add(y * W + x) }
    }

    /// Returns the element at (x, y) without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure that (x, y) is within the bounds of the tile,
    /// i.e., 0 <= x < self.width and 0 <= y < self.height.
    /// Calling this method with out-of-bounds coordinates is undefined behavior.
    pub fn get_unchecked(&self, x: usize, y: usize) -> T {
        debug_assert!(x < self.width as usize && y < self.height as usize);
        unsafe { *self.ptr.add(y * W + x) }
    }
}

impl<T, const W: usize, const H: usize> TiledBufferTileMut<T, W, H>
where
    T: Copy,
{
    pub const WIDTH: usize = W;
    pub const HEIGHT: usize = H;
    pub const STRIDE: usize = W * std::mem::size_of::<T>();

    /// Returns a value of the element at (x, y) with bounds checking.
    /// Panics if (x, y) is out of the tile’s logical bounds.
    pub fn at(&self, x: usize, y: usize) -> T {
        if x >= self.width as usize || y >= self.height as usize {
            panic!(
                "TiledBufferTile index out of bounds: ({}, {}) not in (0..{}, 0..{})",
                x, y, self.width, self.height
            );
        }

        // safe because bounds were checked
        unsafe { *self.ptr.add(y * W + x) }
    }

    /// Returns a value of the element at (x, y) without bounds checking.
    pub fn at_unchecked(&self, x: usize, y: usize) -> T {
        debug_assert!(x < self.width as usize && y < self.height as usize);
        unsafe { *self.ptr.add(y * W + x) }
    }

    /// Returns a mutable reference to the element at (x, y) with bounds checking.
    /// Panics if (x, y) is out of the tile’s logical bounds.
    pub fn get(&mut self, x: usize, y: usize) -> &mut T {
        if x >= self.width as usize || y >= self.height as usize {
            panic!(
                "TiledBufferTile index out of bounds: ({}, {}) not in (0..{}, 0..{})",
                x, y, self.width, self.height
            );
        }

        // safe because bounds were checked
        unsafe { &mut *self.ptr.add(y * W + x) }
    }

    /// Returns a mutable reference to the element at (x, y) without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure that (x, y) is within the bounds of the tile,
    /// i.e., 0 <= x < self.width and 0 <= y < self.height.
    /// Calling this method with out-of-bounds coordinates is undefined behavior.
    pub fn get_unchecked(&self, x: usize, y: usize) -> &mut T {
        debug_assert!(x < self.width as usize && y < self.height as usize);
        unsafe { &mut *self.ptr.add(y * W + x) }
    }
}

// impl<'a, T, const W: usize, const H: usize> std::ops::Index<(usize, usize)> for TiledBufferTile<'a, T, W, H> {
//     type Output = T;
//     fn index(&self, (x, y): (usize, usize)) -> T {
//         self.get_unchecked(x, y)
//     }
// }

// impl<'a, T, const W: usize, const H: usize> std::ops::Index<(usize, usize)> for TiledBufferTileMut<'a, T, W, H> {
//     type Output = T;
//     fn index(&self, (x, y): (usize, usize)) -> &T {
//         self.get_unchecked(x, y)
//     }
// }
//
// impl<'a, T, const W: usize, const H: usize> std::ops::IndexMut<(usize, usize)> for TiledBufferTileMut<'a, T, W, H> {
//     fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
//         self.get_unchecked(x, y)
//     }
// }

// pub struct TiledBufferTile<'a, T, const W: usize, const H: usize> {
//     /// X offset of the tile inside the buffer, in elements
//     pub origin_x: u16,
//
//     /// Y offset of the tile inside the buffer, in elements
//     pub origin_y: u16,
//
//     /// Logical width of the tile, 0 < width <= W
//     pub width: u16,
//
//     /// Logical height of the tile, 0 < height <= H
//     pub height: u16,
//
//     /// Pointer to the first element of the tile
//     pub ptr: *const T,
//
//     /// Marker for lifetime
//     _marker: std::marker::PhantomData<&'a T>,
// }

pub struct TiledBuffer<T, const W: usize, const H: usize> {
    /// Logical width of the buffer.
    width: u16,

    /// Logical height of the buffer.
    height: u16,

    /// Number of W*H tiles along X
    tiles_x: u16,

    /// Number of W*H titles along Y
    tiles_y: u16,

    /// The data itself. Row-major order of tiles inside.
    values: Vec<T>,
}

impl<T: Copy + Zeroable + Pod + Default, const W: usize, const H: usize> TiledBuffer<T, W, H> {
    pub const TILE_WIDTH: usize = W;
    pub const TILE_HEIGHT: usize = H;

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn tiles_x(&self) -> u16 {
        self.tiles_x
    }

    pub fn tiles_y(&self) -> u16 {
        self.tiles_y
    }

    pub fn new(width: u16, height: u16) -> Self {
        assert!(width > 0 && height > 0);
        let mut t = Self::default();
        let tiles_x = (width + W as u16 - 1) / W as u16;
        let tiles_y = (height + H as u16 - 1) / H as u16;
        let physical_width = tiles_x * W as u16;
        let physical_height = tiles_y * H as u16;
        t.width = width;
        t.height = height;
        t.tiles_x = tiles_x;
        t.tiles_y = tiles_y;
        t.values
            .resize_with(physical_width as usize * physical_height as usize, Default::default);
        t
    }

    pub fn fill(&mut self, value: T) {
        for v in self.values.iter_mut() {
            *v = value;
        }
    }

    pub fn at(&self, x: u16, y: u16) -> T {
        debug_assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        debug_assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        // self.elems[(y as usize) * (self.stride as usize) + (x as usize)]
        let tile_x = x / W as u16;
        let tile_y = y / H as u16;
        let tile = self.tile(tile_x, tile_y); // TODO: this is super-inefficient
        tile.get_unchecked(x as usize % W, y as usize % H)
    }

    pub fn at_mut(&mut self, x: u16, y: u16) -> &mut T {
        debug_assert!(x < self.width, "x out of bounds: {} >= {}", x, self.width);
        debug_assert!(y < self.height, "y out of bounds: {} >= {}", y, self.height);
        // &mut self.elems[(y as usize) * (self.stride as usize) + (x as usize)]
        let tile_x = x / W as u16;
        let tile_y = y / H as u16;
        let start_index = (tile_y as usize * self.tiles_x as usize + tile_x as usize) * (W * H);
        let offset = (y as usize % H) * W + (x as usize % W);
        &mut self.values[start_index + offset]
        // let tile = self.tile_mut(tile_x, tile_y); // TODO: this is super-inefficient
        // tile.get_unchecked(x as usize % W, y as usize % H)
    }

    pub fn tile(&self, tile_x: u16, tile_y: u16) -> TiledBufferTile<T, W, H> {
        assert!(tile_x < self.tiles_x && tile_y < self.tiles_y);
        let start_index = (tile_y as usize * self.tiles_x as usize + tile_x as usize) * (W * H);
        unsafe {
            TiledBufferTile {
                origin_x: tile_x * W as u16,
                origin_y: tile_y * H as u16,
                width: (self.width - tile_x * W as u16).min(W as u16),
                height: (self.height - tile_y * H as u16).min(H as u16),
                ptr: self.values.as_ptr().add(start_index),
                // _marker: std::marker::PhantomData,
            }
        }
    }

    pub fn tile_mut(&mut self, tile_x: u16, tile_y: u16) -> TiledBufferTileMut<T, W, H> {
        assert!(tile_x < self.tiles_x && tile_y < self.tiles_y);
        let start_index = (tile_y as usize * self.tiles_x as usize + tile_x as usize) * (W * H);
        unsafe {
            TiledBufferTileMut {
                origin_x: tile_x * W as u16,
                origin_y: tile_y * H as u16,
                width: (self.width - tile_x * W as u16).min(W as u16),
                height: (self.height - tile_y * H as u16).min(H as u16),
                ptr: self.values.as_mut_ptr().add(start_index),
                // _marker: std::marker::PhantomData,
            }
        }
    }

    pub fn as_flat_buffer(&self) -> Buffer<T> {
        let mut buffer = Buffer::<T>::new(self.width, self.height);

        // Fast path: write row chunks directly into the flat buffer.
        // Assumes Buffer<T> exposes a contiguous mutable slice.
        let dst = buffer.as_mut_slice();

        let width = self.width as usize;
        let height = self.height as usize;
        let tiles_x = self.tiles_x as usize;
        let tiles_y = self.tiles_y as usize;

        for ty in 0..tiles_y {
            // Number of logical rows present in this tile row (handles bottom edge)
            let rows_in_tile_row = std::cmp::min(H, height.saturating_sub(ty * H));

            for row in 0..rows_in_tile_row {
                let y = ty * H + row; // logical y
                let dst_row_start = y * width; // start of the destination row
                let mut dst_col = 0; // running x inside the destination row

                for tx in 0..tiles_x {
                    // Columns present in this tile (handles right edge)
                    let cols_in_tile = std::cmp::min(W, width.saturating_sub(tx * W));
                    if cols_in_tile == 0 {
                        break;
                    }

                    // Base of the (tx, ty) tile in `values`
                    let tile_base = (ty * tiles_x + tx) * (W * H);
                    // Start of the `row` within that tile
                    let src_row_start = tile_base + row * W;

                    let src = &self.values[src_row_start..src_row_start + cols_in_tile];
                    let dst_start = dst_row_start + dst_col;
                    let dst_end = dst_start + cols_in_tile;

                    dst[dst_start..dst_end].copy_from_slice(src);
                    dst_col += cols_in_tile;
                }
            }
        }

        buffer

        // let mut buffer = Buffer::<T>::new(self.width, self.height);
        // for y in 0..self.height {
        //     for x in 0..self.width {
        //         *buffer.at_mut(x, y) = self.at(x, y);
        //     }
        // }
        // buffer
    }
}

impl<T, const W: usize, const H: usize> Default for TiledBuffer<T, W, H> {
    fn default() -> Self {
        Self { width: 0, height: 0, tiles_x: 0, tiles_y: 0, values: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_basic() {
        // Create a 6x6 buffer with 4x4 tiles
        let mut buf = TiledBuffer::<u32, 4, 4>::new(6, 6);
        // Fill with sequential values
        for (i, v) in buf.values.iter_mut().enumerate() {
            *v = i as u32;
        }

        // Tile (0, 0)
        let tile00 = buf.tile(0, 0);
        assert_eq!(tile00.origin_x, 0);
        assert_eq!(tile00.origin_y, 0);
        assert_eq!(tile00.width, 4);
        assert_eq!(tile00.height, 4);
        // Check a few values
        assert_eq!(tile00.get(0, 0), 0);
        assert_eq!(tile00.get(3, 0), 3);
        assert_eq!(tile00.get(0, 3), 12);
        assert_eq!(tile00.get(3, 3), 15);

        // Tile (1, 0)
        let tile10 = buf.tile(1, 0);
        assert_eq!(tile10.origin_x, 4);
        assert_eq!(tile10.origin_y, 0);
        assert_eq!(tile10.width, 2);
        assert_eq!(tile10.height, 4);
        assert_eq!(tile10.get(0, 0), 16);
        assert_eq!(tile10.get(1, 0), 17);
        assert_eq!(tile10.get(0, 3), 28);
        assert_eq!(tile10.get(1, 3), 29);

        // Tile (0, 1)
        let tile01 = buf.tile(0, 1);
        assert_eq!(tile01.origin_x, 0);
        assert_eq!(tile01.origin_y, 4);
        assert_eq!(tile01.width, 4);
        assert_eq!(tile01.height, 2);
        assert_eq!(tile01.get(0, 0), 32);
        assert_eq!(tile01.get(3, 0), 35);
        assert_eq!(tile01.get(0, 1), 36);
        assert_eq!(tile01.get(3, 1), 39);
    }

    #[test]
    fn test_tile_bounds() {
        // Buffer 5x5, tile size 4x4
        let buf = TiledBuffer::<u32, 4, 4>::new(5, 5);
        let tile = buf.tile(1, 1);
        assert_eq!(tile.origin_x, 4);
        assert_eq!(tile.origin_y, 4);
        assert_eq!(tile.width, 1);
        assert_eq!(tile.height, 1);
    }
}

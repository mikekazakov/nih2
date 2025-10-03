#[derive(Clone, Copy, Debug)]
pub struct U32x4 {
    #[cfg(target_arch = "x86_64")]
    inner: core::arch::x86_64::__m128i,

    #[cfg(target_arch = "aarch64")]
    inner: core::arch::aarch64::uint32x4_t,
}

impl U32x4 {
    /// Construct from array
    #[inline(always)]
    pub fn load(values: [u32; 4]) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_loadu_si128(values.as_ptr() as *const __m128i) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vld1q_u32(values.as_ptr()) }
            }
        }
    }

    /// Store back into array
    #[inline(always)]
    pub fn store(self) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.inner);
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vst1q_u32(out.as_mut_ptr(), self.inner);
            }
        }

        out
    }

    /// Add two vectors
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_add_epi32(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vaddq_u32(self.inner, other.inner) }
            }
        }
    }

    /// Bitwise AND
    #[inline(always)]
    pub fn bitand(self, other: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                Self { inner: _mm_and_si128(self.inner, other.inner) }
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                Self { inner: vandq_u32(self.inner, other.inner) }
            }
        }
    }

    /// Check if any lane is nonzero
    #[inline(always)]
    pub fn any_nonzero(self) -> bool {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                // Test if all bits are zero
                _mm_testz_si128(self.inner, self.inner) == 0
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                vmaxvq_u32(self.inner) != 0
            }
        }
    }

    #[inline(always)]
    pub fn extract_lane0(self) -> u32 {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                _mm_cvtsi128_si32(v) as u32
            }

            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                // vgetq_lane_u32::<0>(self.inner) // lane index 0
                vgetq_lane_u32(self.inner, 0)
            }
        }
    }
}

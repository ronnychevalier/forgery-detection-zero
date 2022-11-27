#![warn(missing_docs)]

use std::f64::consts::{LN_10, PI};
use std::ops::{Index, IndexMut};
use std::sync::Mutex;

use bitvec::bitvec;

use image::{GenericImageView, ImageBuffer, Luma};

use itertools::Itertools;

use libm::lgamma;

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::convert::LuminanceImage;
use crate::{ForgedRegion, Grid};

/// The result of the vote to which grid a pixel is aligned with.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Vote {
    /// The pixel is aligned with a grid.
    AlignedWith(Grid),

    /// A vote is invalid when the pixel is within a 7 pixel wide region around the image border or when there is a tie.
    Invalid,
}

impl Vote {
    /// Converts a [`Vote`] to an [`Option`].
    ///
    /// It returns a grid if the vote is valid, `None` otherwise.
    pub fn grid(&self) -> Option<Grid> {
        match self {
            Vote::AlignedWith(grid) => Some(*grid),
            Vote::Invalid => None,
        }
    }
}

/// The grid origin vote map of an image.
///
/// Each pixel in the image may belong to one of the 64 overlapping grids.
/// This vote map contains the result of the vote for each pixel.
#[derive(Clone)]
pub struct Votes {
    votes: Box<[Vote]>,

    pub(crate) width: u32,
    pub(crate) height: u32,
    log_nt: f64,
}

impl Index<usize> for Votes {
    type Output = Vote;

    fn index(&self, index: usize) -> &Self::Output {
        &self.votes[index]
    }
}

impl Index<[u32; 2]> for Votes {
    type Output = Vote;

    fn index(&self, xy: [u32; 2]) -> &Self::Output {
        &self.votes[(xy[0] + xy[1] * self.width) as usize]
    }
}

impl IndexMut<usize> for Votes {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.votes[index]
    }
}

impl Votes {
    /// Computes the votes for the given luminance image
    pub(crate) fn from_luminance(image: &LuminanceImage) -> Self {
        struct State {
            zero: Vec<u32>,
            votes: Vec<Vote>,
        }
        let cosine = cosine_table();
        let zero = vec![0u32; (image.width() * image.height()) as usize];
        let votes = vec![Vote::Invalid; (image.width() * image.height()) as usize];

        let lock = Mutex::new(State { zero, votes });

        let iter = 0..image.height() - 7;
        #[cfg(feature = "rayon")]
        let iter = iter.into_par_iter();
        iter.for_each(|y| {
            for x in 0..image.width() - 7 {
                // SAFETY: The range of the loop makes `x` always less than `image.width() - 7` and `y` always less than `image.height() - 7`.
                let number_of_zeroes = unsafe { compute_number_of_zeros(&cosine, image, x, y) };
                // SAFETY: Same argument as above.
                let const_along = unsafe { is_const_along_x_or_y(image, x, y) };

                {
                    let mut state = lock.lock().unwrap();

                    // check all pixels in the block and update votes
                    for xx in x..x + 8 {
                        for yy in y..y + 8 {
                            let index = (xx + yy * image.width()) as usize;
                            // SAFETY: The loops iterate within the bounds of the image.
                            // `state.zero` and `state.votes` have the same size as the image.
                            // Thus, `index` is always within the bounds of the vec.
                            unsafe {
                                match number_of_zeroes.cmp(state.zero.get_unchecked(index)) {
                                    std::cmp::Ordering::Equal => {
                                        // if two grids are tied in number of zeros, do not vote
                                        *state.votes.get_unchecked_mut(index) = Vote::Invalid;
                                    }
                                    std::cmp::Ordering::Greater => {
                                        // update votes when the current grid has more zeros
                                        *state.zero.get_unchecked_mut(index) = number_of_zeroes;
                                        *state.votes.get_unchecked_mut(index) = if const_along {
                                            Vote::Invalid
                                        } else {
                                            Vote::AlignedWith(Grid::from_xy(x, y))
                                        };
                                    }
                                    std::cmp::Ordering::Less => (),
                                }
                            }
                        }
                    }
                }
            }
        });

        let mut votes = lock.into_inner().unwrap().votes;

        // set pixels on the border to non valid votes - only pixels that
        // belong to the 64 full 8x8 blocks inside the image can vote
        for xx in 0..image.width() {
            for yy in 0..7 {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = Vote::Invalid;
            }
            for yy in (image.height() - 7)..image.height() {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = Vote::Invalid;
            }
        }
        for yy in 0..image.height() {
            for xx in 0..7 {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = Vote::Invalid;
            }
            for xx in (image.width() - 7)..image.width() {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = Vote::Invalid;
            }
        }

        Self {
            votes: votes.into_boxed_slice(),
            width: image.width(),
            height: image.height(),
            log_nt: 2.0f64.mul_add(
                f64::from(image.height()).log10(),
                2.0f64.mul_add(64.0f64.log10(), 2.0 * f64::from(image.width()).log10()),
            ),
        }
    }

    pub(crate) fn detect_global_grids(&self) -> (Option<Grid>, [f64; 64]) {
        let mut grid_votes = [0u32; 64];
        let p = 1.0 / 64.0;

        // count votes per possible grid origin
        // and keep track of the grid with the maximum of votes
        let most_voted_grid = self.votes.iter().filter_map(Vote::grid).max_by_key(|grid| {
            let votes = &mut grid_votes[grid.0 as usize];
            *votes += 1;
            *votes
        });

        // compute the NFA value for all the significant grids.  votes are
        // correlated by irregular 8x8 blocks dividing by 64 gives a rough
        // count of the number of independent votes
        let n = self.width * self.height / 64;
        let lnfa_grids: [f64; 64] = std::array::from_fn(|i| {
            let k = grid_votes[i] / 64;

            log_nfa(n, k, p, self.log_nt)
        });

        // meaningful grid -> main grid found!
        if let Some(most_voted_grid) = most_voted_grid {
            if lnfa_grids[most_voted_grid.0 as usize] < 0.0 {
                return (Some(most_voted_grid), lnfa_grids);
            }
        }

        (None, lnfa_grids)
    }

    /// Detects zones which are inconsistent with a given grid
    pub(crate) fn detect_forgeries(
        &self,
        grid_to_exclude: Option<Grid>,
        grid_max: Grid,
    ) -> Box<[ForgedRegion]> {
        let p = 1.0 / 64.0;

        // Distance to look for neighbors in the region growing process.
        // A meaningful forgery must have a density of votes of at least
        // 1/64. Thus, its votes should not be in mean further away one
        // from another than a distance of 8. One could use a little
        // more to allow for some variation in the distribution.
        let w = 9;

        // minimal block size that can lead to a meaningful detection
        let min_size = (64.0 * self.log_nt / 64.0f64.log10()).ceil() as usize;

        let mut used = bitvec![0; self.votes.len()];

        let mut forged_regions = Vec::new();

        // region growing of zones that voted for other than the main grid
        for x in 0..self.width {
            for y in 0..self.height {
                let index = (x + y * self.width) as usize;
                if used[index] {
                    continue;
                }
                let grid = self[index].grid();
                if grid == grid_to_exclude {
                    continue;
                }
                let grid = if let Some(grid) = grid {
                    grid
                } else {
                    continue;
                };
                if grid.0 > grid_max.0 {
                    continue;
                }

                let mut x0 = x; /* region bounding box */
                let mut y0 = y;
                let mut x1 = x;
                let mut y1 = y;

                used.set(index, true);

                let mut regions_xy = vec![(x, y)];

                // iteratively add neighbor pixel of pixels in the region
                let mut i = 0;
                while i < regions_xy.len() {
                    let (reg_x, reg_y) = regions_xy[i];
                    for xx in reg_x.saturating_sub(w)..=reg_x.saturating_add(w).min(self.width - 1)
                    {
                        for yy in
                            reg_y.saturating_sub(w)..=reg_y.saturating_add(w).min(self.height - 1)
                        {
                            let index = (xx + yy * self.width) as usize;
                            if used[index] {
                                continue;
                            }
                            if self[index] != Vote::AlignedWith(grid) {
                                continue;
                            }

                            used.set(index, true);
                            regions_xy.push((xx, yy));
                            if xx < x0 {
                                x0 = xx;
                            }
                            if yy < y0 {
                                y0 = yy;
                            }
                            if xx > x1 {
                                x1 = xx;
                            }
                            if yy > y1 {
                                y1 = yy;
                            }
                        }
                    }
                    i += 1;
                }

                // compute the number of false alarms (NFA) for the regions with at least the minimal size
                if regions_xy.len() >= min_size {
                    let n = (x1 - x0 + 1) * (y1 - y0 + 1) / 64;
                    let k = regions_xy.len() / 64;
                    let lnfa = log_nfa(n, k as u32, p, self.log_nt);
                    if lnfa < 0.0 {
                        // meaningful different grid found
                        forged_regions.push(ForgedRegion {
                            start: (x0, y0),
                            end: (x1, y1),
                            grid,
                            lnfa,
                            regions_xy: regions_xy.into_boxed_slice(),
                        });
                    }
                }
            }
        }

        forged_regions.into_boxed_slice()
    }

    /// Transforms the votes into a luminance image.
    ///
    /// When there is a tie, the pixel is black.
    /// Otherwise, the pixel has a color between `0` and `63` to represent its vote.
    pub fn to_luma_image(&self) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        ImageBuffer::from_fn(self.width, self.height, |x, y| {
            let value = if let Vote::AlignedWith(grid) = self[[x, y]] {
                grid.0
            } else {
                255
            };

            Luma([value])
        })
    }
}

fn cosine_table() -> [[f64; 8]; 8] {
    let mut cosine = [[0.0; 8]; 8];

    (0..8).cartesian_product(0..8).for_each(|(k, l)| {
        cosine[k as usize][l as usize] =
            (2.0f64.mul_add(f64::from(k), 1.0) * f64::from(l) * PI / 16.0).cos();
    });

    cosine
}

/// Computes the logarithm of the number of false alarms (NFA) to base 10.
///
/// `NFA = NT.b(n,k,p)` the return value is `log10(NFA)`
///
/// - `n,k,p` - binomial parameters.
/// - `logNT` - logarithm of Number of Tests
fn log_nfa(n: u32, k: u32, p: f64, log_nt: f64) -> f64 {
    // an error of 10% in the result is accepted
    let tolerance = 0.1;
    let p_term = p / (1.0 - p);

    debug_assert!(k <= n && (0.0..=1.0).contains(&p));

    if n == 0 || k == 0 {
        return log_nt;
    } else if n == k {
        return f64::from(n).mul_add(p.log10(), log_nt);
    }

    let log1term =
        lgamma(f64::from(n) + 1.0) - lgamma(f64::from(k) + 1.0) - lgamma(f64::from(n - k) + 1.0)
            + f64::from(k) * p.ln()
            + f64::from(n - k) * (1.0 - p).ln();
    let mut term = log1term.exp();
    if term == 0.0 {
        if f64::from(k) > f64::from(n) * p {
            return log1term / LN_10 + log_nt;
        }

        return log_nt;
    }

    let mut bin_tail = term;
    for i in (k + 1)..=n {
        let bin_term = f64::from(n - i + 1) * 1.0 / f64::from(i);
        let mult_term = bin_term * p_term;

        term *= mult_term;
        bin_tail += term;

        if bin_term < 1.0 {
            // when bin_term<1 then mult_term_j<mult_term_i for j>i.
            // then, the error on the binomial tail when truncated at
            // the i term can be bounded by a geometric serie of form
            // term_i * sum mult_term_i^j.
            let err =
                term * ((1.0 - mult_term.powf(f64::from(n - i + 1))) / (1.0 - mult_term) - 1.0);

            // one wants an error at most of tolerance*final_result, or:
            // tolerance * abs(-log10(bin_tail)-logNT).
            // now, the error that can be accepted on bin_tail is
            // given by tolerance*final_result divided by the derivative
            // of -log10(x) when x=bin_tail. that is:
            // tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
            // finally, we truncate the tail if the error is less than:
            // tolerance * abs(-log10(bin_tail)-logNT) * bin_tail
            if err < tolerance * (-bin_tail.log10() - log_nt).abs() * bin_tail {
                break;
            }
        }
    }

    bin_tail.log10() + log_nt
}

/// Computes DCT for 8x8 blocks staring at x,y and count its zeros
///
/// # Safety
///
/// `x` must be less than `image.width() - 7` and `y` must be less than `image.height() - 7`.
unsafe fn compute_number_of_zeros(
    cosine: &[[f64; 8]; 8],
    image: &LuminanceImage,
    x: u32,
    y: u32,
) -> u32 {
    let vec = image.as_raw();
    (0..8)
        .cartesian_product(0..8)
        .filter(|(i, j)| *i > 0 || *j > 0)
        .map(|(i, j)| {
            let normalization = 0.25
                * (if i == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 })
                * (if j == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 });
            let dct_ij = (0..8)
                .cartesian_product(0..8)
                .map(|(xx, yy)| {
                    let index = (x + xx + (y + yy) * image.width()) as usize;
                    let pixel = vec.get_unchecked(index);
                    pixel * cosine[xx as usize][i as usize] * cosine[yy as usize][j as usize]
                })
                .sum::<f64>()
                * normalization;
            // the finest quantization in JPEG is to integer values.
            // in such case, the optimal threshold to decide if a
            // coefficient is zero or not is the midpoint between
            // 0 and 1, thus 0.5
            u32::from(dct_ij.abs() < 0.5)
        })
        .sum()
}

/// Checks whether the block is constant along x or y axis
///
/// # Safety
///
/// `x` must be less than `image.width() - 7` and `y` must be less than `image.height() - 7`.
unsafe fn is_const_along_x_or_y(image: &LuminanceImage, x: u32, y: u32) -> bool {
    let along_y = || {
        for yy in 0..8 {
            let v1 = image.unsafe_get_pixel(x, y + yy);
            for xx in 1..8 {
                let v2 = image.unsafe_get_pixel(x + xx, y + yy);
                if v1 != v2 {
                    return false;
                }
            }
        }
        true
    };
    let along_x = || {
        for xx in 0..8 {
            let v1 = image.unsafe_get_pixel(x + xx, y);
            for yy in 1..8 {
                let v2 = image.unsafe_get_pixel(x + xx, y + yy);
                if v1 != v2 {
                    return false;
                }
            }
        }
        true
    };
    along_x() || along_y()
}

#![doc = include_str!("../README.md")]

use std::f64::consts::{LN_10, PI};
use std::ops::{Index, IndexMut};
use std::sync::Mutex;

use bitvec::bitvec;
use bitvec::vec::BitVec;

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};

use itertools::Itertools;

use libm::lgamma;

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

mod convert;

use convert::{LuminanceImage, ToLumaZero};

/// Represents the errors that can be raised when using [`Zero`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The image and its 99% JPEG quality equivalent have different dimensions
    #[error("image and jpeg99 have different dimensions")]
    DifferentDimensions,
}

type Result<T> = std::result::Result<T, Error>;

/// A grid is an unsigned integer between `0` and `63`
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Grid(pub u8);

impl Grid {
    const fn from_xy(x: u32, y: u32) -> Self {
        Self(((x % 8) + (y % 8) * 8) as u8)
    }

    pub const fn x(&self) -> u8 {
        self.0 % 8
    }

    pub const fn y(&self) -> u8 {
        self.0 / 8
    }
}

/// An area of an image that has been forged.
#[derive(Default, Clone, Debug)]
pub struct ForgedRegion {
    /// Bottom left of the bounding box
    pub start: (u32, u32),

    /// Top right of the bounding box
    pub end: (u32, u32),

    pub grid: Grid,
    pub lnfa: f64,
    regions_xy: Box<[(u32, u32)]>,
}

pub struct ForeignGridAreas {
    votes: Votes,
    forged_regions: Box<[ForgedRegion]>,
    lnfa_grids: [f64; 64],
    main_grid: Option<Grid>,
}

impl ForeignGridAreas {
    pub fn votes(&self) -> &Votes {
        &self.votes
    }

    /// Builds a [forgery mask](`ForgeryMask`) that considers the pixels that are part of an area that have a grid different from the main one as forged
    pub fn build_forgery_mask(&self) -> ForgeryMask {
        ForgeryMask::from_regions(&self.forged_regions, self.votes.width, self.votes.height)
    }

    /// Get all the parts of the image where a JPEG grid is detected with its grid origin different from the main JPEG grid
    pub fn forged_regions(&self) -> &[ForgedRegion] {
        self.forged_regions.as_ref()
    }

    pub fn lnfa_grids(&self) -> [f64; 64] {
        self.lnfa_grids
    }

    pub fn main_grid(&self) -> Option<Grid> {
        self.main_grid
    }

    /// Whether it is likely that the image has been cropped.
    ///
    /// If the origin of the main grid is different from `(0, 0)`,
    /// it is likely that the image has been cropped.
    pub fn is_cropped(&self) -> bool {
        self.main_grid.map_or(false, |grid| grid.0 > 0)
    }

    /// Detects the areas of the image that are missing a grid.
    ///
    /// # Errors
    ///
    /// It returns an error if the given image does not have the same dimension as the original image.
    pub fn detect_missing_grid_areas(
        &self,
        jpeg_99: &DynamicImage,
    ) -> Result<Option<MissingGridAreas>> {
        let main_grid = if let Some(main_grid) = self.main_grid {
            main_grid
        } else {
            return Ok(None);
        };
        let jpeg_99 = jpeg_99.to_luma32f_zero();

        if (self.votes.width, self.votes.height) != jpeg_99.dimensions() {
            return Err(Error::DifferentDimensions);
        }

        let mut jpeg_99_votes = Votes::from_luminance(&jpeg_99);
        // update votemap by avoiding the votes for the main grid
        for x in 0..self.votes.width {
            for y in 0..self.votes.height {
                let index = (x + y * self.votes.width) as usize;
                if self.votes[index] == Some(main_grid) {
                    jpeg_99_votes[index] = None;
                }
            }
        }

        // Try to detect an imposed JPEG grid. No grid is to be excluded
        // and we are interested only in grid with origin (0,0), so:
        // grid_to_exclude = None and grid_max = 0
        let jpeg_99_forged_regions = jpeg_99_votes.detect_forgeries(None, Grid(0));

        Ok(Some(MissingGridAreas {
            votes: jpeg_99_votes,
            missing_regions: jpeg_99_forged_regions,
        }))
    }
}

/// Contains the result for the detection of missing grid areas
pub struct MissingGridAreas {
    votes: Votes,

    missing_regions: Box<[ForgedRegion]>,
}

impl MissingGridAreas {
    pub fn votes(&self) -> &Votes {
        &self.votes
    }

    /// Get all the parts of the image that have missing JPEG traces
    pub fn missing_regions(&self) -> &[ForgedRegion] {
        self.missing_regions.as_ref()
    }

    /// Builds a [forgery mask](`ForgeryMask`) that considers the pixels that are part of an area that have missing JPEG traces as forged
    pub fn build_forgery_mask(self) -> ForgeryMask {
        ForgeryMask::from_regions(&self.missing_regions, self.votes.width, self.votes.height)
    }
}

/// JPEG grid detector applied to forgery detection
pub struct Zero {
    luminance: LuminanceImage,
}

impl Zero {
    /// Initializes a forgery detection using the given image
    pub fn from_image(image: &DynamicImage) -> Self {
        let luminance = image.to_luma32f_zero();

        Self { luminance }
    }

    /// Runs the forgery detection algorithm
    pub fn detect_forgeries(self) -> ForeignGridAreas {
        let votes = Votes::from_luminance(&self.luminance);
        let (main_grid, lnfa_grids) = votes.detect_global_grids();
        let forged_regions = votes.detect_forgeries(main_grid, Grid(63));

        ForeignGridAreas {
            votes,
            forged_regions,
            lnfa_grids,
            main_grid,
        }
    }
}

/// The grid origin vote map of an image.
///
/// Each pixel in the image may belong to one of the 64 overlapping grids.
/// This vote map contains the result of the vote for each pixel.
#[derive(Clone)]
pub struct Votes {
    /// A vote is an unsigned integer between `0` and `63`
    votes: Box<[Option<Grid>]>,

    width: u32,
    height: u32,
    log_nt: f64,
}

impl Index<usize> for Votes {
    type Output = Option<Grid>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.votes[index]
    }
}

impl Index<[u32; 2]> for Votes {
    type Output = Option<Grid>;

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
    fn from_luminance(image: &LuminanceImage) -> Self {
        struct State {
            zero: Vec<u32>,
            votes: Vec<Option<Grid>>,
        }
        let cosine = cosine_table();
        let zero = vec![0u32; (image.width() * image.height()) as usize];
        let votes = vec![None; (image.width() * image.height()) as usize];

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
                                        *state.votes.get_unchecked_mut(index) = None;
                                    }
                                    std::cmp::Ordering::Greater => {
                                        // update votes when the current grid has more zeros
                                        *state.zero.get_unchecked_mut(index) = number_of_zeroes;
                                        *state.votes.get_unchecked_mut(index) = if const_along {
                                            None
                                        } else {
                                            Some(Grid::from_xy(x, y))
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
                votes[index] = None;
            }
            for yy in (image.height() - 7)..image.height() {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = None;
            }
        }
        for yy in 0..image.height() {
            for xx in 0..7 {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = None;
            }
            for xx in (image.width() - 7)..image.width() {
                let index = (xx + yy * image.width()) as usize;
                votes[index] = None;
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

    fn detect_global_grids(&self) -> (Option<Grid>, [f64; 64]) {
        let mut grid_votes = [0u32; 64];
        let p = 1.0 / 64.0;

        // count votes per possible grid origin
        // and keep track of the grid with the maximum of votes
        let most_voted_grid = self.votes.iter().flatten().max_by_key(|grid| {
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
                return (Some(*most_voted_grid), lnfa_grids);
            }
        }

        (None, lnfa_grids)
    }

    /// Detects zones which are inconsistent with a given grid
    fn detect_forgeries(
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
                if self[index] == grid_to_exclude {
                    continue;
                }
                let grid = if let Some(grid) = self[index] {
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
                            if self[index] != Some(grid) {
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
            let value = if let Some(value) = self[[x, y]] {
                value.0
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

/// A mask that represents the pixels of an image that have been forged
pub struct ForgeryMask {
    mask: BitVec,
    width: u32,
    height: u32,
}

impl ForgeryMask {
    /// Transforms the forgery mask into a luminance image.
    ///
    /// Each pixel considered forged is white, all the others are black.
    pub fn into_luma_image(self) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        ImageBuffer::from_fn(self.width, self.height, |x, y| {
            let index = (x + y * self.width) as usize;

            Luma([u8::from(self.mask[index]) * 255])
        })
    }

    /// Returns `true` if the pixel at `[x,y]` is considered forged
    pub fn is_forged(&self, x: u32, y: u32) -> bool {
        self.mask
            .get((x + y * self.width) as usize)
            .as_deref()
            .copied()
            .unwrap_or(false)
    }

    /// Returns the width of the forgery mask
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the forgery mask
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Builds a forgery mask for a set of forged regions.
    ///
    /// "Due to variations in the number of votes, the raw forgery mask contains holes.
    /// To give a more useful forgery map,
    /// these holes are filled by a mathematical morphology closing
    /// operator with a square structuring element of size W
    /// (the same as the neighborhood used in the region growing)."
    fn from_regions(regions: &[ForgedRegion], width: u32, height: u32) -> Self {
        let w = 9;
        let mut mask_aux = bitvec![0; width as usize * height as usize];
        let mut forgery_mask = bitvec![0; width as usize * height as usize];

        for forged in regions {
            for &(x, y) in forged.regions_xy.iter() {
                for xx in (x - w)..=(x + w) {
                    for yy in (y - w)..=(y + w) {
                        let index = (xx + yy * width) as usize;
                        mask_aux.set(index, true);
                        forgery_mask.set(index, true);
                    }
                }
            }
        }

        for x in w..width.saturating_sub(w) {
            for y in w..height.saturating_sub(w) {
                let index = (x + y * width) as usize;
                if !mask_aux[index] {
                    for xx in (x - w)..=(x + w) {
                        for yy in (y - w)..=(y + w) {
                            let index = (xx + yy * width) as usize;
                            forgery_mask.set(index, false);
                        }
                    }
                }
            }
        }

        Self {
            mask: forgery_mask,
            width,
            height,
        }
    }
}

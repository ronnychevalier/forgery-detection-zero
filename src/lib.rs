/// An attempt to reimplement [ZERO](https://github.com/tinankh/ZERO) in Rust.
///
/// At the moment, it is a C-like Rust implementation copied from the original C implementation.
/// It will be refactored later.
use std::f64::consts::{LN_10, PI};

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel};

use itertools::Itertools;

use libm::lgamma;

#[derive(Default, Clone, Copy)]
pub struct ForgedRegion {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub grid: i32,
    pub lnfa: f64,
}

fn cosine_table() -> [[f64; 8]; 8] {
    let mut cosine = [[0.0; 8]; 8];

    (0..8).cartesian_product(0..8).for_each(|(k, l)| {
        cosine[k as usize][l as usize] =
            (2.0f64.mul_add(f64::from(k), 1.0) * f64::from(l) * PI / 16.0).cos();
    });

    cosine
}

fn compute_grid_votes_per_pixel(image: &ImageBuffer<Luma<f64>, Vec<f64>>) -> Vec<i32> {
    let cosine = cosine_table();
    let mut zero = vec![0i32; (image.width() * image.height()) as usize];
    let mut votes = vec![-1i32; (image.width() * image.height()) as usize];

    for x in 0..image.width() - 7 {
        for y in 0..image.height() - 7 {
            let mut const_along_x = true;
            let mut const_along_y = true;

            // check whether the block is constant along x or y axis
            for xx in 0..8 {
                if !const_along_x && !const_along_y {
                    break;
                }
                for yy in 0..8 {
                    if !const_along_x && !const_along_y {
                        break;
                    }

                    if image.get_pixel(x + xx, y + yy) != image.get_pixel(x, y + yy) {
                        const_along_x = false;
                    }
                    if image.get_pixel(x + xx, y + yy) != image.get_pixel(x + xx, y) {
                        const_along_y = false;
                    }
                }
            }

            // compute DCT for 8x8 blocks staring at x,y and count its zeros
            let number_of_zeroes = (0..8)
                .cartesian_product(0..8)
                .filter(|(i, j)| *i > 0 || *j > 0)
                .map(|(i, j)| {
                    let mut dct_ij: f64 = 0.0;

                    for xx in 0..8 {
                        for yy in 0..8 {
                            let pixel = image.get_pixel(x + xx, y + yy).0[0];
                            dct_ij += pixel
                                * cosine[xx as usize][i as usize]
                                * cosine[yy as usize][j as usize];
                        }
                    }

                    dct_ij *= 0.25
                        * (if i == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 })
                        * (if j == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 });

                    // the finest quantization in JPEG is to integer values.
                    // in such case, the optimal threshold to decide if a
                    // coefficient is zero or not is the midpoint between
                    // 0 and 1, thus 0.5
                    i32::from(dct_ij.abs() < 0.5)
                })
                .sum();

            // check all pixels in the block and update votes
            for xx in x..x + 8 {
                for yy in y..y + 8 {
                    let index = (xx + yy * image.width()) as usize;
                    // if two grids are tied in number of zeros, do not vote
                    if number_of_zeroes == zero[index] {
                        votes[index] = -1;
                    } else if number_of_zeroes > zero[index] {
                        // update votes when the current grid has more zeros
                        zero[index] = number_of_zeroes;
                        votes[index] = if const_along_x || const_along_y {
                            -1
                        } else {
                            ((x % 8) + (y % 8) * 8) as i32
                        };
                    }
                }
            }
        }
    }

    // set pixels on the border to non valid votes - only pixels that
    // belong to the 64 full 8x8 blocks inside the image can vote
    for xx in 0..image.width() {
        for yy in 0..7 {
            let index = (xx + yy * image.width()) as usize;
            votes[index] = -1;
        }
        for yy in (image.height() - 7)..image.height() {
            let index = (xx + yy * image.width()) as usize;
            votes[index] = -1;
        }
    }
    for yy in 0..image.height() {
        for xx in 0..7 {
            let index = (xx + yy * image.width()) as usize;
            votes[index] = -1;
        }
        for xx in (image.width() - 7)..image.width() {
            let index = (xx + yy * image.width()) as usize;
            votes[index] = -1;
        }
    }

    votes
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

    assert!(k <= n && p >= 0.0 && p <= 1.0);

    if n == 0 || k == 0 {
        return log_nt;
    }

    if n == k {
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

fn detect_global_grids(votes: &[i32], width: u32, height: u32) -> (i32, [f64; 64]) {
    let log_nt =
        2.0 * 64.0f64.log10() + 2.0 * f64::from(width).log10() + 2.0 * f64::from(height).log10();
    let mut grid_votes = [0; 64];
    let mut max_votes = 0;
    let mut most_voted_grid = -1;
    let p = 1.0 / 64.0;

    // count votes per possible grid origin
    for x in 0..width {
        for y in 0..height {
            let index = (x + y * width) as usize;

            if votes[index] >= 0 && votes[index] < 64 {
                let grid = votes[index];
                grid_votes[grid as usize] += 1;

                // keep track of maximum of votes and the associated grid
                if grid_votes[grid as usize] > max_votes {
                    max_votes = grid_votes[grid as usize];
                    most_voted_grid = grid;
                }
            }
        }
    }

    // compute the NFA value for all the significant grids.  votes are
    // correlated by irregular 8x8 blocks dividing by 64 gives a rough
    // count of the number of independent votes
    let n = width * height / 64;
    let lnfa_grids: [f64; 64] = std::array::from_fn(|i| {
        let k = grid_votes[i] / 64;

        log_nfa(n, k, p, log_nt)
    });

    // meaningful grid -> main grid found!
    if most_voted_grid >= 0 && most_voted_grid < 64 && lnfa_grids[most_voted_grid as usize] < 0.0 {
        return (most_voted_grid, lnfa_grids);
    }

    (-1, lnfa_grids)
}

/// Detects zones which are inconsistent with a given grid
fn detect_forgeries(
    votes: &[i32],
    width: u32,
    height: u32,
    grid_to_exclude: i32,
    grid_max: i32,
) -> (Vec<ForgedRegion>, Vec<i32>) {
    let log_nt =
        2.0 * 64.0f64.log10() + 2.0 * f64::from(width).log10() + 2.0 * f64::from(height).log10();
    let p = 1.0 / 64.0;

    // Distance to look for neighbors in the region growing process.
    // A meaningful forgery must have a density of votes of at least
    // 1/64. Thus, its votes should not be in mean further away one
    // from another than a distance of 8. One could use a little
    // more to allow for some variation in the distribution.
    let w = 9;

    // minimal block size that can lead to a meaningful detection
    let min_size = (64.0 * log_nt / 64.0f64.log10()).ceil() as usize;

    let mut mask_aux = vec![0; (width * height) as usize];
    let mut used = vec![false; (width * height) as usize];

    let mut forgery_mask = vec![0; (width * height) as usize];
    let mut forgery_mask_reg = vec![0; (width * height) as usize];

    let mut forged_regions = Vec::new();

    // region growing of zones that voted for other than the main grid
    for x in 0..width {
        for y in 0..height {
            let index = (x + y * width) as usize;
            if used[index]
                || votes[index] == grid_to_exclude
                || votes[index] < 0
                || votes[index] > grid_max
            {
                continue;
            }

            let grid = votes[index];
            let mut x0 = x; /* region bounding box */
            let mut y0 = y;
            let mut x1 = x;
            let mut y1 = y;

            used[index] = true;

            let mut regions_xy = vec![(x, y)];

            // iteratively add neighbor pixel of pixels in the region
            let mut i = 0;
            while i < regions_xy.len() {
                let (reg_x, reg_y) = regions_xy[i];
                for xx in reg_x.saturating_sub(w)..=reg_x.saturating_add(w) {
                    for yy in reg_y.saturating_sub(w)..=reg_y.saturating_add(w) {
                        if xx < width && yy < height {
                            let index = (xx + yy * width) as usize;
                            if used[index] {
                                continue;
                            }
                            if votes[index] != grid {
                                continue;
                            }

                            used[index] = true;
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
                }
                i += 1;
            }

            // compute the number of false alarms (NFA) for the regions with at least the minimal size
            if regions_xy.len() >= min_size {
                let n = (x1 - x0 + 1) * (y1 - y0 + 1) / 64;
                let k = regions_xy.len() / 64;
                let lnfa = log_nfa(n, k as u32, p, log_nt);
                if lnfa < 0.0 {
                    // meaningful different grid found
                    forged_regions.push(ForgedRegion {
                        x0,
                        y0,
                        x1,
                        y1,
                        grid,
                        lnfa,
                    });

                    // mark points of the region in the forgery mask
                    for (reg_x, reg_y) in &regions_xy {
                        let index = reg_x + reg_y * width;
                        forgery_mask[index as usize] = 255;
                    }
                }
            }
        }
    }

    // regularized forgery mask by a morphologic closing operator
    for x in w..(width - w) {
        for y in w..(height - w) {
            let index = (x + y * width) as usize;
            if forgery_mask[index] != 0 {
                for xx in (x - w)..=(x + w) {
                    for yy in (y - w)..=(y + w) {
                        let index = (xx + yy * width) as usize;
                        mask_aux[index] = 255;
                        forgery_mask_reg[index] = 255;
                    }
                }
            }
        }
    }

    for x in w..(width - w) {
        for y in w..(height - w) {
            let index = (x + y * width) as usize;
            if mask_aux[index] == 0 {
                for xx in (x - w)..=(x + w) {
                    for yy in (y - w)..=(y + w) {
                        let index = (xx + yy * width) as usize;
                        forgery_mask_reg[index] = 0;
                    }
                }
            }
        }
    }

    (forged_regions, forgery_mask_reg)
}

pub fn zero(
    jpeg: &DynamicImage,
    jpeg_99: Option<&DynamicImage>,
) -> (
    i32,
    [f64; 64],
    Vec<ForgedRegion>,
    Vec<i32>,
    Vec<i32>,
    Option<(Vec<ForgedRegion>, Vec<i32>, Vec<i32>)>,
) {
    let luminance = jpeg.to_luma32f_zero();
    let votes = compute_grid_votes_per_pixel(&luminance);
    let (main_grid, lnfa_grids) = detect_global_grids(&votes, jpeg.width(), jpeg.height());
    let (forged_regions, forgery_mask) =
        detect_forgeries(&votes, jpeg.width(), jpeg.height(), main_grid, 63);

    if main_grid > -1 {
        if let Some(jpeg_99) = jpeg_99 {
            let jpeg_99_luminance = jpeg_99.to_luma32f_zero();
            let mut jpeg_99_votes = compute_grid_votes_per_pixel(&jpeg_99_luminance);
            // update votemap by avoiding the votes for the main grid
            for x in 0..jpeg.width() {
                for y in 0..jpeg.height() {
                    let index = (x + y * jpeg.width()) as usize;
                    if votes[index] == main_grid {
                        jpeg_99_votes[index] = -1;
                    }
                }
            }

            // Try to detect an imposed JPEG grid.  No grid is to be excluded
            // and we are interested only in grid with origin (0,0), so:
            // grid_to_exclude = -1 and grid_max = 0
            let (jpeg_99_forged_regions, jpeg_99_forgery_mask) =
                detect_forgeries(&jpeg_99_votes, jpeg_99.width(), jpeg_99.height(), -1, 0);

            return (
                main_grid,
                lnfa_grids,
                forged_regions,
                forgery_mask,
                votes,
                Some((jpeg_99_forged_regions, jpeg_99_forgery_mask, jpeg_99_votes)),
            );
        }
    }

    (
        main_grid,
        lnfa_grids,
        forged_regions,
        forgery_mask,
        votes,
        None,
    )
}

trait ToLumaZero {
    fn to_luma32f_zero(&self) -> ImageBuffer<Luma<f64>, Vec<f64>>;
}

#[inline]
fn rgb_to_luma_zero(rgb: &[u8]) -> f64 {
    let r = f64::from(rgb[0]);
    let g = f64::from(rgb[1]);
    let b = f64::from(rgb[2]);

    (b.mul_add(0.114, r.mul_add(0.299, g * 0.587))).round()
}

impl ToLumaZero for DynamicImage {
    fn to_luma32f_zero(&self) -> ImageBuffer<Luma<f64>, Vec<f64>> {
        let mut buffer: ImageBuffer<Luma<_>, Vec<_>> =
            ImageBuffer::new(self.width(), self.height());
        for (to, from) in buffer.pixels_mut().zip(self.pixels()) {
            let gray = to.channels_mut();
            let rgb = from.2.channels();
            gray[0] = rgb_to_luma_zero(rgb);
        }

        buffer
    }
}

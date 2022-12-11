#![doc = include_str!("../README.md")]
use bitvec::bitvec;
use bitvec::vec::BitVec;

#[cfg(feature = "image")]
use image::{DynamicImage, ImageBuffer, Luma};

#[cfg(feature = "image")]
mod convert;
mod vote;

#[cfg(feature = "image")]
use convert::ToLumaZero;

pub use vote::{Vote, Votes};

/// Represents the errors that can be raised when using [`Zero`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The dimensions given with the raw image are invalid (the size of the array is inconsistent with the width and height)
    #[error("inconsistency between the raw image array and the width and height provided")]
    InvalidRawDimensions,

    /// The JPEG encoding failed
    #[error("failed to encode the original image to a 99% quality JPEG: {0}")]
    Encoding(#[from] jpeg_encoder::EncodingError),

    /// The JPEG decoding failed
    #[cfg(feature = "image")]
    #[error("failed to decode the image: {0}")]
    Decoding(#[from] image::ImageError),
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
    luminance: LuminanceImage,
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
    /// It returns an error if it failed to encode the original image as 99% quality JPEG.
    #[cfg(feature = "image")]
    pub fn detect_missing_grid_areas(&self) -> Result<Option<MissingGridAreas>> {
        let main_grid = if let Some(main_grid) = self.main_grid {
            main_grid
        } else {
            return Ok(None);
        };

        let jpeg_99 = self.luminance.to_jpeg_99_luminance()?;

        let mut jpeg_99_votes = Votes::from_luminance(&jpeg_99);
        // update votemap by avoiding the votes for the main grid
        for (&vote, vote_99) in self.votes.iter().zip(jpeg_99_votes.iter_mut()) {
            if vote == Vote::AlignedWith(main_grid) {
                *vote_99 = Vote::Invalid;
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

impl IntoIterator for ForeignGridAreas {
    type Item = ForgedRegion;

    type IntoIter = std::vec::IntoIter<ForgedRegion>;

    fn into_iter(self) -> Self::IntoIter {
        self.forged_regions.into_vec().into_iter()
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
    pub fn forged_regions(&self) -> &[ForgedRegion] {
        self.missing_regions.as_ref()
    }

    /// Builds a [forgery mask](`ForgeryMask`) that considers the pixels that are part of an area that have missing JPEG traces as forged
    pub fn build_forgery_mask(self) -> ForgeryMask {
        ForgeryMask::from_regions(&self.missing_regions, self.votes.width, self.votes.height)
    }
}

impl IntoIterator for MissingGridAreas {
    type Item = ForgedRegion;

    type IntoIter = std::vec::IntoIter<ForgedRegion>;

    fn into_iter(self) -> Self::IntoIter {
        self.missing_regions.into_vec().into_iter()
    }
}

/// JPEG grid detector applied to forgery detection.
///
/// # Examples
///
/// An easy way to detect all regions of an image that have been forged:
///
/// ```no_run
/// # use forgery_detection_zero::Zero;
/// # let jpeg = todo!();
/// #
/// for r in Zero::from_image(&jpeg).into_iter() {
///     println!(
///         "Forged region detected: from ({}, {}) to ({}, {})",
///         r.start.0, r.start.1, r.end.0, r.end.1,
///     )
/// }
/// ```
pub struct Zero {
    luminance: LuminanceImage,
}

impl Zero {
    /// Initializes a forgery detection using the given image
    #[cfg(feature = "image")]
    pub fn from_image(image: &DynamicImage) -> Self {
        let luminance = image.to_luma32f_zero();

        Self { luminance }
    }

    /// Initializes a forgery detection using a raw luminance image
    ///
    /// # Errors
    ///
    /// It returns an error if the raw image array does not have a length consistent with the `width` and `height` parameters.
    pub fn from_luminance_raw(luminance: Box<[f64]>, width: u32, height: u32) -> Result<Self> {
        if luminance.len() != width.saturating_mul(height) as usize {
            return Err(Error::InvalidRawDimensions);
        }

        Ok(Self {
            luminance: LuminanceImage {
                image: luminance,
                width,
                height,
            },
        })
    }

    /// Runs the forgery detection algorithm.
    ///
    /// This is the more advanced API.
    /// If you just want to know the bounding box of each forged region in the image,
    /// you can call [`IntoIterator::into_iter`] instead.
    pub fn detect_forgeries(self) -> ForeignGridAreas {
        let votes = Votes::from_luminance(&self.luminance);
        let (main_grid, lnfa_grids) = votes.detect_global_grids();
        let forged_regions = votes.detect_forgeries(main_grid, Grid(63));

        ForeignGridAreas {
            luminance: self.luminance,
            votes,
            forged_regions,
            lnfa_grids,
            main_grid,
        }
    }
}

#[cfg(feature = "image")]
impl IntoIterator for Zero {
    type Item = ForgedRegion;

    type IntoIter = Box<dyn Iterator<Item = ForgedRegion>>;

    fn into_iter(self) -> Self::IntoIter {
        let foreign_grid_areas = self.detect_forgeries();
        let missing_grid_regions = foreign_grid_areas
            .detect_missing_grid_areas()
            .ok()
            .flatten()
            .into_iter()
            .flat_map(IntoIterator::into_iter);
        let forged_regions = foreign_grid_areas.into_iter().chain(missing_grid_regions);

        Box::new(forged_regions)
    }
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
    #[cfg(feature = "image")]
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

pub(crate) struct LuminanceImage {
    image: Box<[f64]>,
    width: u32,
    height: u32,
}

impl LuminanceImage {
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn as_raw(&self) -> &[f64] {
        &self.image
    }

    /// Gets a pixel without doing bounds checking
    ///
    /// # Safety
    ///
    /// `x` must be less than `image.width()` and `y` must be less than `image.height()`.
    pub(crate) unsafe fn unsafe_get_pixel(&self, x: u32, y: u32) -> &f64 {
        self.image.get_unchecked((x + y * self.width) as usize)
    }
}

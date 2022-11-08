use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use image::io::Reader as ImageReader;
use image::{ImageBuffer, ImageFormat};

use forgery_detection_zero::{Grid, Zero};

/// Detects JPEG grids and forgeries
#[derive(Parser)]
struct Arguments {
    /// Path to the image
    image: PathBuf,

    /// Path to the same image converted to a 99% quality JPEG
    jpeg_99: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();

    let reader = ImageReader::open(&args.image)
        .with_context(|| format!("Failed to open the image {}", args.image.display()))?;
    let jpeg = reader
        .decode()
        .with_context(|| format!("Failed to decode the image {}", args.image.display()))?;

    let jpeg_99 = args.jpeg_99.and_then(|jpeg_99| {
        let file = File::open(jpeg_99).ok()?;
        let reader = BufReader::new(file);

        ImageReader::with_format(reader, ImageFormat::Jpeg)
            .decode()
            .ok()
    });

    let mut global_grids = 0;

    let (forgeries, jpeg_99) = Zero::from_image(&jpeg)
        .with_missing_grids_detection(&jpeg_99)
        .context("The images should have the same dimension")?
        .detect_forgeries();
    if let Some(main_grid) = forgeries.main_grid() {
        println!(
            "main grid found: #{} ({},{}) log(nfa) = {}\n",
            main_grid.0,
            main_grid.x(),
            main_grid.y(),
            forgeries.lnfa_grids()[main_grid.0 as usize]
        );
        global_grids += 1;
    } else {
        println!("No overall JPEG grid found.");
    }

    for (i, &value) in forgeries.lnfa_grids().iter().enumerate() {
        if value < 0.0
            && forgeries
                .main_grid()
                .map_or(true, |grid| grid.0 as usize != i)
        {
            println!(
                "meaningful global grid found: #{i} ({},{}) log(nfa) = {value}\n",
                i % 8,
                i / 8
            );
            global_grids += 1;
        }
    }

    for forged_region in forgeries.forged_regions() {
        if forgeries.main_grid().is_some() {
            println!("\nA meaningful grid different from the main one was found here:");
        } else {
            println!("\nA meaningful grid was found here:");
        }
        print!(
            "bounding box: {} {} to {} {} [{}x{}]",
            forged_region.x0,
            forged_region.y0,
            forged_region.x1,
            forged_region.y1,
            forged_region.x1 - forged_region.x0 + 1,
            forged_region.y1 - forged_region.y0 + 1
        );
        print!(
            " grid: #{} ({},{})",
            forged_region.grid.0,
            forged_region.grid.x(),
            forged_region.grid.y(),
        );

        println!(" log(nfa) = {}", forged_region.lnfa);
    }

    let votes = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
        let value = if let Some(value) = forgeries.votes()[[x, y]] {
            value.0
        } else {
            255
        };
        image::Luma([value])
    });
    votes.save("votes.png")?;

    let forgery_mask = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
        let index = (x + y * jpeg.width()) as usize;

        image::Luma([forgeries.forgery_mask()[index] as u8])
    });
    forgery_mask.save("mask_f.png")?;

    if let Some(missing_grid_areas) = &jpeg_99 {
        if forgeries.main_grid().is_some() {
            for missing_region in missing_grid_areas.missing_regions() {
                println!("\nA region with missing JPEG grid was found here:");
                print!(
                    "bounding box: {} {} to {} {} [{}x{}]",
                    missing_region.x0,
                    missing_region.y0,
                    missing_region.x1,
                    missing_region.y1,
                    missing_region.x1 - missing_region.x0 + 1,
                    missing_region.y1 - missing_region.y0 + 1
                );
                print!(
                    " grid: #{} ({},{})",
                    missing_region.grid.0,
                    missing_region.grid.x(),
                    missing_region.grid.y(),
                );

                println!(" log(nfa) = {}", missing_region.lnfa);
            }
        }

        let votes = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
            let value = if let Some(value) = missing_grid_areas.votes()[[x, y]] {
                value.0
            } else {
                255
            };

            image::Luma([value])
        });
        votes.save("votes_jpeg.png")?;

        let forgery_mask = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
            let index = (x + y * jpeg.width()) as usize;

            image::Luma([missing_grid_areas.forgery_mask()[index] as u8])
        });
        forgery_mask.save("mask_m.png")?;
    }

    let number_of_regions = forgeries.forged_regions().len()
        + jpeg_99.map_or(0, |missing_grid_areas| {
            missing_grid_areas.missing_regions().len()
        });

    if number_of_regions == 0 && forgeries.main_grid().unwrap_or(Grid(0)).0 < 1 {
        println!("\nNo suspicious traces found in the image with the performed analysis.");
    }

    if forgeries.main_grid().unwrap_or(Grid(0)).0 > 0 {
        println!("\nThe most meaningful JPEG grid origin is not (0,0).");
        println!("This may indicate that the image has been cropped.");
    }

    if global_grids > 1 {
        println!("\nThere is more than one meaningful grid. This is suspicious.");
    }

    if number_of_regions > 0 {
        println!("\nSuspicious traces found in the image.");
        println!(
            "This may be caused by image manipulations such as resampling, copy-paste, splicing."
        );
        println!("Please examine the deviant meaningful region to make your own opinion about a potential forgery.");
    }

    Ok(())
}

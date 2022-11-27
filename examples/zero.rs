use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use image::io::Reader as ImageReader;
use image::ImageFormat;

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

    let forgeries = Zero::from_image(&jpeg).detect_forgeries();
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

    let jpeg_99 = jpeg_99
        .map(|jpeg_99| forgeries.detect_missing_grid_areas(&jpeg_99))
        .transpose()
        .context("The images should have the same dimension")?
        .flatten();

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
            forged_region.start.0,
            forged_region.start.1,
            forged_region.end.0,
            forged_region.end.1,
            forged_region.end.0 - forged_region.start.0 + 1,
            forged_region.end.1 - forged_region.start.1 + 1
        );
        print!(
            " grid: #{} ({},{})",
            forged_region.grid.0,
            forged_region.grid.x(),
            forged_region.grid.y(),
        );

        println!(" log(nfa) = {}", forged_region.lnfa);
    }

    let number_of_regions = forgeries.forged_regions().len()
        + jpeg_99.as_ref().map_or(0, |missing_grid_areas| {
            missing_grid_areas.missing_regions().len()
        });

    forgeries.votes().to_luma_image().save("votes.png")?;

    forgeries
        .build_forgery_mask()
        .into_luma_image()
        .save("mask_f.png")?;

    if let Some(missing_grid_areas) = jpeg_99 {
        if forgeries.main_grid().is_some() {
            for missing_region in missing_grid_areas.missing_regions() {
                println!("\nA region with missing JPEG grid was found here:");
                print!(
                    "bounding box: {} {} to {} {} [{}x{}]",
                    missing_region.start.0,
                    missing_region.start.1,
                    missing_region.end.0,
                    missing_region.end.1,
                    missing_region.end.0 - missing_region.start.0 + 1,
                    missing_region.end.1 - missing_region.start.1 + 1
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

        missing_grid_areas
            .votes()
            .to_luma_image()
            .save("votes_jpeg.png")?;

        missing_grid_areas
            .build_forgery_mask()
            .into_luma_image()
            .save("mask_m.png")?;
    }

    if number_of_regions == 0 && forgeries.main_grid().unwrap_or(Grid(0)).0 < 1 {
        println!("\nNo suspicious traces found in the image with the performed analysis.");
    }

    if forgeries.is_cropped() {
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

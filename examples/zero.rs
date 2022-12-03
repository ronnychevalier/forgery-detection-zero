use anyhow::Context;

use image::io::Reader as ImageReader;

use forgery_detection_zero::{Grid, Zero};

fn main() -> anyhow::Result<()> {
    let image_path = if let Some(image_path) = std::env::args().nth(1) {
        image_path
    } else {
        println!("Detects JPEG grids and forgeries\n");
        println!("Usage: zero <IMAGE_PATH>\n");

        anyhow::bail!("You need to provide a path to the image to analyze.");
    };

    let jpeg = ImageReader::open(&image_path)
        .with_context(|| format!("Failed to open the image {}", &image_path))?
        .decode()
        .with_context(|| format!("Failed to decode the image {}", &image_path))?;

    let mut global_grids = 0;

    let foreign_grid_areas = Zero::from_image(&jpeg).detect_forgeries();
    if let Some(main_grid) = foreign_grid_areas.main_grid() {
        println!(
            "main grid found: #{} ({},{}) log(nfa) = {}\n",
            main_grid.0,
            main_grid.x(),
            main_grid.y(),
            foreign_grid_areas.lnfa_grids()[main_grid.0 as usize]
        );
        global_grids += 1;
    } else {
        println!("No overall JPEG grid found.");
    }

    let missing_grid_areas = foreign_grid_areas
        .detect_missing_grid_areas()
        .context("Failed to detect the missing grid areas")?;

    for (i, &value) in foreign_grid_areas.lnfa_grids().iter().enumerate() {
        if value < 0.0
            && foreign_grid_areas
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

    for forged_region in foreign_grid_areas.forged_regions() {
        if foreign_grid_areas.main_grid().is_some() {
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

    let number_of_regions = foreign_grid_areas.forged_regions().len()
        + missing_grid_areas.as_ref().map_or(0, |missing_grid_areas| {
            missing_grid_areas.forged_regions().len()
        });

    foreign_grid_areas
        .votes()
        .to_luma_image()
        .save("votes.png")?;

    foreign_grid_areas
        .build_forgery_mask()
        .into_luma_image()
        .save("mask_f.png")?;

    if let Some(missing_grid_areas) = missing_grid_areas {
        if foreign_grid_areas.main_grid().is_some() {
            for missing_region in missing_grid_areas.forged_regions() {
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

    if number_of_regions == 0 && foreign_grid_areas.main_grid().unwrap_or(Grid(0)).0 < 1 {
        println!("\nNo suspicious traces found in the image with the performed analysis.");
    }

    if foreign_grid_areas.is_cropped() {
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

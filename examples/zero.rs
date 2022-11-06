use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;

use image::io::Reader as ImageReader;
use image::{ImageBuffer, ImageFormat};

use zero_rs::zero;

#[derive(Parser)]
struct Arguments {
    image: PathBuf,
    jpeg_99: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Arguments::parse();

    let reader = ImageReader::open(args.image)?;
    let jpeg = reader.decode()?;

    let jpeg_99 = args.jpeg_99.and_then(|jpeg_99| {
        let file = File::open(jpeg_99).ok()?;
        let reader = BufReader::new(file);

        ImageReader::with_format(reader, ImageFormat::Jpeg)
            .decode()
            .ok()
    });

    let mut global_grids = 0;

    let (main_grid, lnfa_grids, forged_regions, forgery_mask, votes, jpeg_99) =
        zero(&jpeg, jpeg_99.as_ref());

    if main_grid == -1 {
        println!("No overall JPEG grid found.");
    } else if main_grid > -1 {
        println!(
            "main grid found: #{main_grid} ({},{}) log(nfa) = {}\n",
            main_grid % 8,
            main_grid / 8,
            lnfa_grids[main_grid as usize]
        );
        global_grids += 1;
    }

    for (i, &value) in lnfa_grids.iter().enumerate() {
        if value < 0.0 && i != main_grid as usize {
            println!(
                "meaningful global grid found: #{i} ({},{}) log(nfa) = {value}\n",
                i % 8,
                i / 8
            );
            global_grids += 1;
        }
    }

    for forged_region in &forged_regions {
        if main_grid != -1 {
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
            forged_region.grid,
            forged_region.grid % 8,
            forged_region.grid / 8
        );
        println!(" log(nfa) = {}", forged_region.lnfa);
    }

    let votes = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
        let index = (x + y * jpeg.width()) as usize;

        image::Luma([votes[index] as u8])
    });
    votes.save("votes.png")?;

    let forgery_mask = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
        let index = (x + y * jpeg.width()) as usize;

        image::Luma([forgery_mask[index] as u8])
    });
    forgery_mask.save("mask_f.png")?;

    if let Some((missing_regions, jpeg_99_forgery_mask, jpeg_99_votes)) = &jpeg_99 {
        if main_grid > -1 {
            for missing_region in missing_regions {
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
                    missing_region.grid,
                    missing_region.grid % 8,
                    missing_region.grid / 8
                );
                println!(" log(nfa) = {}", missing_region.lnfa);
            }
        }

        let votes = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
            let index = (x + y * jpeg.width()) as usize;

            image::Luma([jpeg_99_votes[index] as u8])
        });
        votes.save("votes_jpeg.png")?;

        let forgery_mask = ImageBuffer::from_fn(jpeg.width(), jpeg.height(), |x, y| {
            let index = (x + y * jpeg.width()) as usize;

            image::Luma([jpeg_99_forgery_mask[index] as u8])
        });
        forgery_mask.save("mask_m.png")?;
    }

    let number_of_regions = forged_regions.len() + jpeg_99.map_or(0, |jpeg_99| jpeg_99.0.len());

    if number_of_regions == 0 && main_grid < 1 {
        println!("\nNo suspicious traces found in the image with the performed analysis.");
    }

    if main_grid > 0 {
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

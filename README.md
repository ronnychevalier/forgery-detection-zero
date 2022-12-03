# forgery-detection-zero

An implementation of [ZERO](https://doi.org/10.5201/ipol.2021.390): a JPEG grid detector applied to forgery detection in digital images.

The approach is described in the following paper:

```text
Tina Nikoukhah, Jérémy Anger, Miguel Colom, Jean-Michel Morel, and Rafael Grompone von Gioi,
ZERO: a Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and its Applications in Image Forensics,
Image Processing On Line, 11 (2021), pp. 396–433. https://doi.org/10.5201/ipol.2021.390
```

The original implementation is [written in C](https://github.com/tinankh/ZERO).

## Library example

Simple usage:

```rust,no_run
# use forgery_detection_zero::Zero;
# let jpeg = todo!();
#
for r in Zero::from_image(&jpeg).into_iter() {
    println!(
        "Forged region detected: from ({}, {}) to ({}, {})",
        r.start.0, r.start.1, r.end.0, r.end.1,
    )
}
```

More advanced usage:

```rust,no_run
# use forgery_detection_zero::Zero;
# let jpeg = todo!();
#
let foreign_grid_areas = Zero::from_image(&jpeg).detect_forgeries();
let missing_grid_areas = foreign_grid_areas
    .detect_missing_grid_areas()
    .unwrap()
    .unwrap();
let forged_regions = foreign_grid_areas
    .forged_regions()
    .iter()
    .chain(missing_grid_areas.forged_regions());
for r in forged_regions {
    println!(
        "Forged region detected: from ({}, {}) to ({}, {})",
        r.start.0, r.start.1, r.end.0, r.end.1,
    )
}
```

## CLI example

You can use the example to generate the forgery masks of an image:

```shell
cargo r --release --example zero image.jpg
```

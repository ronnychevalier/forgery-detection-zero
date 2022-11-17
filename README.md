# forgery-detection-zero

An implementation of [ZERO](https://doi.org/10.5201/ipol.2021.390): a JPEG grid detector applied to forgery detection in digital images.

The approach is described in the following paper:

```text
Tina Nikoukhah, Jérémy Anger, Miguel Colom, Jean-Michel Morel, and Rafael Grompone von Gioi,
ZERO: a Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and its Applications in Image Forensics,
Image Processing On Line, 11 (2021), pp. 396–433. https://doi.org/10.5201/ipol.2021.390
```

This library is based on the [original implementation written in C](https://github.com/tinankh/ZERO).

At the moment, it is a C-like Rust implementation very close to the original implementation.
It is in the process of being refactored to be more idiomatic.

## CLI example

Let's say you have an image called `image.jpg`.
First, convert this image to a 99% quality JPEG using imagemagick `convert` tool:

```shell
convert  -quality 99% image.jpg image_99.jpg
```

Then you can use the example to generate the forgery masks:

```shell
cargo r --release --example zero image.jpg image_99.jpg
```

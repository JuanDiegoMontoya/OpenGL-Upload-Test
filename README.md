# OpenGL Upload Test

A quick and dirty program to test quick methods of texture uploading in OpenGL.

## Currently implemented:

- [x] Serial loading from disk, serial upload
- [x] Parallel loading from disk, serial upload
- [x] Parallel loading from disk, parallel upload (with PBOs)
    - The PBOs and textures resources are still created serially
- [ ] Parallel loading from disk, parallel upload (with one big PBO)
- [ ] Parallel loading from disk, parallel upload with shared contexts

For loading images from disk, stb_image was used. For all parallelism, `std::for_each` with `std::execution::par` was used.

## Methodology

A simple timer class was used to time how long it took to load the textures.
The images used were 216 doges of various resolutions ranging from 128x128 to 2048x2048 pixels (PoT only).
The specific images don't matter much as long as sufficient data is provided to test parallelism and bulk uploading, since in any test stbi_load_* will perform roughly the same.
The results are the average eyeballed over a few runs.
For the PBO method, I tested a variety of combinations of all the buffer storage flags and none seemed to have an effect on upload performance.

## Hardware Used

| - | - |
| - | - |
| CPU | Ryzen 9 5950X |
| RAM | 32 GB DDR4 3600 |
| GPU | RTX 3070 |
| Graphics Driver | 511.65 |
| Storage | Force MP510 M.2 |

## Results

| Method | Time(ms) |
| - | - |
| Serial Load, Serial Upload | 3150 |
| Parallel Load, Serial Upload | 430 |
| Parallel Load, Parallel Upload (PBOs) | 520 |

## Analysis

It's obvious that loading from disk should be parallelized, especially when considering how trivial it is to do with `std::execution::par`.

The next question is: how can I parallelize uploading this data to the GPU? This is actually extremely cheap even with over 900 MBs of data.

My hypothesis was to create a PBO per texture, map the pointers, then `memcpy` the texture data to those pointers in parallel.
As it turns out, this method is actually slower than serially uploading!
This is likely due to the extra overhead of the driver allocating mapped memory and later DMA'ing on its own, as we aren't actually accessing device-local memory with mapped pointers (most likely)!

On AMD hardware, the situation may be different as there is indeed 256 MB of host-local, device-visible pinned memory that can be mapped and written to, which could affect the performance characteristics of uploading data (at the expense of performance when using the data).

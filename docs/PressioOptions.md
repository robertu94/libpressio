# Configuration Options {#pressiooptions}

## Compressors

### BLOSC

BLOSC is a collection of lossless compressors optimized to transfer data more quickly than a direct memory fetch can preform.  More information on BLOSC can be found on its [project homepage](https://blosc.org/pages/blosc-in-depth/)

option                  | type        | description
------------------------|-------------|------------
`blosc:blocksize` | uint32 | the desired blocksize, defaults to automatic; see project documentation for restrictions
`blosc:clevel` | int32 | the desired compression level from 0 (no compression) to 9 (max compression)
`blosc:compressor` | char* | a compressor name corresponding to a blosc compressor codec
`blosc:doshuffle` | int32 | what if any kind of pre-bit shuffling to preform
`blosc:numinternalthreads` | int32 | number of threads used internally by the library

### ImageMagick

ImageMagick is a robust library that preforms a wide array of image compression and manipulation.  Only a fraction of its api is exposed.  More information on ImageMagick can be found on its [project homepage](https://imagemagick.org/)

**Warning** When using ImageMagick with floating point values, they must first be normalized between 0 and 1.


option                  | type        | description
------------------------|-------------|------------
`magick:compressed_magick` | char* | the image format to use.
`magick:quality` | uint32 | the quality to use for compression if it applies
`magick:samples_magick` | char* | the pixel format to assume for input

### MGARD

MGARD is a error bounded lossy compressor based on using multi-level grids. At time of writing, MGARD is experimental and disabled from building in libpressio by default.  More information can be found on onis [project homepage](https://github.com/CODARcode/MGARD)


option                  | type        | description
------------------------|-------------|------------
`mgard:norm_of_qoi` | double | for use with a precomputed norm of the quality of interest
`mgard:qoi_double` | void* | a function pointer to a quality of interest function
`mgard:qoi_float` | void* | a function pointer to a quality of interest function
`mgard:s` | double | the norm in which the error will be preserved
`mgard:tolerance` | double | upper bound for the desired tolerance

### SZ

SZ is an error bounded lossy compressor that uses prediction based methods to compress data.
More information can be found about SZ on its [project homepage](https://github.com/disheng222/sz).

option                  | type        | description
------------------------|-------------|------------
`sz:abs_err_bound` | double | the absolute error bound
`sz:accelerate_pw_rel_compression` | int32 | trade compression ratio for a faster pw_rel compression
`sz:app` | char* | access a application specific mode of SZ
`sz:config_file` | char* | Write-Only, filepath passed to `SZ_Init()`
`sz:config_struct` | void* | Write-Only, structure passed to `SZ_Init_Params()`
`sz:data_type` | double | an internal option to control compression
`sz:error_bound_mode` | int32 | which error bound mode to use
`sz:gzip_mode` | int32 | Which mode to pass to GZIP when used
`sz:lossless_compressor` | int32 | Which lossless compressor to use for stage 4
`sz:max_quant_intervals` | uint32 | the maximum number of quantization intervals
`sz:max_range_radius` | uint32 | an internal option to control compression
`sz:plus_bits` | int32 | Internal option Used in `accelerate_pw_rel_compression` mode
`sz:pred_threshold` | float | an internal option used to control compression
`sz:prediction_mode` | int32 | an internal option used to control compression
`sz:psnr_err_bound` | double | a bound on the PSNR used in the PSNR error bound mode
`sz:pw_rel_err_bound` | double | a pointwise-relative error bound
`sz:pwr_type` | int32 | an interal option to control compression, point-wise relative error bound byte, example: 25
`sz:quantization_intervals` |  uint32 | he number of quantization intervals to use, the default 0 means automatic
`sz:random_access` | int32 | internal option use the random access mode when compiled in
`sz:rel_err_bound` | double | the value range relative error bound
`sz:sample_distance` | int32 | internal option used to control compression.
`sz:segment_size` | int32 | internal option used to control compression. number of points in each segement for `pw_relBoundRatio`
`sz:snapshot_cmpr_step` | int32 | the frequency of preforming single snapshot based compression in time based compression
`sz:sol_id` | int32 | an internal option used to control compression.
`sz:sz_mode` | int32 | SZ Mode either `SZ_BEST_COMPRESSION` or `SZ_BEST_SPEED`
`sz:user_params` | void* | arguments passed to the application specific mode of SZ.  Use in conjunction with `sz:app`

### ZFP

ZFP is an error bounded lossy compressor that uses a transform which is similar to a discrete cosine transform.  More information on ZFP can be found on its [project homepage](https://zfp.readthedocs.io/en/release0.5.5/)

option                  | type        | description
------------------------|-------------|------------
`zfp:accuracy` | double | Write-only, absolute error tolerance for fixed-accuracy mode
`zfp:dims` | int32 | Write-only, the dimensionality of the input data, used in fixed-rate mode
`zfp:execution` | int32 | which execution mode to use
`zfp:maxbits` | uint32 |  maximum number of bits to store per block
`zfp:maxprec` | uint32 | maximum number of bit planes to store
`zfp:minbits` | uint32 | minimum number of bits to store per block
`zfp:minexp` | int32 | minimum floating point bit plane number to store
`zfp:mode` | uint32 | a compact encoding of compressor parmeters
`zfp:omp_chunk_size` | uint32 | OpenMP chunk size used in OpenMP mode
`zfp:omp_threads` | uint32 | number of OpenMP threads to use in OpenMP mode
`zfp:precision` | uint32 | Write-only, The precision specifies how many uncompressed bits per value to store, and indirectly governs the relative error.
`zfp:rate` | double | Write-only the rate used in fixed rate mode
`zfp:type` | uint32 | Write-only, the type used in fixed rate mode
`zfp:wra` | int32 | Write-only, write random access used in fixed rate mode


## Metrics

Some metrics also support the use of configuration options.


### External

The external metrics module allows running 3rd party metrics without having to port them to C++.  More information can be found [in libpressio's documentation](@ref usingexternalmetric)

option                  | type        | description
------------------------|-------------|------------
`external:command`      | char*       | the command to run

# Built-in Modules {#builtins}

Libpressio provides a number of builtin compressor and metrics modules.

## Compressor Plugins

+ `sz` -- the SZ error bounded lossy compressor
+ `zfp` -- the ZFP error bounded lossy compressor
+ `mgard` -- the MGARD error bounded lossy compressor
+ `blosc` -- the blosc lossless compressor
+ `magick` -- the ImageMagick image compression/decompression library

## Metrics Plugins

+ `time` -- time information on each compressor API
+ `error_stat` -- statistics on the difference between the uncompressed and decompressed values that can be computed in one pass in linear time.
+ `pearson` -- computes the pearson coefficient of correlation and pearson coefficient of determination.
+ `size` -- information on the size of the compressed and decompressed data
+ `external` -- run an external program to collect some metrics, see [using an external metric for more information](@ref usingexternalmetric)

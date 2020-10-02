# Metric Results {#metricsresults}

LibPressio supports a variety of basic metrics for evaluating compression.

## Error Stat

Compute common error statistics that can be computed in a single pass.

Metric                  | Type        | Description
------------------------|-------------|-------
`error_stat:average_difference` | double  | average difference in value
`error_stat:average_error` | double  | average absolute difference in value
`error_stat:difference_range` | double  | max difference - min difference
`error_stat:error_range` | double  | max absolute difference - min absolute difference
`error_stat:max_error` | double  | max absolute difference
`error_stat:max_rel_error` | double  | max absolute difference / value range
`error_stat:min_error` | double  | min absolute difference
`error_stat:min_rel_error` | double  | min absolute difference / value range
`error_stat:mse` | double  | mean squared error
`error_stat:psnr` | double  | full-range PSNR
`error_stat:rmse` | double  | root mean squared error
`error_stat:value_max` | double  | max value in the input dataset
`error_stat:value_mean` | double  | mean value in the input dataset
`error_stat:value_min` | double  | min value in the input dataset
`error_stat:value_range` | double  | value range in the input dataset
`error_stat:value_std` | double  | sample standard deviation on the input dataset

## Pearson's Coefficients

Computes pearson's statistics and related quantities

Metric                  | Type        | Description
------------------------|-------------|-------
`pearson:r` | double  | Pearson's correlation coefficient between the input and output dataset
`pearson:r2` | double  |  uncorrected Pearson's determination coefficient between the input and output dataset

## Size

Computes information about the size

Metric                  | Type        | Units  | Description
------------------------|-------------|--------|-------
`size:bit_rate` | double  | bits per element | the number of bits per symbol in the dataset
`size:compressed_size` | uint32  | bytes | the size of the compressed dataset
`size:compression_ratio` | double  | unitless | the ratio of the input size to compressed size.
`size:decompressed_size` | uint32  | bytes | the size of the decompressed dataset
`size:uncompressed_size` | uint32  | bytes | the size of the input dataset

## Time

Computes information about the runtime of compressors

Metric                  | Type        | Units  | Description
------------------------|-------------|--------|-------
`time:check_options` | uint32  | ms | time to check options
`time:compress` | uint32  | ms | time to compress
`time:decompress` | uint32  | ms | time to decompress
`time:get_options` | uint32  | ms | time to get options
`time:set_options` | uint32  | ms | time to set options


## Composite

The composite metric is special in that it is not activated explicitly, but when other metrics are enabled.  If all the metrics in the activated column are activated, this metric will have a value.

Metric                  | Type        | Units  | Description | Activated
------------------------|-------------|--------|--------|------------
`composite:compression_rate` | double  | kb/s | kilobytes compressed per second | time, size
`composite:decompression_rate` | double  | kb/s | kilobytes decompressed per second | time, size

Additionally, if the `LIBPRESSIO_HAS_LUA` build configuration is enabled, custom composite metrics may be provided as lua scripts.
The scripts are passed to the plugin via the `composite:scripts` metric option.
The scripts are parsed in the order provided and subsequent scripts are given the values of previous scripts.
Each scripts are allowed to use the `base` and `math` lua standard libraries and are passed the options as a table of doubles called `metrics`.
Each scripts should return a tuple of `string`, `double` and will be named `composite:$name` and assigned the value provided in the double.
If a given script errors or fails, the value it provides will not be created.


## External

The metrics supported by external are too complicated to summerize here, [please see its documentation](@ref usingexternalmetric)

## SZ

These are the metrics reported by the SZ compressor.  Requires the `BUILD_STATS` cmake option for SZ to be enabled.

Metric                  | Type        | Units  | Description
------------------------|-------------|--------|-------
sz:block_size  | uint32 |elements |  size of a block in SZ
sz:huffman_coding_size  | uint32 | bytes| the size of codes in the huffman tree
sz:huffman_compression_ratio  | float | |  the compression ratio due to just the huffman tree
sz:huffman_node_count  | uint32 |nodes | the number of nodes in the huffman tree
sz:huffman_tree_size  | uint32 |bytes | the size of the huffman tree
sz:lorenzo_blocks  | uint32 | elements |  the number of blocks that used the lorenzo predictor
sz:lorenzo_percent  | float | |  the percentage of blocks that use the lorenzo predictor
sz:regression_blocks  | uint32 | elements |  the number of blocks that used the regression predictor
sz:regression_percent  | float | | the percentage of blocks that used the regression predictor
sz:total_blocks  | uint32 | elements |  the total number of blocks processed
sz:unpredict_count  | uint32 | elements |  the number of unpredictable blocks
sz:use_mean  | int32 | | true/false if the mean was used

## MGARD

These are the metrics reported by the MGARD compressor.

Metric                  | Type        | Units  | Description
------------------------|-------------|--------|-------
mgard:norm_of_qoi  | double | | the norm computed in QOI mode
mgard:norm_time  | uint32 | ms | the time required to compute the norm in QOI mode

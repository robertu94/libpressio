# Using the External Metric {#usingexternalmetric}

LibPressio provides an `external` metrics module that runs a script or program provided by the user.
This exists to better reuse existing analysis scripts or routines that users may wish to use without requiring them be ported to C++ and where upstreaming them does not make sense because they are too niche to be broadly applicable.
In order to correctly communicate between LibPressio and an External script, strict communication semantics must be followed which may require writing a small wrapper in a high level language.

The external module is not intended to replace writing metrics modules in C/C++.
Pull requests for such modules will not be accepted.


## Configuration Options

`external:command` -- the command to execute,  the options passed by the module will be appended to this string
`external:io_format` -- the format to write the data to disk.  It can be any format supported by `pressio_supported_io_modules`

Additionally any options passed to this metric will be passed to the IO format module.

## Command line Arguments

The `external` plugin will provide the following command line arguments to the script.
These may change from version to version.

`--api` the maximum API version number the external module supports, begins at 1.  The current version is 2

`--input` path to a temporary file containing the input data prior to compression. (new in version 2) It will be according to the `external:io_format` option

`--decompressed` path to a temporary file containing the input data prior to compression. (new in version 2) It will be according to the `external:io_format` option

`--dim` dimension the dimensions of the dataset from low to high.  This argument may be passed more than once.  If passed more than once, the dimensions are given in order same order as the `pressio_data_new` functions.

`--type` type of the input data.  Valid types include: "float", "double", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "byte".


### Example Usage

`/path/to/script --input /tmp/input.f32 --decompressed /tmp/decompressed.f32 --dim 500 --dim 500 --dim 100 --type float`

In this case `external:command` is set to `/path/to/script`

`/usr/bin/env python myscript.py --input /tmp/input.f32 --decompressed /tmp/decompressed.f32 --dim 500 --dim 500 --dim 100 --type float`

In this case `external:command` is set to `/usr/bin/env python myscript.py`

## Expected Return Code

The external command is expected to return 0 on success, positive values on an error, and negative on warning.


## Expected Standard Output

A line `external:api=$version_number\n` where `$version_number` is a positive integer.

### version 1:

After the API line, one or more lines conforming to the following pattern:

`$var_name=$value\n`

Where:

+ `$var_name` is any string that does not contain a literal ascii `=` or `\n` character.  It may not begin with the prefix `external:`.
+ `=` is a literal ascii "=" character
+ `$value` is a value that can be parsed using `strtod`
+ `\n` is a newline character


An example output could be:

```
external:api=1
auto_cor1=.099
auto_cor2=.099
auto_cor3=.097
ssim=.64
my_analysis=1.03e-3
```

### version 2:

No changes were made to the output format since version 1.


## Expected Standard Error


If the return code is non-zero, a warning or error message SHOULD be printed to stderr.  It SHOULD be a human readable message designed for use in debugging.

These warnings/errors will be reported to the user with in the `metric_results` under the key `external:stderr`.


An example output could be:

```
foobar analysis only supports being run on 2d data
```

## Example 

Assume that an external metrics module ran and exited with a return code of 1 generating the output above will generate following keys in an arbitrary order.

```
external:results:auto_cor1=.099
external:results:auto_cor2=.099
external:results:auto_cor3=.097
external:results:ssim=.64
external:results:my_analysis=1.03e-3
external:return_code=1
external:error_code=0
external:stderr="foobar analysis only supports being run on 2d data"
```

## Non Guarantees

All of the above arguments MAY not be passed in later versions of the API.

The order of the command line arguments is unspecified and may be passed in any order unless other wise noted (i.e. dimensions are passed in the same order as to one of the `pressio_data_new` functions).  

The `external` plugin may pass additional command line arguments not specified here.

The environment of the script is unspecified.

Do not assume that your command is passed to a shell.

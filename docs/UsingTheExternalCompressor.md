# Using the External Compressor

LibPressio provides an `external` compressor module for interacting with 3rd party compressors that would be extremely difficult to port to C or C++.
In order for LibPressio to communicate with the external compressor strict semantics must be obeyed.

The external module is not intended to replace writing compressor modules in C/C++.
Pull requests for these will not be accepted.

@tableofcontents

## Configuration Options


If the external metric script requires 2 or more input files, it MUST pass the `external:field_names` option with a length corresponding to the number of input files.
If other arguments taking a array of strings are passed, their length must equal the number of input files or be 0 or be unset.
If the length is 0 or unset, a default option will be used for each input file.
If the length is non-zero, the ith option corresponds to the ith file.

| Option                     |Version Added | Type     | Description                                                                                             |
|----------------------------|--------------|----------|---------------------------------------------------------------------------------------------------------|
| `external:command`         | 1            |`char*`   | the command to execute,  the options passed by the module will be appended to this string               |
| `external:io_format`       | 2            |`char*[]` | the format to write the data to disk.  It can be any format supported by `pressio_supported_io_modules` The ith entry in the list corresponds to the ith input file. |
| `external:field_names`     | 3            |`char*[]` | the prefix to use for arguments relating to the files.  For example, if the list `["foo"]` is provided, arguments would be like `--foo_dim`.  If this list is empty, The script will be passed arguments of the form `--dim`.|
| `external:prefix`          | 3            |`char*[]` | the prefix to use for arguments relating to the files. The default is an empty prefix.                  |
| `external:suffix`          | 3            |`char*[]` | the suffix to use for arguments relating to the files. The default is an empty suffix.                  |
| `external:work_dir`        | 3            |`char*`   | the path to call the script from, defaults to the current working directory                             |
| `external:launch_method`   | 3            |`char*`   | the method used to launch the worker task.  It can be one of "forkexec" or "mpispawn"                   |
| `external:config_name`     | 4            |`char*`   | A string passed to the external metric for the "configuration name", by default "external"              |

## "Well-Known" Command line Arguments

The `external` plugin will provide the following command line arguments to the script.
These may change from version to version.

`--api` the maximum API version number the external module supports, begins at 5.  The current version is 5

`--config_name` the value passed to the `external:config_name` option.  The implementation *may* use this value to as a basis to name log files or other auxiliary outputs.

`--mode` the mode of the external compressor.  One of `get_options`, `set_options`, `get_configuration`, `get_documentation`, `compress`, `decompress`.

`--name` the name of the external compressor used for hierarchical options setting.

`--options` a string  JSON suitable for parsing with `pressio_options_new_json` containing the options of the compressor.  The `get_options`, `get_configuration`, and `get_documentation` setting must produce output without this setting,  all other documented modes MAY require this option.

If the mode is compress or decompress, the following arguments will be provided and "prefixed" according to the `external:field_names` argument if it is set. See the table above in "Configuration Options" and example usage below.

`--input` path to a temporary file or shared memory containing the input data prior to compression. It will be interpreted according to the `external:in_io_format` option

`--output` (optional) path to a temporary file or shared memory pre-allocated for output. It will be interpreted according to the `external:out_io_format` option.

`--idim` and `--odim` dimension the dimensions of the (input and output) dataset from low to high.  This argument may be passed more than once.  If passed more than once, the dimensions are given in order same order as the `pressio_data_new` functions.

`--idim` and `--odim` dimension the dimensions of the dataset from low to high.  This argument may be passed more than once.  If passed more than once, the dimensions are given in order same order as the `pressio_data_new` functions.

`--itype` and `--otype` type of the input and output data.  Valid types include: "bool", "float", "double", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "byte".

## "Custom Defined" Command line Arguments

It **is** guaranteed, that no future arguments will use names beginning with `external` and may be safely used for custom arguments specified by the user.

## Error Codes

External Metrics indicates it's own status (as opposed to the script's status) using the `external:error_code` parameter.
Here are the values it can be:

| Value | Meaning                                                                                 | 
|-------|-----------------------------------------------------------------------------------------|
| 0     | Successful launch                                                                       |
| 1     | Pipe Error -- failed to create the pipe                                                 |
| 2     | Fork Error -- failed to create the child process                                        |
| 3     | Exec Error -- failed to exec the child process to the script                            |
| 4     | Format Error -- the script returned output that the module did not understand           |

# Non Guarantees

All of the above arguments MAY not be passed in later versions of the API.

The order of the command line arguments is unspecified and may be passed in any order unless other wise noted (i.e. dimensions are passed in the same order as to one of the `pressio_data_new` functions).  

The `external` plugin may pass additional command line arguments not specified here.  It **is** guaranteed, that no future arguments will use names beginning with `external` and may be safely used for custom arguments specified by the user.

The environment of the script is unspecified.

Do not assume that your command is passed to a shell.

## Version 5

### Expected Standard Output

A line `external:api=$version_number\n` where `$version_number` is a positive integer.

After the API line, one or more lines conforming to the following pattern:

`$var_name=$value\n`

Where:

+ `$var_name` is any string that does not contain a literal ascii `=` or `\n` character.  It may not begin with the prefix `external:`.
+ `=` is a literal ascii "=" character
+ `$value` is a value containing only ascii characters not the '\n'. If it can be parsed with `stdtod` it will be treated a double, otherwise as a string.  If an `$var_name` is repeated it will be treated as a 1d pressio array of doubles or strings respectively
+ `\n` is a newline character

For the compress and decompress command, at least one set of keys must be provided of the form

`output:(\d+):path`
`output:(\d+):dtype`
`output:(\d+):dims`

If --output flag(s) were passed, the data should be written to the described location, and `output:(\d+):path` should match these paths.

If the space provided is insufficient for the external compressor may return an error.  The numeric value should start at 0, and increment by 1 for each data set to read in.

Lines that begin with `metric:` will be treated as metrics internal to the compressor.

An example output for compress/decompress could be:

```
external:api=5
output:0:path=/tmp/data.out
output:0:dtype=byte
output:0:dims=4096
```

or 

```
external:api=5
output:0:path=/tmp/data.out
output:0:dtype=float
output:0:dims=500
output:0:dims=500
output:0:dims=100
```

or 

```
external:api=5
output:0:path=/tmp/data.out
output:0:dtype=float
output:0:dims=512
output:0:dims=512
output:0:dims=512
metric:stage1_time=300
metric:stage2_time=20
metric:stage3_time=84
```

for the `get_options`, `get_configuration` and `get_documentation` commands output could be like

```
external:api=5
cast:cast=float
```

where each line after the header is an option

for the `set_options` command output could be like

```
external:api=5
```

This may change in the future to add additional lines

### Expected Standard Error


If the return code is non-zero, a warning or error message SHOULD be printed to stderr.  It SHOULD be a human readable message designed for use in debugging.

These warnings/errors will be reported to the user with in the `metric_results` under the key `external:stderr`.


An example output could be:

```
foobar analysis only supports being run on 2d data
```

### Expected Return Code

The external command is expected to return 0 on success, positive values on an error, and negative on warning.

### Example 

Assume that an external metrics module ran and exited with a return code of 1 with the 3rd output aove will generate following keys in an arbitrary order.

```
external:results:stage1_time=300
external:results:stage2_time=20
external:results:stage3_time=84
external:return_code=1
external:error_code=0
external:stderr="foobar analysis only supports being run on 2d data"
```

And will read the data with dimensions 512, 512, 512 with type float from `/tmp/data.out` as interpreted by the current io_format (the default being POSIX).

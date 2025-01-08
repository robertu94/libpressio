# Using the External Metric {#usingexternalmetric}

LibPressio provides an `external` metrics module that runs a script or program provided by the user.
This exists to better reuse existing analysis scripts or routines that users may wish to use without requiring them be ported to C++ and where upstreaming them does not make sense because they are too niche to be broadly applicable.
In order to correctly communicate between LibPressio and an External script, strict communication semantics must be followed which may require writing a small wrapper in a high level language.

The external module is not intended to replace writing metrics modules in C/C++.
Pull requests for such modules will not be accepted.

@tableofcontents


## Configuration Options

*New in external API 3* if the external metric script requires 2 or more input files, it MUST pass the `external:field_names` option with a length corresponding to the number of input files.
Additionally, if other arguments taking a array of strings are passed, their length must equal the number of input files or be 0 or be unset.
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


*New in external API 2* "global" IO module options may passed as well.

*New in external API 3* If `external:field_names` is unset and the metrics module is unnamed, the IO module will also be assigned the name `external`.  If `external:field_names` is unset and the metrics module is named, the IO module will be assigned the same name as the external metric module.  If `external:field_names` is set, the IO modules used will be assigned names according the field names.  If the external metric is unnamed, the io module that writes out the `x` field will be assigned the name `x`.  If the external metric is assigned a name, the io module is a assigned the name of the external metric followed by a `/` followed by the field name.  For example, if the external metric name is`imaging_metric`, and the field name is `x`, the io module that writes out the x field will be assigned the name `imaging_metric/x`.


## "Well-Known" Command line Arguments

The `external` plugin will provide the following command line arguments to the script.
These may change from version to version.

`--api` the maximum API version number the external module supports, begins at 1.  The current version is 5

*New in external API 4* `--config_name` the value passed to the `external:config_name` option.  The implementation *may* use this value to as a basis to name log files or other auxiliary outputs.

**New Requirement in API 3** If the script is called with zero of the following well-known arguments, it MUST output a lists of the known results with default values in-case they are omitted from calls to the script when called with some set of well-known flags.  Scripts MAY return different lists of values when called with defined custom arguments.  See the example below

*New in external API 3* The following arguments will be "prefixed" according to the `external:field_names` argument if it is set. See the table above in "Configuration Options" and example usage below.

*New in external API 7* When there are multiple datasets provided as input, they always begin with `--input`.

*New in external API 7* `--eval_uuid` a string containing a UUID that _should_ be unique per invocation of the metric

`--input` path to a temporary file containing the input data prior to compression. (new in version 2) It will be interpreted according to the `external:io_format` option

`--decompressed` path to a temporary file containing the input data prior to compression. (new in version 2) It will be interpreted according to the `external:io_format` option

`--dim` dimension the dimensions of the dataset from low to high.  This argument may be passed more than once.  If passed more than once, the dimensions are given in order same order as the `pressio_data_new` functions.

`--type` type of the input data.  Valid types include: "bool", "float", "double", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "byte".


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

## Example Usage

`/path/to/script --input /tmp/input.f32 --decompressed /tmp/decompressed.f32 --dim 500 --dim 500 --dim 100 --type float`

In this case `external:command` is set to `/path/to/script`

`/usr/bin/env python myscript.py --input /tmp/input.f32 --decompressed /tmp/decompressed.f32 --dim 500 --dim 500 --dim 100 --type float`

In this case `external:command` is set to `/usr/bin/env python myscript.py`

`/path/to/script --foo_input /tmp/foo_input.f32 --foo_decompressed /tmp/decompressed.f32 --foo_dim 500 --foo_dim 500 --foo_dim 100 --foo_type float`

In this case, `external:command` is set to `/path/to/script` and `external:feild_names` is set to `['foo']`

`/path/to/script --external_custom_arg 3`

In this case, `external:command` is set to `/path/to/script --external_custom_arg 3`. Where `--external_custom_arg` is a script defined custom argument.

**NOTE** the names provided in this example are illustrative only, the real names are generated by `mkstemp`

# Non Guarantees

All of the above arguments MAY not be passed in later versions of the API.

The order of the command line arguments is unspecified and may be passed in any order unless other wise noted (i.e. dimensions are passed in the same order as to one of the `pressio_data_new` functions).  

The `external` plugin may pass additional command line arguments not specified here.  It **is** guaranteed, that no future arguments will use names beginning with `external` and may be safely used for custom arguments specified by the user.

The environment of the script is unspecified.

Do not assume that your command is passed to a shell.

# "forkexec" launch method

## Version 1

### Expected Standard Output

A line `external:api=$version_number\n` where `$version_number` is a positive integer.


After the API line, one or more lines conforming to the following pattern:

`$var_name=$value\n`

Where:

+ `$var_name` is any string that does not contain a literal ascii `=` or `\n` character.  It may not begin with the prefix `external:`.
+ `=` is a literal ascii "=" character
+ `$value` is a value treated as a double if parse-able with `strtod` (all versions) or a string containing no newlines otherwise (starting in version 6)
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

## Version 2

No changes were made to the output format since version 1.

## Version 3

The output from a script has changed to reflect the zero well-known argument case. Otherwise the format is unchanged.

### Expected Standard Output

**New requirement in external API 3** If the script is called with zero well-known arguments, an example output could be:

```
external:api=1
auto_cor1=0.0
auto_cor2=0.0
auto_cor3=0.0
ssim=0.0
my_analysis=0.0
foobar=2.7
```


### Example 

Assume that an external metrics module ran and exited with a return code of 1 generating the output above will generate following keys in an arbitrary order.

```
external:results:auto_cor1=.099
external:results:auto_cor2=.099
external:results:auto_cor3=.097
external:results:ssim=.64
external:results:my_analysis=1.03e-3
external:results:foobar=2.7
external:return_code=1
external:error_code=0
external:stderr="foobar analysis only supports being run on 2d data"
```

**Note** that `external:results:foobar` only appears in the zero well-known argument version of the output.
It is thus presented with its default value.

## Version 4

No changes were made to the output format.  The `--config_name` argument was introduced

# Version 5

No changes were made to the output format.  A double field `external:duration` is now included for external metrics which contains the runtime in seconds.

# Version 6

In addition to doubles, strings can now be returned from external metrics.  These strings cannot contain a newline character.

# Version 7

When there are multiple data fields, `--input`  is guaranteed to start each fields arguments

# Version `json:1`

An alternative output format for stdout that allows providing values in a more complex way using JSON.

```
external:api=json:1
$json
```

where `$json` is an object parse-able using `pressio_options_from_json`. Here is an example:

```
external:api=json:1
{"auto_cor1":0.0, "auto_cor2": 0.0, "auto_cor3": 0.0, "ssim": 0.0, "my_analysis": 0.0, "foobar": 2.7}
```


# "mpispawn" launch method

This method spawns the exernal metric using the `MPI_Comm_spawn` routine with a series of calls similar to the following:

```cpp
std::vector<char*> args; // contains the command line arguments beginning with
                         // the name of the executable and ending with an nullptr
MPI_Info info;
MPI_Info_create(&info);
MPI_Info_set(info, "wdir", workdir.c_str());
MPI_Comm_spawn(args.front(), args.data()+1, 1, info, 0,  MPI_COMM_SELF, &child, &error_code); 
```

## Version 1 and 2

The `mpispawn` launch method does not  support external metrics api versions 1 or 2.

## Version 3 - 7

Equivelent to `forkexec` with the following exceptions.

Instead of connecting standard out and standard error, the external script should communicate with the external metric module using a series of MPI calls presented below.


```cpp
int status_code = 0;    //a status code; see `forkexec` return code
std::string stdout_str; //the output of the external metric; see `forkexec` format for stdout
std::string stderr_str; //the output of the external metric; see `forkexec` format for stderr

int stdout_len = stdout_str.size();
int stderr_len = stderr_str.size();

MPI_Comm parent;
MPI_Comm_get_parent(&parent);
MPI_Send(&status_code, 1, MPI_INT, 0, 0, parent);
MPI_Send(&stdout_len, 1, MPI_INT, 0, 0, parent);
MPI_Send(stdout_str.c_str(), stdout_len, MPI_CHAR, 0, 0, parent);
MPI_Send(&stderr_len, 1, MPI_INT, 0, 0, parent);
MPI_Send(stderr_str.c_str(), stderr_len, MPI_CHAR, 0, 0, parent);
```

# "remote" launch method

This method uses curl to connect to a server that provides the metric

# Version 3-7

Equivelent to `forkexec` with the following exceptions.

When arguments are passed, they are provided in a json list inside a dictionary with the key "args"

example

```json
{ "args": ["--api=7"] }
```

When the request is returned it is nested in a json dictionary with keys stderr, stdout, and return_code formatted as described for `forkexec`

example

```json
{
    "stdout": "external:api=7\nfoo=3",
    "stderr": "",
    "return_code": 0,
}
```

# `python` launch method

Equivilent to the `forkexec` method with the following exceptions.

the list of arguments are passed as a python list of strings with the name `cmd`
the the stdout, stderr, and return code are returned via variables `stdout`, `stderr` and `ret` respectively.

**limitation** this metric implements thread-safety with a global mutex grabbed at invocation time.

example script (assumes python 3.6 or later)`

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--api", type=int)
parser.add_argument("--input")
parser.add_argument("--decompressed")
parser.add_argument("--dim", type=int, action="append")
parser.add_argument("--dtype")
args, _ = parser.parse_known_args(cmd)

if args.api is not None:
    foo = 3
else:
    foo = 0

stdout = f"""external:api=7
foo={foo}
stderr = ""
"""
```

# Standard Options

## LibPressio

### Core

+ `pressio:children <char*[]>` children of this libpressio meta object
+ `pressio:highlevel <char*[]>` ordered list of settings that the uesr should consider modifying first
+ `pressio:prefix <char*>` prefix of the this libpresiso meta object
+ `pressio:stability <char*>` level of stablity provided by the compressor
+ `pressio:thread_safe <threadsafety>` level of thread safety provided by the compressor
+ `pressio:type <char*>` type of the libpressio meta object
+ `pressio:version <char*>` the version string from the compressor
+ `pressio:version_epoch <uint64>` the epoch version number; this is a libpressio specific value used if the major_version does not accurately reflect backward incompatibility
+ `pressio:version_major <int32>` the major version number
+ `pressio:version_minor <int32>` the minor version number
+ `pressio:version_patch <int32>` the patch version number

### Compressors

#### Error Bounds

+ `pressio:abs <double>` a pointwise absolute error bound
+ `pressio:rel <double>` a pointwise value-range relative error bound
+ `pressio:pw_rel <double>` a pointwise value-range relative error bound
+ `pressio:lossless <int32>` the balance between high speed vs high compression lossless compression

#### Performance

+ `pressio:nthreads <uint32>` number of threads to use

#### Metrics

+ `pressio:metric <char*>` metrics to collect when using the compressor


### IO

+ `io:path <char*>` path to the file on disk

### Launch

+ `external:commands <char*[]>`
+ `external:workdir <char*>`
+ `external:connection_string <char*>`

## LibPressio-Predictor 

### Opt 

#### Inputs and Outputs

+ `opt:inputs <char*[]>` what inputs to use.
+ `opt:output <char*[]>` what variables are outputs.  If optimization is not multi-objective, other objectives are collected but not used.

#### Upper and Lower Bounds

+ `opt:is_integral <data>` is a bound integer or double?
+ `opt:lower_bound <data>` lower bound for searching
+ `opt:upper_bound <data>` upper bound for searching
+ `opt:prediction <data>` prediction/inital guess

#### Objectives and Targets

+ `opt:target <double>` what target to use in target mode
+ `opt:objective_mode` what objective code (integer) to use
+ `opt:objective_mode_name <char*>` what objective mode name to use (min, max, target)

#### Evaluation Limits

+ `opt:max_iterations <uint32>` max iterations to search
+ `opt:max_seconds <uint32>` soft target max seconds to search, however, because cancellation is cooperative, this may be exceeded
+ `opt:global_rel_tolerance <double>` global search stop search
+ `opt:local_rel_tolerance <double>` local tolerance to stop local refinement

#### Pre-Evaluation History

+ `opt:evaluations <data>` results from a prior evaluation

#### Performance

+ `opt:inter_iteration <uint32>` should inter-iteration cancellation support be used
+ `opt:do_decompress <int32>` should decompression be run

#### Search

+ `opt:compressor <char*>` what compressor to run
+ `opt:search <char*>` what search method to use
+ `opt:search_metrics <char*>` what callbacks to use when running a search


### Predictors


#### Compressors

+ `predictors:runtime <char*[]>` list of options that invalidate the runtime of the compressor
+ `predictors:error_dependent <char*[]>` list of options that invalidate the error dependent metrics
+ `predictors:error_agnostic <char*[]>` list of optinos that invalidate error agnostic metrics

#### Metrics

First  provide

+ `predictors:requires_decompress <char*[]>|bool` list of metrics that a metric requires decompression to collect or a bool to indicate that none of or all or the metrics require decompression.

Then provide EITHER:

+ `predictors:invalidate <char*[]>` what invalidates all metrics returned by `get_metrics_results()` which contains a possibly empty subest of `predictors:runtime`, `predictors:error_dependent`, `predictors:error_agnostic`, `predictors:data`, or `predictors:nondeterministc`.

or a non-empty subset of

+ `predictors:runtime <char*[]>` the list of metrics returned by `get_metrics_results()` that are effected by runtime
+ `predictors:error_dependent <char*[]>` list of metrics returned by `get_metrics_results()`that effected by the error characteristics
+ `predictors:error_agnostic <char*[]>` list of metrics returned by `get_metrics_results()`that are agnostic to errors
+ `predictors:data <char*[]>` list of metrics returned by `get_metrics_results()` depend only one the uncompressed data
+ `predictors:nondeterministc <char*[]>` list of metrics returned that are never consistent returned from `get_metrics_results()`

#### Scheme

`predictors:training` is not an option, but is used by schemes, to indicate that a metrics is only required for trianing

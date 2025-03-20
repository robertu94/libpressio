# Contribution Guidelines

Thanks for you interest in LibPressio!  Here are a few things you can do to speed up the merging of you contribution.

## General Things


+ Please make sure that you have permission from your home institution to
  contribute code under the license for LibPressio.  This can take the form of
  a copyright assignment or a statement of policy + a written acknowledgment
  that you are contributing your code under the LibPressio license.
+ LibPressio maintains a linear history with version numbers for every change.
  Please squash all commits and update the version number appropriately as
  documented in the README.
+ Your code must pass all CI tests before it will be merged.
+ LibPressio code must compile with C++11.  This is to support some super
  computers that only have ancient compilers.  However, many of the C++14,17,
  and 20 library features have been backported in the `compat` namespace.  Feel
  free to use these.

## Plugins

LibPressio uses a key-value system to refer to configuration settings.

Each compressor may find specific configuration settings for its specific compressor with settings beginning with its compressor id as prefix (i.e. configurations for SZ begin with `sz:`).  [Refer to the specific compressors documentation](build/Compressors.md) for further documentation for each settings.

+ Use the `io:`, `metrics:` and `pressio:` where appropriate.
+ Avoid unnecessary casts.  Prefer C++ style casts where required.
+ Prefer full names to abbreviations.
+ Do not contribute dead code or code that has been commented out.
+ Please read the semantics for each of the functions that you provide
  implementations of and conform to these requirements.  If they are unclear,
  open an issue, and we will try to clarify the ambiguity.
+ Prefer the greatest level of thread safety possible for your plugin.  Avoid
  global state shared between plugin instances where possible.
+ Option names in `get_options` and `set_options` should be `snake_case`
+ The options in CMake should be `BIG_SNAKE_CASE` and follow naming conventions
  used for the other options
+ When adding a enum option strongly consider adding a `${enum_name}_str`
  option and configuration entry that accept strings to set the enum and list
  the values of the enum respectively.  See the SZ plugin for an example of
  this.
+ When setting returning an error or warning, alway use the `set_error`
  function.

# Contributors

The following people have contributed code to LibPressio in alphabetical order:

+ Ali Gok
+ Arham Khan
+ Emily E. Lattanzio
+ Hengzhi Chen
+ Jiannan Tian
+ Robert Underwood
+ Sheng Di
+ Victoriana Malvoso

# Acknowledgments

The code authors would like to acknowledge others who have contributed ideas,
suggestions, or other support during the development of LibPressio in
alphabetical order:

+ Ali Gok
+ Alison Baker
+ Amy Apon
+ Amy Burton
+ Dorit Hammerling
+ Justin Sybrant
+ Franck Cappello
+ Jon C. Calhoun
+ Kai Zhao
+ Sheng Di
+ Peter Lindstrom

# Funding Acknowledgments

This research was supported by the Exascale Computing Project (ECP),Project
Number: 17-SC-20- SC, a collaborative effort of two DOE organizations - the
Office of Science and the National Nuclear Security Administration, Responsible
for the planning and preparation of a capable Exascale ecosystem, including
software, applications, hardware, advanced system engineering and early testbed
platforms, to support the nation’s Exascale computing imperative.

The material was supported by the U.S. Department of Energy, Office of Science,
under contract DE-AC02-06CH11357, and supported by the National Science
Foundation under Grant No. , .


We acknowledge the computing resources provided on Bebop, which is operated by
the Laboratory Computing Resource Center at Argonne National Laboratory.

This research used resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (DOE) Office of Science user facility at Argonne National Laboratory and is based on research supported by the U.S. DOE Office of Science-Advanced Scientific Computing Research Program, under Contract No. DE-AC02-06CH11357.


### Feature Supported by the DOE ZF project (2024-...)

The material was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contract DE-AC02-06CH11357

#### 2025

+ Scheme interface for LibPressio Predict
+ Added support for MSZ

### Feature Supported by the NSF [FZ project](https://fzframework.org/) (2023-...)

This material is based upon work supported by the National Science Foundation under Grant No. 2311875.


#### 2025

+ Added support for Grib files

#### 2024

+ Preliminary implementation of LibPressio-JIT to just-in-time compiler compressor modules
+ Added support for QoZ compressor

#### 2023

+ Added support for $option:min and $option:max and pressio:highlevel to facilitate auto-tuning
+ View_segment Metrics API to inspect compressor internal state

### Feature Supported by [Illumine project](https://lcls.slac.stanford.edu/depts/data-systems/projects/illumine) (2024-...)

This work is supported by the U.S. Department of Energy (DOE) Office of Science, Advanced Scientific Computing Research and Basic Energy Sciences Advanced Scientific Computing Research for DOE User Facilities award ILLUMINE - Intelligent Learning for Light Source and Neutron Source User Measurements Including Navigation and Experiment Steering.

#### 2025

+ Added various functions to improve debuggability and usability of external metrics

#### 2024

+ Added support for LC and LC-GPU
+ Added support for pre-allocated buffers
+ Added Python support for Domains to support cuPY/PyTorch/TensorFlow arrays
 
### Feature Supported by SDR project (June 2021-...)

The material was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research (ASCR), under contract DE-AC02-06CH11357.

#### 2025

+ Support for multi-field external metrics

#### 2024

+ Added `predictors:*` interfaces to handle metrics validation
+ Added support for cuSZx
+ Added preliminary support for LibPressio-opt to LibPressio-predict
+ Added support for domains

#### 2023

+ Created LibPressio dataset to test with entire datasets quickly
+ Added support for Python based external metrics and compressors
+ Integrated various metrics from QCat
+ Wrote LibPressio-Predict to integrate compression prediction
+ Added support for cuSZp


#### 2022

+ Added `pressio:abs` and `pressio:rel`
+ Added support for `pressio:pw_rel`
+ Added support for MPI_COMM objects
+ Added support for cuSZ
+ Added SZx
+ Added utility to compute HDF5 cd_vals
+ Added `pressio:nthreads`

#### 2021


+ Added support for discrete choices to OptZConfig (e.g. which predictor)
+ Added support for SZ2’s multi-threaded mode
+ Added support for MGARD-GPU

### Features supported by the [ECP-EZ Project](https://szcompressor.org/) part of [The Exascale Computing Project](https://www.exascaleproject.org/) (2020-2024)

This research was also supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. The material was also supported by the U.S. Department of Energy, Office of Science, under contract DEAC02-06CH11357.

#### 2024

+ Bugfixes as needed/project closeout

#### 2023

+ Added support for masked binning and alternative mask formats to ROIBIN-SZ

#### 2022

+ initial cuFILE support
+ initial bzip2 support
+ Added smoke tests to spack

#### 2021

+ Added support for external metrics HTTP endpoints and extensible launch semantics
+ Added support for numpy from the python bindings
+ Added HDF5 filter support
+ Initial support for automatic parallelization of CPU compressors
+ Initial support for generic pw_rel support for abs compressors
+ Added support for NDZip

#### 2020

+ Created spack packages for LibPressio and OptZConfig.
+ Added support for user-defined metrics to LibPressio-Opt
+ Added MPI support to LibPressio-Opt
+ Interoperablity with PETSC
+ Initial Support for ROIBIN-SZ

2016-2019 LibPressio Development Started in 2019 and was supported by ECP starting in 2020, ECP ran from 2016-2024

### Features Supported by [NSF Using Error-Bounded Lossy Compression to Improve High-Performance Computing Systems and Applications](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1910197&HistoricalAwards=false)

This material is based upon work supported by the National Science Foundation under Grant No. 1910197.

+ **2020** Created LibPressio-Opt/FRaZ/OptZConfig
+ **2020** Support for ARC data correction mechanism

### Features Supported by the [JLESC](https://jlesc.github.io/) (2018-...)

This work was completed as part of the Joint Laboratory for Extreme Scale Computing supported by Advanced Scientific Computing Research at US DOE.

+ **2024** Integration with GFarm and TAR, Integration with TeZIP with Riken-RCCS
+ **2023** Compression for Linear Algebra with the University of Tennessee 

### Features supported by the [DOE SCGSR](https://science.osti.gov/wdts/scgsr) (2019)

This material is also based upon work supported by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists, Office of Science Graduate Student Research (SCGSR) program. The SCGSR program is administered by the Oak Ridge Institute for Science and Education (ORISE) for the DOE. ORISE is managed by ORAU under contract number DE-SC0014664. All opinions expressed in this paper are the authors and do not necessarily reflect the policies and views of DOE, ORAU, or ORISE

+ LibPressio Created with support for SZ, ZFP
+ Added support for user-defined metrics in C++
+ Python bindings added and support for MGARD
+ ImageMagick support
+ Added support for external metrics scripts
+ Added fpzip support




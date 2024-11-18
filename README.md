# LibPressio

Pressio is latin for compression.  LibPressio is a C++ library with C compatible bindings to abstract between different lossless and lossy compressors and their configurations.  It solves the problem of having to having to write separate application level code for each lossy compressor that is developed.  Instead, users write application level code using LibPressio, and the library will make the correct underlying calls to the compressors.  It provides interfaces to represent data, compressors settings, and compressors.

Documentation for the `master` branch can be [found here](https://robertu94.github.io/libpressio/)

# Using LibPressio

Example using the CLI from [`pressio-tools`](https://github.com/robertu94/pressio-tools)
We also have C, C++, Rust, Julia, and Python bindings.

```bash
pressio -i ~/git/datasets/hurricane/100x500x500/CLOUDf48.bin.f32 \
    -b compressor=sz3 -o abs=1e-4 -O all \
    -m time -m size -m error_stat -M all \
    -w /path/to/output.dec
```

The reccomended way to learn LibPressio is with self-paced [LibPressio Tutorial](https://github.com/robertu94/libpressio_tutorial).
Here you will find examples of how to use LibPressio in a series of lessons for several common languages.

You can also find a [recording of the tutorial on YouTube](https://youtu.be/hZ_dFCMxmGw).

## Getting Started

After skimming the example, LibPressio has 6 major headers that you will need to use:

Type                  | Use 
----------------------|-------------------
`pressio.h`             | Error reporting and aquiring handles to compressors
`pressio_compressor.h`  | Used to compress and decompress data, provided by plugins
`pressio_data.h`        | Represents data and associated metadata (size, type, dimentionality, memory ownership)
`pressio_options.h`     | Maps between names and values, used for options for compressors and metrics results
`pressio_metrics.h`     | A set of metrics to run while compressors run
`pressio_io.h`     | An extension header that provides methods to load or store data from/to persistent storage

All of these are included by the convience header `libpressio.h`.

You can pick up the more advanced features as you need them.

You can also find more examples in `test/` or in the [LibPressio intresting scripts collection](https://github.com/robertu94/libpressio-interesting-scripts) which catalogs intresting higher-level use cases.

## Supported Compressors and Metrics

Libpressio provides a number of builtin compressor and metrics modules.
All of these are **disabled by default**.
They can be enabled by passing the corresponding `LIBPRESSIO_HAS_*` variable to CMake.

Additionally, Libpressio is extensible.
For information on writing a compressor plugin see [Writing a Compressor Plugin](docs/WritingACompressorPlugin.md)
For information on writing a metrics plugin see [Writing a Metrics Plugin](docs/WritingAMetricsPlugin.md)


### Compressor Plugins

1st party compressors plugins can be found in [src/plugins/compressors](https://github.com/robertu94/libpressio/tree/master/src/plugins/compressors)

See the [compressor settings page](build/Compressors.md) for information on how to configure them.


### Metrics Plugins

1st party compressors plugins can be found in [src/plugins/metrics](https://github.com/robertu94/libpressio/tree/master/src/plugins/metrics)

See the [metrics results page](build/Metrics.md) for information on what they produce

### IO Plugins

1st party compressors plugins can be found in [src/plugins/io](https://github.com/robertu94/libpressio/tree/master/src/plugins/io)

See the [io settings page](build/IO.md) for information on how to configure them

# Installation

## Installing LibPressio using Spack

LibPressio can be built using [spack](https://github.com/spack/spack/).  This example will install libpressio with only the SZ3 plugin.

```bash
git clone https://github.com/spack/spack
source ./spack/share/spack/setup-env.sh
spack install libpressio+sz3
```

More information on spack can be found in the [spack documentation](https://spack.readthedocs.io/en/latest/) or [my quick start guides for systems that I use](https://robertu94.github.io/guides)

You can see the other available versions and compilation options by calling `spack info libpressio`

The following language bindings are in this repository.

+ `C` -- (default) if you need a stable interface
+ `C++` -- (default) if you want a more productive interface, or want to extend LibPressio
+ `Python` -- (`+python`; BUILD_PYTHON_WRAPPER) if you know or want to intergate Python
+ `HDF5` -- (`+hdf5+json`; LIBPRESSIO_HAS_HDF AND LIBPRESSIO_HAS_JSON) you already use HDF5

The following bindings must be installed seperately:

+ `R` -- [r-libpressio](https://github.com/robertu94/libpressio-r) if you know or want to integrate with R
+ `Bash/CLI` -- [libpressio-tools](https://github.com/robertu94/pressio-tools)  if you want to quickly prototype from the CLI

The following bindings are experimental and can be installed manually:

+ `Julia` -- [libpressio-jl](https://github.com/robertu94/LibPressio.jl) if you know or want to integrate with Julia
+ `Rust` -- [libpressio-rs](https://github.com/robertu94/libpressio-rs) if you know or want to integrate with Rust

## Doing a development build with spack

The easiest way to do a development build of libpressio is to use Spack envionments.

```bash
# one time setup: create an envionment
spack env create -d mydevenviroment
spack env activate mydevenvionment

# one time setup: tell spack to set LD_LIBRARY_PATH with the spack envionment's library paths
spack config add modules:prefix_inspections:lib64:[LD_LIBRARY_PATH]
spack config add modules:prefix_inspections:lib:[LD_LIBRARY_PATH]

# one time setup: install libpressio-tools and checkout 
# libpressio for development
spack add libpressio-tools
spack develop libpressio@git.master

# compile and install (repeat as needed)
spack install 
```


## Manual Installation

Libpressio unconditionally requires:

+ `cmake`
+ `pkg-config`
+ [`std_compat`](https://github.com/robertu94/std_compat)
+ either:
  + `gcc-4.8.5` or later
  + `clang-7.0.0` or later using either `libc++` or `libstdc++`.  Beware that system libraries may need to be recompiled with `libc++` if using `libc++`

Dependency versions and optional dependencies are documented [in the spack package](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/libpressio/package.py).


## Configuring LibPressio Manually

LibPressio uses a fairly standard CMake buildsystem.
For more information on [CMake refer to these docs](https://robertu94.github.io/learning/cmake)

The set of configuration options for LibPressio can be found using `cmake -L $BUILD_DIR`.
For information on what these settings do, see the [spack package](https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/libpressio/package.py)

# API Stability

Please refer to [docs/stability.md](docs/stability.md).

# How to Contribute

Please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of contributors, sponsors, and contribution guidelines.

# Bug Reports

Please files bugs to the Github Issues page on the CODARCode libpressio repository.

Please read this post on [how to file a good bug report](https://codingnest.com/how-to-file-a-good-bug-report/).  After reading this post, please provide the following information specific to libpressio:

+ Your OS version and distribution information, usually this can be found in `/etc/os-release`
+ the output of `cmake -L $BUILD_DIR`
+ the version of each of libpressio's dependencies listed in the README that you have installed. Where possible, please provide the commit hashes.


# Citing LibPressio

If you find LibPressio useful, please cite this paper:

```
@inproceedings{underwood2021productive,
  title={Productive and Performant Generic Lossy Data Compression with LibPressio},
  author={Underwood, Robert and Malvoso, Victoriana and Calhoun, Jon C and Di, Sheng and Cappello, Franck},
  booktitle={2021 7th International Workshop on Data Analysis and Reduction for Big Scientific Data (DRBSD-7)},
  pages={1--10},
  year={2021},
  organization={IEEE}
}
```

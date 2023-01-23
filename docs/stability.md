# Stability

As of version 1.0.0, LibPressio will follow the following API stability guidelines:

+ The functions defined in files in `./include` excluding files in the `./include/libpressio_ext/` or its subdirectories may be considered to be stable.  Furthermore, all files in this set are C compatible.
+ The functions defined in files in `./include/libpressio_ext/` are to be considered unstable.
+ The functions and modules defined in the low-level python bindings are stable (import `pressio`).
+ The functions and modules defined in the higher-level python bindings are unstable (import `libpressio`).
+ Any functions listed above, in `docs/MetricResults.md` or in `docs/MetricResults.md` as experimental are unstable
+ Any configurable that has a key `pressio:stability` with a value of `experimental` or `unstable` are unstable.  Modules that are experimental may crash or have other severe deficiencies, modules that are unstable generally will not crash, but may have options changed according to the unstable API guarantees.
+ Any configurable that has a key `pressio:stability` with a value of `stable` conforms to the LibPressio stability guarantees
+ Any configurable that has the key `pressio:stability` with a value of `external` indicates that options/configuration returned by this module are controlled by version of the external library that it depends upon and may change at any time without changing the LibPressio version number.

Stable means:

+ New APIs may be introduced with the increase of the minor version number.
+ APIs may gain additional overloads for C++ compatible interfaces with an increase in the minor version number.
+ An API may change the number or type of parameters with an increase in the major version number.
+ An API may be removed with the change of the major version number
+ New options/configuration names may appear with a increase in the minor version number
+ Existing options/configuration names may be removed or changed with an increase in the major version number

Unstable means:

+ The API or options/configuration may change for any reason with the increase of the minor version number

Additionally, the performance of functions, memory usage patterns may change for both stable and unstable code with the increase of the patch version.


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

+ Robert Underwood

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
Foundation under Grant No. 1619253 and 1910197.

We acknowledge the computing resources provided on Bebop, which is operated by
the Laboratory Computing Resource Center at Argonne National Laboratory.

This material is also based upon work supported by the U.S. Department of
Energy, Office of Science, Office of Workforce Development for Teachers and
Scientists, Office of Science Graduate Student Research (SCGSR) program. The
SCGSR program is administered by the Oak Ridge Institute for Science and
Education (ORISE) for the DOE. ORISE is managed by ORAU under contract number
DE-SC0014664. All opinions expressed in this paper are the authors and do not
necessarily reflect the policies and views of DOE, ORAU, or ORISE


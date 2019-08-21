/*
python bindings for pressio sz extensions
*/

%module pressio_sz

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include <sz/defines.h>
%}

%rename("%(strip:[SZ_])s") "";
%include <sz/defines.h>


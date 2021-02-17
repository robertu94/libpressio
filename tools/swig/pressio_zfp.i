/*
python bindings for pressio zfp extensions
*/

%module pressio_zfp

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_STRICT_BYTE_CHAR
#include <zfp.h>
%}

%rename ("zfp_exec") exec;
%rename("%(strip:[ZFP_])s") "";
%rename("%(strip:[zfp_])s") "";
%include <zfp/types.h>
%include <zfp/system.h>
%include <zfp.h>

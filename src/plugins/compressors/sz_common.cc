#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <cstdlib>

#include <sz/sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/std_compat.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "pressio_option.h"
#include "std_compat/memory.h"
#include "sz_common.h"

namespace libpressio {

int libpressio_type_to_sz_type(pressio_dtype type) {
    switch(type)
    {
      case pressio_float_dtype:  return SZ_FLOAT;
      case pressio_double_dtype: return SZ_DOUBLE;
      case pressio_uint8_dtype: return SZ_UINT8;
      case pressio_int8_dtype: return SZ_INT8;
      case pressio_uint16_dtype: return SZ_UINT16;
      case pressio_int16_dtype: return SZ_INT16;
      case pressio_uint32_dtype: return SZ_UINT32;
      case pressio_int32_dtype: return SZ_INT32;
      case pressio_uint64_dtype: return SZ_UINT64;
      case pressio_int64_dtype: return SZ_INT64;
      case pressio_byte_dtype: return SZ_INT8;
      case pressio_bool_dtype: throw std::runtime_error("unsupported dtype");
    }
    return -1;
}

sz_init_handle::sz_init_handle() {
  SZ_Init(nullptr);
}
sz_init_handle::~sz_init_handle() {
  SZ_Finalize();
}
std::mutex pressio_get_sz_lock;
std::shared_ptr<sz_init_handle> pressio_get_sz_init_handle() {
  std::lock_guard<std::mutex> guard(pressio_get_sz_lock);
  static std::weak_ptr<sz_init_handle> handle;
  std::shared_ptr<sz_init_handle> sp_handle;
  if((sp_handle = handle.lock())) {
    return sp_handle;
  } else {
    sp_handle = std::make_shared<sz_init_handle>();
    handle = sp_handle;
    return sp_handle;
  }
}

}

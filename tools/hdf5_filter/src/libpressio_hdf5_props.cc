#include "H5Ppublic.h"
#include <string>
#include <cassert>
#include <libpressio_ext/cpp/options.h>
#include "libpressio_hdf5_filter.h"

extern "C" {


    /**
     * @brief creates a compressor_id property list item
     * 
     * @param[in] name name of the property being created
     * @param[in] size size of the property in bytes
     * @param[in,out] inital_value default value of the property which will be passed to H5Pregsiter2
     * @return herr_t negative on error
     */
    herr_t libpressio_compressor_create(const char* name, size_t size,  void* inital_value) {
        if(strcmp(name,"libpressio_compressor_options") == 0) {
            if(size != sizeof(pressio_options**)) {
                return -1;
            }
            pressio_options** initial_op_value = static_cast<pressio_options**>(inital_value);
            *initial_op_value = new pressio_options();
            return 0;
        } else if(strcmp(name, "libpressio_compressor_id") == 0) {
            if(size != sizeof(std::string**)) {
                return -1;
            }
            std::string** initial_op_value = static_cast<std::string**>(inital_value);
            *initial_op_value = new std::string();
            return 0;

            return 0;
        }

        return -2;
    }

    herr_t libpressio_compressor_get(hid_t, const char* name, size_t size, void* new_value) {
        if(strcmp(name,"libpressio_compressor_options") == 0) {
            if(size != sizeof(pressio_options**)) {
                return -1;
            }
            pressio_options** initial_op_value = static_cast<pressio_options**>(new_value);
            *initial_op_value = new pressio_options(**initial_op_value);
            return 0;
        } else if(strcmp(name, "libpressio_compressor_id") == 0) {
            if(size != sizeof(std::string**)) {
                return -1;
            }
            std::string** initial_op_value = static_cast<std::string**>(new_value);
            *initial_op_value = new std::string(**initial_op_value);
            return 0;

            return 0;
        }

        return -2;
    }

    herr_t libpressio_compressor_set(hid_t, const char* name, size_t size, void* new_value) {
        if(strcmp(name,"libpressio_compressor_options") == 0) {
            if(size != sizeof(pressio_options**)) {
                return -1;
            }
            pressio_options** initial_op_value = static_cast<pressio_options**>(new_value);
            *initial_op_value = new pressio_options(**initial_op_value);
            return 0;
        } else if(strcmp(name, "libpressio_compressor_id") == 0) {
            if(size != sizeof(std::string**)) {
                return -1;
            }
            std::string** initial_op_value = static_cast<std::string**>(new_value);
            *initial_op_value = new std::string(**initial_op_value);
            return 0;

            return 0;
        }

        return -2;
    }

    herr_t libpressio_compressor_delete(hid_t, const char* name, size_t size, void* value) {
      if(strcmp(name,"libpressio_compressor_options") == 0) {
        if(size != sizeof(pressio_options**)) {
          return -1;
        }
        pressio_options** initial_op_value = static_cast<pressio_options**>(value);
        delete *initial_op_value;
        *initial_op_value = nullptr;
        return 0;
      } else if(strcmp(name, "libpressio_compressor_id") == 0) {
        if(size != sizeof(std::string*)) {
          return -1;
        }
        std::string** initial_op_value = static_cast<std::string**>(value);
        delete *initial_op_value;
        *initial_op_value = nullptr;
        return 0;

        return 0;
      }
      return 0;

    }
    herr_t libpressio_compressor_copy(const char* name, size_t size, void* new_value) {
        if(strcmp(name,"libpressio_compressor_options") == 0) {
            if(size != sizeof(pressio_options**)) {
                return -1;
            }
            pressio_options** initial_op_value = static_cast<pressio_options**>(new_value);
            *initial_op_value = new pressio_options(**initial_op_value);
            return 0;
        } else if(strcmp(name, "libpressio_compressor_id") == 0) {
            if(size != sizeof(std::string**)) {
                return -1;
            }
            std::string** initial_op_value = static_cast<std::string**>(new_value);
            *initial_op_value = new std::string(**initial_op_value);
            return 0;

            return 0;
        }

        return -2;
    }

    herr_t libpressio_compressor_options_compare(const void* value1, const void* value2, size_t) {
      auto lhs = static_cast<pressio_options const* const*>(value1);
      auto rhs = static_cast<pressio_options const* const*>(value2);
      if ((**lhs) == (**rhs)) {
        return 0;
      } else {
        return 1;
      }
    }
    herr_t libpressio_compressor_ids_compare(const void* value1, const void* value2, size_t) {
      auto lhs = static_cast<std::string const* const*>(value1);
      auto rhs = static_cast<std::string const* const*>(value2);
      if ((**lhs) == (**rhs)) {
        return 0;
      } else if((**lhs) < (**rhs)) {
        return -1;
      } else {
        return 1;
      }
    }
    herr_t libpressio_compressor_close(const char* name, size_t size, void* value) {
      if(strcmp(name,"libpressio_compressor_options") == 0) {
        if(size != sizeof(pressio_options**)) {
          return -1;
        }
        pressio_options** initial_op_value = static_cast<pressio_options**>(value);
        delete *initial_op_value;
        *initial_op_value = nullptr;
        return 0;
      } else if(strcmp(name, "libpressio_compressor_id") == 0) {
        if(size != sizeof(std::string*)) {
          return -1;
        }
        std::string** initial_op_value = static_cast<std::string**>(value);
        delete *initial_op_value;
        *initial_op_value = nullptr;
        return 0;

        return 0;
      }
      return 0;
    }

    herr_t H5Pset_libpressio(hid_t dcpl, const char* compressor_id, pressio_options const* options) {
      herr_t error = 0;

      if((error = H5Pset_filter(dcpl, H5Z_FILTER_LIBPRESSIO, 0, 0, nullptr)) < 0 ) {
          return error;
      }

      std::string** compressor_id_str = new std::string*(new std::string(compressor_id));
      pressio_options** options_ptr = new pressio_options*(new pressio_options(*options));

      if(H5Pexist(dcpl,"libpressio_compressor_options") > 0) {
        error = H5Pset(dcpl, "libpressio_compressor_options", &options);
      } else {
        error = H5Pinsert2(
            dcpl,
            "libpressio_compressor_options",
            sizeof(pressio_options**),
            options_ptr,
            libpressio_compressor_set,
            libpressio_compressor_get,
            libpressio_compressor_delete,
            libpressio_compressor_copy,
            libpressio_compressor_options_compare,
            libpressio_compressor_close
            );
      }
      if(error < 0) {
        delete *compressor_id_str;
        delete compressor_id_str;
        delete *options_ptr;
        delete options_ptr;
        return error;
      }

      if(H5Pexist(dcpl,"libpressio_compressor_id") > 0 ) {
        error =  H5Pset(dcpl, "libpressio_compressor_id", &compressor_id);
      } else {
        error = H5Pinsert2(
            dcpl,
            "libpressio_compressor_id",
            sizeof(std::string**),
            compressor_id_str,
            libpressio_compressor_set,
            libpressio_compressor_get,
            libpressio_compressor_delete,
            libpressio_compressor_copy,
            libpressio_compressor_ids_compare,
            libpressio_compressor_close
            );
      }

      if(error < 0) {
        delete *compressor_id_str;
        delete compressor_id_str;
        delete *options_ptr;
        delete options_ptr;
        return error;
      } else {
        delete compressor_id_str;
        delete options_ptr;
      }

      return error;
    }
}

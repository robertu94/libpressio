#include "libpressio_ext/cpp/serializable.h"
#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/options.h>
#include <cassert>
#include <stdexcept>


namespace {
  MPI_Datatype pressio_dtype_to_mpi(pressio_dtype dtype) {
    switch(dtype) {
      case pressio_double_dtype:
        return MPI_DOUBLE;
      case pressio_float_dtype:
        return MPI_FLOAT;
      case pressio_int8_dtype:
        return MPI_INT8_T;
      case pressio_int16_dtype:
        return MPI_INT16_T;
      case pressio_int32_dtype:
        return MPI_INT32_T;
      case pressio_int64_dtype:
        return MPI_INT64_T;
      case pressio_uint8_dtype:
        return MPI_UINT8_T;
      case pressio_uint16_dtype:
        return MPI_UINT16_T;
      case pressio_uint32_dtype:
        return MPI_UINT32_T;
      case pressio_uint64_dtype:
        return MPI_UINT64_T;
      case pressio_byte_dtype:
        return MPI_BYTE;
      case pressio_bool_dtype:
        return MPI_CXX_BOOL;
      default:
        assert(false && "missing dtype in mpi serializer");
        //return something to satisfy the compiler
        return MPI_INT;
    }
  }
}

namespace distributed {
namespace comm {
namespace serializer {

  int serializer<pressio_data>::send(pressio_data const& data, int dest, int tag, MPI_Comm comm) {
    int ret = 0;
    ret |= comm::send(data.dtype(), dest, tag, comm);
    ret |= comm::send(data.dimensions(), dest, tag, comm);
    int has_data = data.has_data();
    ret |= comm::send(has_data, dest, tag, comm);
    if(has_data) {
      ret |= MPI_Send(data.data(), data.num_elements(), pressio_dtype_to_mpi(data.dtype()), dest, tag, comm);
    }

    return ret;
  }

  int serializer<pressio_data>::recv(pressio_data& data, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int ret = 0;
    pressio_dtype dtype;
    std::vector<size_t> dims;
    int has_data = 0;
    MPI_Status local_status;
    MPI_Status* s = (status == MPI_STATUS_IGNORE) ? &local_status: status;

    ret |= comm::recv(dtype, source, tag, comm, s);
    ret |= comm::recv(dims, s->MPI_SOURCE, s->MPI_TAG, comm, s);
    ret |= comm::recv(has_data, s->MPI_SOURCE, s->MPI_TAG, comm, s);
    if(has_data) {
      //allocate owning data
      data = pressio_data::owning(dtype, dims);
      ret |= MPI_Recv(data.data(), data.num_elements(), pressio_dtype_to_mpi(dtype), s->MPI_SOURCE, s->MPI_TAG, comm, s);
    } else {
      //allocate empty data
      data = pressio_data::empty(dtype, dims);
    }

    return ret;
  }

  int serializer<pressio_data>::bcast(pressio_data& data, int root, MPI_Comm comm) {
    int ret = 0, rank;
    pressio_dtype dtype = data.dtype();
    std::vector<size_t> dims = data.dimensions();
    int has_data = data.has_data();
    ret |= comm::bcast(dtype, root, comm);
    ret |= comm::bcast(dims, root, comm);
    ret |= comm::bcast(has_data, root, comm);
    MPI_Comm_rank(comm, &rank);

    if(has_data) {
      if(rank != root) {
        data = pressio_data::owning(dtype, dims);
      }
      MPI_Bcast(data.data(), data.num_elements(), pressio_dtype_to_mpi(dtype), root, comm);
    } else if(rank != root) {
      data = pressio_data::empty(dtype, dims);
    }
    return ret;
  }

  int serializer<pressio_dtype>::bcast(pressio_dtype& dtype, int root, MPI_Comm comm) {
    int ret = 0;
    int dtype_i = dtype;
    ret |= comm::bcast(dtype_i, root, comm);
    dtype = pressio_dtype(dtype_i);
    return ret;
  }


  int serializer<pressio_dtype>::send(pressio_dtype const& dtype, int dest, int tag, MPI_Comm comm) {
    int ret = 0;
    ret |= comm::send((int32_t)dtype, dest, tag, comm);
    return ret;
  }

  int serializer<pressio_dtype>::recv(pressio_dtype& dtype, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int dtype_i, ret=0;
    ret |= comm::recv(dtype_i, source, tag, comm, status);
    dtype = pressio_dtype(dtype_i);
    return ret;
  }

  int serializer<pressio_option>::send(pressio_option const& option, int dest, int tag, MPI_Comm comm) {
    int ret = 0;
    pressio_option_type type = option.type(); 
    bool has_value = option.has_value();
    ret |= comm::send((int32_t)type, dest, tag, comm);
    ret |= comm::send((int32_t)has_value, dest, tag, comm);
    if(has_value) {
      switch(type) {
        case pressio_option_int8_type:
          {
          auto value = option.get_value<int8_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_uint8_type:
          {
          auto value = option.get_value<uint8_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_int16_type:
          {
          auto value = option.get_value<int16_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_uint16_type:
          {
          auto value = option.get_value<uint16_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_int32_type:
          {
          auto value = option.get_value<int32_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_uint32_type:
          {
          auto value = option.get_value<uint32_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_int64_type:
          {
          auto value = option.get_value<int64_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_uint64_type:
          {
          auto value = option.get_value<uint64_t>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_float_type:
          {
          auto value = option.get_value<float>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_double_type:
          {
          auto value = option.get_value<double>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_bool_type:
          {
          auto value = option.get_value<bool>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_charptr_type:
          {
          auto const& value = option.get_value<std::string>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_charptr_array_type:
          {
          auto const& value = option.get_value<std::vector<std::string>>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;
        case pressio_option_data_type:
          {
          auto const& value = option.get_value<pressio_data>();
          ret |= comm::send(value, dest, tag, comm);
          }
          break;

        case pressio_option_unset_type:
          //intensional no-op
          break;
        case pressio_option_userptr_type:
        default:
          //indicate an error
          return 1;
      }
    }
    return ret;
  }

  int serializer<pressio_option>::recv(pressio_option& option, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int ret = 0;
    MPI_Status local_status;
    MPI_Status* s = (status == MPI_STATUS_IGNORE) ? &local_status: status;
    int type, has_value;
    ret |= comm::recv(type, source, tag, comm, s);
    ret |= comm::recv(has_value, source, tag, comm, s);
    if(has_value) {
      switch(type) {
        case pressio_option_int32_type:
          {
          int value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = value;
          }
          break;
        case pressio_option_uint32_type:
          {
          unsigned int value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = value;
          }
          break;
        case pressio_option_float_type:
          {
          float value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = value;
          }
          break;
        case pressio_option_double_type:
          {
          double value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = value;
          }
          break;
        case pressio_option_bool_type:
          {
          bool value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = value;
          }
          break;
        case pressio_option_charptr_type:
          {
          std::string value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = std::move(value);
          }
          break;
        case pressio_option_charptr_array_type:
          {
          std::vector<std::string> value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = std::move(value);
          }
          break;
        case pressio_option_data_type:
          {
          pressio_data value;
          ret |= comm::recv(value, source, tag, comm, s);
          option = std::move(value);
          }
          break;

        case pressio_option_unset_type:
          //intensional no-op
          option = {};
          break;
        case pressio_option_userptr_type:
        default:
          //indicate an error
          return 1;
      }
    } else {
      option.set_type(pressio_option_type(type));
    }

    return ret;
  }

  int serializer<pressio_option>::bcast(pressio_option &option, int root, MPI_Comm comm) {
    int ret = 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    pressio_option_type type;
    int has_value;
    if (rank == root) {
      // sending
      type = option.type();
      int32_t type_i = (int32_t)type;
      has_value = option.has_value();
      ret |= comm::bcast(type_i, root, comm);
      ret |= comm::bcast(has_value, root, comm);
      if (has_value) {
        switch (type) {
        case pressio_option_int8_type: {
          auto value = option.get_value<int8_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_bool_type: {
          auto value = option.get_value<bool>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_uint8_type: {
          auto value = option.get_value<uint8_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_int16_type: {
          auto value = option.get_value<int16_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_uint16_type: {
          auto value = option.get_value<uint16_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_int32_type: {
          auto value = option.get_value<int32_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_uint32_type: {
          auto value = option.get_value<uint32_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_int64_type: {
          auto value = option.get_value<int64_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_uint64_type: {
          auto value = option.get_value<uint64_t>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_float_type: {
          auto value = option.get_value<float>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_double_type: {
          auto value = option.get_value<double>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_charptr_type: {
          std::string value = option.get_value<std::string>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_charptr_array_type: {
          std::vector<std::string> value = option.get_value<std::vector<std::string>>();
          ret |= comm::bcast(value, root, comm);
        } break;
        case pressio_option_data_type: {
          pressio_data const& value = option.get_value<pressio_data>();
          pressio_data value_ref = pressio_data::nonowning(value.dtype(), value.data(), value.dimensions());
          ret |= comm::bcast(value_ref, root, comm);
        } break;

        case pressio_option_unset_type:
          // intensional no-op
          break;
        case pressio_option_userptr_type:
        default:
          // indicate an error
          return 1;
        }
      }

    } else {
      // recv
      int type_i;
      ret |= comm::bcast(type_i, root, comm);
      ret |= comm::bcast(has_value, root, comm);
      type = pressio_option_type(type_i);

      if (has_value) {
        switch (type) {
        case pressio_option_bool_type: {
          bool value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_int8_type: {
          int8_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_uint8_type: {
          uint8_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_int16_type: {
          int16_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_uint16_type: {
          uint16_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_int32_type: {
          int32_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_uint32_type: {
          uint32_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_int64_type: {
          int64_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_uint64_type: {
          uint64_t value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_float_type: {
          float value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_double_type: {
          double value;
          ret |= comm::bcast(value, root, comm);
          option = value;
        } break;
        case pressio_option_charptr_type: {
          std::string value;
          ret |= comm::bcast(value, root, comm);
          option = std::move(value);
        } break;
        case pressio_option_charptr_array_type: {
          std::vector<std::string> value;
          ret |= comm::bcast(value, root, comm);
          option = std::move(value);
        } break;
        case pressio_option_data_type: {
          pressio_data value;
          ret |= comm::bcast(value, root, comm);
          option = std::move(value);
        } break;

        case pressio_option_unset_type:
          // intensional no-op
          option = {};
          break;
        case pressio_option_userptr_type:
        default:
          // indicate an error
          return 1;
        }
      } else { // has no-value
        option.set_type(type);
      }
    }
    return ret;
  }

  int serializer<pressio_options>::send(pressio_options const& option, int dest, int tag, MPI_Comm comm) {
    int ret = 0;
    std::vector<std::pair<std::string, pressio_option>> options_v(std::begin(option), std::end(option));
    ret |= comm::send(options_v, dest, tag, comm);
    return ret;
  }

  int serializer<pressio_options>::recv(pressio_options& options, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int ret = 0;
    std::vector<std::pair<std::string, pressio_option>> options_v;
    ret |= comm::recv(options_v, source, tag, comm, status);
    std::move(std::begin(options_v), std::end(options_v), std::inserter(options, options.end()));
    return ret;
  }

  int serializer<pressio_options>::bcast(pressio_options& options, int root, MPI_Comm comm) {
    int ret = 0;
    int rank;
    MPI_Comm_size(comm, &rank);
    std::vector<std::pair<std::string, pressio_option>> options_v;

    if(rank == root) {
      std::copy(std::begin(options), std::end(options), std::back_inserter(options_v));
    }

    comm::bcast(options_v, root, comm);

    if(rank != root) {
      std::move(std::begin(options_v), std::end(options_v), std::inserter(options, options.begin()));
    }

    return ret;
  }


}
}
}

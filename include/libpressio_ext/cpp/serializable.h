/**
 * \file
 * \brief serializers for LibDistributed
 */
#ifndef LIBPRESSIO_SERIALIZABLE_H
#define LIBPRESSIO_SERIALIZABLE_H

#include <libdistributed/libdistributed_comm.h>
#include <pressio_dtype.h>

struct pressio_data;
struct pressio_option;
struct pressio_options;


namespace distributed {
namespace comm {
namespace serializer {

/**
 * serializer for pressio_dtypes tyeps
 */
template <>
struct serializer<pressio_dtype> {
  using mpi_type = std::false_type;
  static MPI_Datatype dtype() { return MPI_INT; }
  static std::string name() { return "pressio_data"; }
  static int send(pressio_dtype const& dtype, int dest, int tag, MPI_Comm comm);
  static int recv(pressio_dtype& dtype, int source, int tag, MPI_Comm comm, MPI_Status* status);
  static int bcast(pressio_dtype& dtype, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_data tyeps
 */
template <>
struct serializer<pressio_data> {
  using mpi_type = std::false_type;
  static MPI_Datatype dtype() { return MPI_INT; }
  static std::string name() { return "pressio_data"; }
  static int send(pressio_data const& data, int dest, int tag, MPI_Comm comm);
  static int recv(pressio_data& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  static int bcast(pressio_data& data, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_option tyeps
 */
template <>
struct serializer<pressio_option> {
  using mpi_type = std::false_type;
  static MPI_Datatype dtype() { return MPI_INT; }
  static std::string name() { return "pressio_data"; }
  static int send(pressio_option const& data, int dest, int tag, MPI_Comm comm);
  static int recv(pressio_option& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  static int bcast(pressio_option& data, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_options tyeps
 */
template <>
struct serializer<pressio_options> {
  using mpi_type = std::false_type;
  static MPI_Datatype dtype() { return MPI_INT; }
  static std::string name() { return "pressio_data"; }
  static int send(pressio_options const& data, int dest, int tag, MPI_Comm comm);
  static int recv(pressio_options& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  static int bcast(pressio_options& data, int root, MPI_Comm comm);
};

}
}
}


#endif /* end of include guard: LIBPRESSIO_SERIALIZABLE_H */

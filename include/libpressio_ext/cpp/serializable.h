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
  /** is the type representable with mpi_types */
  using mpi_type = std::false_type;
  /** type MPI_Datatype when mpi_type is true */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** name of the data type */
  static std::string name() { return "pressio_data"; }
  /** 
   * send the value
   * \param[in] dtype the datatype to serialize
   * \param[in] dest the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \returns an error code
   * */
  static int send(pressio_dtype const& dtype, int dest, int tag, MPI_Comm comm);
  /** 
   * recv the value
   * \param[out] dtype the datatype to serialize
   * \param[in] source the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \param[out] status the optional status to receive
   * \returns an error code
   * */
  static int recv(pressio_dtype& dtype, int source, int tag, MPI_Comm comm, MPI_Status* status);
  /**
   * broadcast a value
   *
   * \param[in] dtype the datatype to broadcast
   * \param[in] root the root to broadcast from
   * \param[in] comm the communicator to use
   */
  static int bcast(pressio_dtype& dtype, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_data tyeps
 */
template <>
struct serializer<pressio_data> {
  /** is the type representable with mpi_types */
  using mpi_type = std::false_type;
  /** type MPI_Datatype when mpi_type is true */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** name of the data type */
  static std::string name() { return "pressio_data"; }
  /** 
   * send the value
   * \param[in] data the data to serialize
   * \param[in] dest the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \returns an error code
   * */
  static int send(pressio_data const& data, int dest, int tag, MPI_Comm comm);
  /** 
   * recv the value
   * \param[out] data the datatype to serialize
   * \param[in] source the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \param[out] status the optional status to receive
   * \returns an error code
   * */
  static int recv(pressio_data& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  /**
   * broadcast a value
   *
   * \param[in,out] data the datatype to broadcast
   * \param[in] root the root to broadcast from
   * \param[in] comm the communicator to use
   */
  static int bcast(pressio_data& data, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_option tyeps
 */
template <>
struct serializer<pressio_option> {
  /** is the type representable with mpi_types */
  using mpi_type = std::false_type;
  /** type MPI_Datatype when mpi_type is true */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** name of the data type */
  static std::string name() { return "pressio_data"; }
  /** 
   * send the value
   * \param[in] data the data to serialize
   * \param[in] dest the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \returns an error code
   * */
  static int send(pressio_option const& data, int dest, int tag, MPI_Comm comm);
  /** 
   * recv the value
   * \param[out] data the datatype to serialize
   * \param[in] source the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \param[out] status the optional status to receive
   * \returns an error code
   * */
  static int recv(pressio_option& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  /**
   * broadcast a value
   *
   * \param[in] data the datatype to broadcast
   * \param[in] root the root to broadcast from
   * \param[in] comm the communicator to use
   */
  static int bcast(pressio_option& data, int root, MPI_Comm comm);
};

/**
 * serializer for pressio_options tyeps
 */
template <>
struct serializer<pressio_options> {
  /** is the type representable with mpi_types */
  using mpi_type = std::false_type;
  /** type MPI_Datatype when mpi_type is true */
  static MPI_Datatype dtype() { return MPI_INT; }
  /** name of the data type */
  static std::string name() { return "pressio_data"; }
  /** 
   * send the value
   * \param[in] data the data to serialize
   * \param[in] dest the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \returns an error code
   * */
  static int send(pressio_options const& data, int dest, int tag, MPI_Comm comm);
  /** 
   * recv the value
   * \param[out] data the datatype to serialize
   * \param[in] source the destination to send to
   * \param[in] tag the tag to use
   * \param[in] comm the comm to serialize to
   * \param[out] status the optional status to receive
   * \returns an error code
   * */
  static int recv(pressio_options& data, int source, int tag, MPI_Comm comm, MPI_Status* status);
  /**
   * broadcast a value
   *
   * \param[in] data the datatype to broadcast
   * \param[in] root the root to broadcast from
   * \param[in] comm the communicator to use
   */
  static int bcast(pressio_options& data, int root, MPI_Comm comm);
};

}
}
}


#endif /* end of include guard: LIBPRESSIO_SERIALIZABLE_H */

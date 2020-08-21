#ifndef LIBPRESIO_DISTRIBUTED_MANAGER_H
#define LIBPRESIO_DISTRIBUTED_MANAGER_H
#include <mpi.h>
#include <libdistributed/libdistributed_work_queue.h>
#include <libdistributed/libdistributed_work_queue_options.h>
#include <libpressio_ext/compat/optional.h>
#include <utility>
#include <cassert>
#include "configurable.h"
#include "errorable.h"
#include "options.h"
#include "data.h"
#include <pressio_option.h>
#include <pressio_options.h>

/**
 * \file
 * \brief a helper class and functions for using libdistributed with libpressio
 */

/**
 * build an array of workers, tries to be fault tolerant in degenerate cases
 */
compat::optional<std::vector<size_t>> distributed_build_groups(const unsigned int size, const unsigned int n_workers_groups, const unsigned int n_masters, const unsigned int root);
int distributed_world_size();

/**
 * helper class for interacting with libdistributed
 */
class pressio_distributed_manager: public pressio_configurable, public pressio_errorable {
  public:
  static size_t unlimited;
  pressio_distributed_manager(unsigned int max_ranks_per_worker = 1, unsigned int max_masters = 1):
    groups(*distributed_build_groups(distributed_world_size(), 1, 0, 0)),
    max_masters(max_masters),
    max_ranks_per_worker(max_ranks_per_worker),
    n_workers(0),
    n_masters(0)
  {}

  template <class TaskRandomIt, class MasterFn, class WorkerFn>
  void work_queue(TaskRandomIt begin, TaskRandomIt end, WorkerFn&& workerfn, MasterFn&& masterfn) {
    distributed::queue::work_queue_options<typename distributed::queue::iterator_to_request_type<TaskRandomIt>::type> options(comm);
    options.set_root(root);
    options.set_groups(groups);
    distributed::queue::work_queue( options, begin, end, std::forward<WorkerFn>(workerfn), std::forward<MasterFn>(masterfn));
  }

  template <class T>
  int send(T const& t, int dest, int tag=0) {
    return distributed::comm::send(t, dest, tag, comm);
  }

  template <class T>
  int recv(T& t, int source, int tag=0, MPI_Status* s=nullptr) {
    return distributed::comm::recv(t, source, tag, comm, s);
  }

  template <class T>
  int bcast(T& t, int bcast_root=0) {
    return distributed::comm::bcast(t, bcast_root, comm);
  }

  int comm_size() const {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

  int comm_rank() const {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  struct pressio_options 	get_options () const override {
    pressio_options opts;
    set(opts, "distributed:mpi_comm", (void*)comm);
    if(max_masters > 1 || max_masters == 0) {
      set(opts, "distributed:n_masters", n_masters);
    }
    if(max_ranks_per_worker > 1 || max_ranks_per_worker == 0) {
      set(opts, "distributed:n_worker_groups", n_workers);
      if(max_masters > 1 || max_masters == 0) {
        set(opts, "distributed:groups", pressio_data(groups.begin(), groups.end()));
      }
    }
    return opts;
  }
  virtual int	set_options (struct pressio_options const &options) override {
    get(options, "distributed:root", &root);
    get(options, "distributed:mpi_comm", (void**)&comm);
    int size;
    MPI_Comm_size(comm, &size);

    pressio_data groups_data;
    auto workers_set = get(options, "distributed:n_worker_groups", &n_workers);
    auto masters_set = get(options, "distributed:n_masters", &n_masters);
    if(get(options, "distributed:groups", &groups_data) == pressio_options_key_set) {
      groups = groups_data.to_vector<size_t>();
      n_workers = compat::nullopt;
      n_masters = compat::nullopt;

    } else if(workers_set == pressio_options_key_set && masters_set == pressio_options_key_set) {
      n_masters = 1;
      n_workers = size - *n_masters;
      auto tmp = distributed_build_groups(size, *n_workers, *n_masters, root);
      if(tmp) {
        groups = std::move(*tmp);
      }
    } else if (workers_set == pressio_options_key_set) {
      n_masters = 1;
      auto tmp =  distributed_build_groups(size, *n_workers, *n_masters, root);
      if(tmp) {
        groups = std::move(*tmp);
      }
    } else if (masters_set == pressio_options_key_set) {
      n_workers = size - *n_masters;
      auto tmp = distributed_build_groups(size, *n_workers, *n_masters, root);
      if(tmp){
        groups = std::move(*tmp);
      }
    }
    return 0;
  }

  const char* prefix() const override { return "distributed"; }

  private:
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<size_t> groups;
  unsigned int root = 0;
  unsigned int max_masters = 1;
  unsigned int max_ranks_per_worker = 1;
  compat::optional<unsigned int> n_workers, n_masters;
};
#endif /* end of include guard: LIBPRESIO_DISTRIBUTED_MANAGER_H */

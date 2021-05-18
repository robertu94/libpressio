#include "libpressio_ext/cpp/distributed_manager.h"

size_t pressio_distributed_manager::unlimited = 0;

compat::optional<std::vector<size_t>> distributed_build_groups(const unsigned int size, const unsigned int n_workers_groups, const unsigned int n_masters, const unsigned int root) {
  //special case size == 1 because the master and worker are identical in this case
  if(size <= 1) return std::vector<size_t>{0};

  //if the user tries to allocate more processes than exist report an error
  if(size < (n_workers_groups + n_masters)) return compat::nullopt;

  //if the user ties to allocate all processes as
  if(n_workers_groups == size || n_masters == size) return compat::nullopt;

  //root must be a valid rank
  if(root >= size) return compat::nullopt;


  //the zero rank is guaranteed the code below to be a master prior to the swap below
  const size_t some_master = 0;
  //all processes are initially allocated as masters
  std::vector<size_t> groups(size, 0);

  if(n_workers_groups == 0 && n_masters == 0) {
    //allocate 1 master
    //allocate remaining workers to separate work groups
    for(unsigned int i = 1; i < size; ++i) {
      groups[i] = i;
    }
  } else if(n_workers_groups == 0) {
    //allocate n_masters master
    //all remaining processes are put into a separate work group
    for (unsigned int i = n_masters; i < size; ++i) {
      groups[i] = (i - n_masters) + 1;
    }
  } else if(n_masters == 0) {
    //allocate 1 master
    //create n_workers_groups with an equal number of processes
    //allocate all remaining processes as a master
    unsigned int last_worker = (((size-1)/n_workers_groups)*n_workers_groups) + 1;
    for(unsigned int i = 1; i < last_worker; ++i) {
      groups[i] = ((i - 1) % n_workers_groups) + 1;
    }
  } else {
    //allocate n_masters masters
    //allocate n_workers_groups, some worker groups may have uneven sizes
    for (unsigned int i = n_masters; i < size; ++i) {
      groups[i] = ((i - n_masters) % n_workers_groups) + 1;
    }
  }
  assert(groups[some_master] == 0);


  //ensure that the root process is a master
  if(groups[root] != groups[some_master]) {
    std::swap(groups[root], groups[some_master]);
  }
  return groups;
}


int distributed_world_size() {
  int size = 1, flag = 0;
  MPI_Initialized(&flag);
  if(flag) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
  return size;
}

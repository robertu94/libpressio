#include "pressio_version.h"
#if LIBPRESSIO_HAS_MPI
#include "mpi.h"
#endif
#include "gtest/gtest.h"


int main(int argc, char *argv[])
{
#if LIBPRESSIO_HAS_MPI
  int rank, size, disable_printers=1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  ::testing::InitGoogleTest(&argc, argv);
#if LIBPRESSIO_HAS_MPI
  if(rank == 0){
    int opt;
    while((opt = getopt(argc, argv, "p")) != -1) {
        switch(opt) {
          case 'p':
          disable_printers = 0;
          break;
        default:
          break;
        }
    }
  }
  MPI_Bcast(&disable_printers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //disable printers for non-root process
  if(rank != 0 and disable_printers) {
    auto&& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }
#endif

  int result = RUN_ALL_TESTS();

#if LIBPRESSIO_HAS_MPI
  int all_result=0;
  MPI_Allreduce(&result, &all_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0 && size > 1 && all_result) std::cerr << "one or more tests failed on another process, please check them" << std::endl;
  MPI_Finalize();
#else
  int all_result=result;
#endif

  return all_result;
}

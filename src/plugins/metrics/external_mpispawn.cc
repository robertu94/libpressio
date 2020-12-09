#include "external_launch.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <mpi.h>
#include "std_compat/memory.h"

struct external_mpispawn: public libpressio_launch_plugin {
extern_proc_results launch(std::string const& full_command, std::string const& workdir) const override {
      extern_proc_results results;

      std::istringstream command_stream(full_command);
      std::vector<std::string> args_mem(
          std::istream_iterator<std::string>{command_stream},
          std::istream_iterator<std::string>());
      std::vector<char*> args;
      std::transform(std::begin(args_mem), std::end(args_mem),
          std::back_inserter(args), [](std::string const& s){return const_cast<char*>(s.c_str());});
      args.push_back(nullptr);

      MPI_Info info;
      MPI_Info_create(&info);
      if(not workdir.empty()) {
        MPI_Info_set(info, "wdir", workdir.c_str());
      }

      int error_code;
      MPI_Comm child;
      MPI_Comm_spawn(args.front(), args.data()+1, 1, info, 0,  MPI_COMM_SELF, &child, &error_code); 

      int status_code;
      MPI_Recv(&status_code, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);

      int stdout_len;
      MPI_Recv(&stdout_len, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);
      results.proc_stdout.resize(stdout_len);
      MPI_Recv(&results.proc_stdout[0], stdout_len, MPI_CHAR, 0, 0, child, MPI_STATUS_IGNORE);

      int stderr_len;
      MPI_Recv(&stderr_len, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);
      results.proc_stderr.resize(stderr_len);
      MPI_Recv(&results.proc_stderr[0], stderr_len, MPI_CHAR, 0, 0, child, MPI_STATUS_IGNORE);

      MPI_Comm_free(&child);
      MPI_Info_free(&info);
      return results;
    }
  const char* prefix() const override {
    return "mpispawn";
  }

  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<external_mpispawn>(*this);
  }

};

static pressio_register launch_spawn_plugin(launch_plugins(), "mpispawn", [](){ return compat::make_unique<external_mpispawn>();});

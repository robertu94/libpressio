#include <algorithm>
#include <cmath>
#include <ftk/external/diy/mpi.hpp>
#include <ftk/filters/critical_point_tracker_regular.hh>
#include <ftk/filters/filter.hh>
#include <ftk/ftk_config.hh>
#include <ftk/hypermesh/lattice.hh>
#include <ftk/ndarray.hh>
#include <ftk/numeric/vector_assignment.hh>
#include <glob.h>
#include <iterator>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"
#include <ftk/filters/critical_point_tracker_2d_regular.hh>
#include <ftk/filters/critical_point_tracker_3d_regular.hh>
#include <string>

enum pressio_ftk_field_type {
  scalar = 0,
  vector = 1,
};

struct pressio_ftk_result {
  compat::optional<pressio_data> critical_points;
  compat::optional<size_t> num_connected_components;
};

template <class Tracker>
struct pressio_ftk_critical_point_tracker: public Tracker {
  //inherit constructors from the parent class
  using Tracker::Tracker;

  
  pressio_data get_critical_points() const {
    size_t num_points = 0;
    std::vector<double> critical_points;
    for (auto const& point : this->discrete_critical_points) {
      critical_points.insert(
          critical_points.end(),
          std::begin(point.second.x),
          std::end(point.second.x)
          );
      num_points++;
    }
    //get the extent of the array used for critical_point_t which is stored as the mapped_type of discrete_critical_points
    constexpr size_t extent = std::extent<
      decltype(
          std::declval<
            typename decltype(this->discrete_critical_points)::mapped_type
          >().x)
      >::value;

    return pressio_data::copy(
          pressio_double_dtype,
          critical_points.data(),
          {extent, num_points}
        );
        
  }

  size_t num_critial_points() const {
    return this->connected_components.size();
  }

  void set_comm(diy::mpi::communicator const& comm) {
    this->comm = comm;
  }


  void set_nthreads(int nthreads) {
#if DIY_NO_MPI
    this->nthreads = nthreads;
#else
    (void)nthreads;
#endif
  }


};

struct run_ftk_tracker {
  template <class Tracker>
  pressio_ftk_result run_tracker(Tracker& tracker, double const* begin) const {
    std::vector<size_t> origin(global_dims.size(), 0);


    tracker.set_comm(comm);
    tracker.use_accelerator(accelerator);
    tracker.set_domain(ftk::lattice(domain_latice_start, domain_latice_sizes));
    tracker.set_array_domain(ftk::lattice(origin, global_dims));
    tracker.set_input_array_partial(false);
    tracker.set_scalar_field_source(
        (input_type == pressio_ftk_field_type::scalar) ? ftk::SOURCE_GIVEN : ftk::SOURCE_DERIVED);
    tracker.set_vector_field_source(
        (input_type == pressio_ftk_field_type::vector) ? ftk::SOURCE_GIVEN : ftk::SOURCE_DERIVED);
    tracker.initialize();

    ftk::ndarray<double> source(begin, global_dims);
    switch(input_type) {
    case pressio_ftk_field_type::scalar:
      tracker.push_scalar_field_snapshot(source);
      break;
    case pressio_ftk_field_type::vector:
      tracker.push_vector_field_snapshot(source);
      break;
    }
    tracker.advance_timestep();
    tracker.finalize();

    return {tracker.get_critical_points(), tracker.num_critial_points()};

  }

  template <class T>
  pressio_ftk_result operator()(T const* begin, T const* end) const {
    std::vector<double> converted(begin, end);

    std::vector<std::string> args_s;
    args_s.emplace_back("--nthreads");
    args_s.emplace_back(std::to_string(nthreads));
    args_s.emplace_back("--accelerator");
    args_s.emplace_back("cuda");
    std::vector<const char*> argv;
    std::transform(std::begin(args_s), std::end(args_s), std::back_inserter(argv), [](std::string const& s) {return s.c_str();});
    if(global_dims.size() == 2) {
      auto tracker = pressio_ftk_critical_point_tracker<ftk::critical_point_tracker_2d_regular>();
      return run_tracker(tracker, converted.data());
    } else if(global_dims.size() == 3) {
      auto tracker = pressio_ftk_critical_point_tracker<ftk::critical_point_tracker_3d_regular>();
      return run_tracker(tracker, converted.data());
    } else {
      return {};
    }
  }
  std::vector<size_t> const& global_dims;
  std::vector<size_t> const& domain_latice_start;
  std::vector<size_t> const& domain_latice_sizes;
  pressio_ftk_field_type const &input_type;
  const size_t nthreads;
  int accelerator;
  MPI_Comm comm;
  
};

class pressio_ftk_critical_points_plugin : public libpressio_metrics_plugin {

public:
  void begin_compress(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    uncomp_results = pressio_data_for_each<pressio_ftk_result>(*input, run_ftk_tracker{
        global_dims,
        domain_latice_start,
        domain_latice_sizes,
        field_type,
        nthreads,
        accelerator,
        comm
        });
  }
  void end_decompress(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    decomp_results = pressio_data_for_each<pressio_ftk_result>(*output, run_ftk_tracker{
        global_dims,
        domain_latice_start,
        domain_latice_sizes,
        field_type,
        nthreads,
        accelerator,
        comm
        });
  }

  struct pressio_options get_metrics_results() const override
  {
    pressio_options opt;
    set(opt, "ftk_critical_points:uncompressed_critical_points", uncomp_results.critical_points);
    set(opt, "ftk_critical_points:decompressed_critical_points", decomp_results.critical_points);
    return opt;
  }

  int set_options(struct pressio_options const&) override
  {
    return 0;
  }

  pressio_options get_options() const override
  {
    pressio_options opts{};
    set(opts, "ftk_critical_points:field_type", int(field_type));
    set_type(opts, "ftk_critical_points:field_type_str", pressio_option_charptr_type);

    set(opts, "ftk_critical_points:domain_latice_start", pressio_data(domain_latice_start.begin(), domain_latice_start.end()));

    return opts;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<pressio_ftk_critical_points_plugin>(*this);
  }

  const char* prefix() const override {
    return "ftk_critical_points";
  }

private:
  pressio_ftk_result uncomp_results,decomp_results;
  std::vector<uint64_t> global_dims, domain_latice_start, domain_latice_sizes;
  pressio_ftk_field_type field_type;
  size_t nthreads;
  int accelerator = ftk::FTK_XL_NONE;
  MPI_Comm comm;
};

static pressio_register metrics_ftk_critcal_points_plugin(metrics_plugins(), "ftk_critical_points", []() {
  return compat::make_unique<pressio_ftk_critical_points_plugin>();
});

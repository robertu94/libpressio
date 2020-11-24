#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <ftk/ftk_config.hh>
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
#include "std_compat/memory.h"
#include <ftk/filters/critical_point_tracker_2d_regular.hh>
#include <ftk/filters/critical_point_tracker_3d_regular.hh>
#include <string>
#include <vector>

enum class pressio_ftk_field_type {
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

  
  pressio_data pressio_get_critical_points() const {
    auto const ftk_critical_points = this->get_critical_points();
    size_t num_points = 0;
    std::vector<double> critical_points;
    const size_t extent = this->cpdims();
    for (auto const& point : ftk_critical_points) {
      critical_points.emplace_back(point.t);
      critical_points.insert(
          critical_points.end(),
          std::begin(point.x),
          std::next(std::begin(point.x), extent)
          );
      num_points++;
    }

    return pressio_data::copy(
          pressio_double_dtype,
          critical_points.data(),
          {extent + 1, num_points}
        );
        
  }

  size_t pressio_num_critial_points() const {
    return this->connected_components.size();
  }

};

struct run_ftk_tracker {
  template <class Tracker>
  pressio_ftk_result run_tracker(Tracker& tracker, double const* begin) const {
    std::vector<size_t> origin(global_dims.size(), 0);


    tracker.set_communicator(comm);
    tracker.use_accelerator(accelerator);
    tracker.set_number_of_threads(nthreads);
    tracker.set_domain(domain);
    tracker.set_array_domain(ftk::lattice(origin, global_dims));
    tracker.set_input_array_partial(false);
    tracker.set_scalar_field_source(
        (input_type == pressio_ftk_field_type::scalar) ? ftk::SOURCE_GIVEN : ftk::SOURCE_DERIVED);
    tracker.set_vector_field_source(
        (input_type == pressio_ftk_field_type::vector) ? ftk::SOURCE_GIVEN : ftk::SOURCE_DERIVED);
    tracker.set_jacobian_field_source(ftk::SOURCE_DERIVED);
    if(input_type == pressio_ftk_field_type::scalar) {
      tracker.set_jacobian_symmetric(true);
    }
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

    return {tracker.pressio_get_critical_points(), tracker.pressio_num_critial_points()};

  }

  template <class T>
  pressio_ftk_result operator()(T const* begin, T const* end) const {
    std::vector<double> converted(begin, end);

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
  ftk::lattice const& domain;
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
    const std::vector<size_t> global_dims = input->dimensions();
    std::vector<size_t> domain_latice_start, domain_latice_sizes;
    uncomp_results = pressio_data_for_each<pressio_ftk_result>(*input, run_ftk_tracker{
        global_dims,
        ftk_domain(input),
        field_type,
        nthreads,
        accelerator,
        comm
        });
  }
  void end_decompress(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
    std::vector<size_t> global_dims = output->dimensions();
    std::vector<size_t> domain_latice_start, domain_latice_sizes;
    decomp_results = pressio_data_for_each<pressio_ftk_result>(*output, run_ftk_tracker{
        global_dims,
        ftk_domain(output),
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

  int set_options(struct pressio_options const& opts) override
  {
    {
      int tmp_i;
      std::string tmp_s;
      if(get(opts, "ftk_critical_points:field_type_str", &tmp_i) == pressio_options_key_set) {
        field_type = pressio_ftk_field_type(tmp_i);
      } else if(get(opts, "ftk_critical_points:field_type", &tmp_s) == pressio_options_key_set) {
        auto it = pressio_ftk_field_type_str.find(tmp_s);
        if(it != pressio_ftk_field_type_str.end()) {
          field_type = it->second;
        }
      }
    }

    get(opts, "ftk_critical_points:ftk_accelerator", &accelerator);
    get(opts, "ftk_critical_points:ftk_nthreads", &nthreads);
    get(opts, "ftk_critical_points:mpi_comm", (void**)&comm);

    return 0;
  }

  pressio_options get_options() const override
  {
    pressio_options opts{};
    set(opts, "ftk_critical_points:field_type", int(field_type));
    set_type(opts, "ftk_critical_points:field_type_str", pressio_option_charptr_type);

    set(opts, "ftk_critical_points:ftk_accelerator", accelerator);
    set(opts, "ftk_critical_points:ftk_nthreads", nthreads);
    set(opts, "ftk_critical_points:mpi_comm", (void*)comm);

    return opts;
  }

  pressio_options get_configuration() const override
  {
    pressio_options opts{};
    std::vector<std::string> keys;
    std::transform(
        std::begin(pressio_ftk_field_type_str),
        std::end(pressio_ftk_field_type_str),
        std::back_inserter(keys),
        [](decltype(pressio_ftk_field_type_str)::const_reference c) {
          return c.first;
        });

    set(opts, "ftk_critical_points:field_type_str", keys);

    return opts;
  }


  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<pressio_ftk_critical_points_plugin>(*this);
  }

  const char* prefix() const override {
    return "ftk_critical_points";
  }

private:
  ftk::lattice ftk_domain(pressio_data const* data) const {
    if(field_type == pressio_ftk_field_type::scalar) {
      std::vector<size_t> origin(data->num_dimensions(), 2);
      std::vector<size_t> size = data->dimensions();
      for (auto& i : size) {
        i -= 3;
      }
      return ftk::lattice(origin, size);
    } if(field_type == pressio_ftk_field_type::vector) {
      std::vector<size_t> origin(data->num_dimensions(), 1);
      std::vector<size_t> size = data->dimensions();
      for (auto& i : size) {
        i -= 2;
      }
      return ftk::lattice(origin, size);
    }
    return ftk::lattice{};
  }

  pressio_ftk_result uncomp_results,decomp_results;
  pressio_ftk_field_type field_type = pressio_ftk_field_type::scalar;
  unsigned int nthreads = 1;
  int accelerator = ftk::FTK_XL_NONE;
  MPI_Comm comm = MPI_COMM_WORLD;
  static const std::map<std::string, pressio_ftk_field_type> pressio_ftk_field_type_str ;
};

const std::map<std::string, pressio_ftk_field_type> pressio_ftk_critical_points_plugin::pressio_ftk_field_type_str {
  {"scalar", pressio_ftk_field_type::scalar},
  {"vector", pressio_ftk_field_type::vector},
};

static pressio_register metrics_ftk_critcal_points_plugin(metrics_plugins(), "ftk_critical_points", []() {
  return compat::make_unique<pressio_ftk_critical_points_plugin>();
});

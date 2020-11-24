#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <memory>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"

namespace {
  struct csv_printer {
    csv_printer(std::ofstream& outfile, size_t rows, size_t columns, const char line_delim, const char field_delim):
      rows(rows),
      columns(columns), 
      outfile(outfile),
      line_delim(line_delim),
      field_delim(field_delim)
    {}

    template<class T>
    int operator()(T* begin, T* end) {
      (void) begin;
      (void) end;
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < columns; ++col) {
          outfile << begin[row*columns + col];
          if(col != columns -1) outfile << field_delim;
          else outfile << line_delim;
        }
      }
      return 0;
    }
    const size_t rows, columns;
    std::ofstream& outfile;
    const char line_delim;
    const char field_delim;
  };
}

struct csv_io : public libpressio_io_plugin
{
  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    if(data != nullptr) pressio_data_free(data);

    std::ifstream in{path};
    if(not in) {
      bad_path(path);
      return nullptr;
    }
    size_t sizes[2] = {0,0};
    std::vector<double> builder;
    for(std::string line; std::getline(in, line, line_delim.front()); sizes[0]++) {
      if(sizes[0] < skip_rows) continue;
      std::istringstream line_ss(line);
      size_t column = 0;
      for(std::string value; std::getline(line_ss, value,field_delim.front()); ++column) {
        builder.emplace_back(std::stold(value));
      }
      sizes[1] = column;
    }
    sizes[0] -= skip_rows;
    return pressio_data_new_copy(
        pressio_double_dtype,
        builder.data(),
        2,
        sizes
        );
  }

  virtual int write_impl(struct pressio_data const* data) override{
    std::ofstream outfile{path};
    if(not outfile) return bad_path(path);
    if(pressio_data_num_dimensions(data) != 2) return invalid_dimensions();

    if(headers.size()) {
      if(pressio_data_get_dimension(data, 1) != headers.size()) {
        return invalid_headers();
      }
      for (size_t i = 0; i < headers.size(); ++i) {
        outfile << headers[i] << ((i == headers.size()-1)? line_delim.front(): field_delim.front());
      }
    }
    size_t rows = pressio_data_get_dimension(data, 0), columns = pressio_data_get_dimension(data, 1);
    pressio_data_for_each<int>(*data, csv_printer{outfile, rows, columns, line_delim.front(), field_delim.front()});

    return 0;
  }
  
  virtual struct pressio_options get_configuration_impl() const override{
    return {
      {"pressio:thread_safe",  static_cast<int>(pressio_thread_safety_multiple)}
    };
  }

  virtual int set_options_impl(struct pressio_options const& opts) override{
    get(opts, "io:path", &path);
    get(opts, "csv:headers", &headers);
    get(opts, "csv:skip_rows", &skip_rows);
    std::string tmp;
    if(get(opts, "csv:line_delim", &tmp) == pressio_options_key_set && tmp.size() == 1) {
      line_delim = std::move(tmp);
    }
    if(get(opts, "csv:field_delim", &tmp) == pressio_options_key_set && tmp.size() == 1) {
      field_delim = std::move(tmp);
    }
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set(opts,  "io:path", path);
    set(opts, "csv:headers", headers);
    set(opts, "csv:skip_rows", skip_rows);
    set(opts, "csv:line_delim", line_delim);
    set(opts, "csv:field_delim", field_delim);
    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }

  virtual const char* version() const override{
    return "0.0.1";
  }

  const char* prefix() const override {
    return "csv";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<csv_io>(*this);
  }

  private:
  int invalid_dimensions() { return set_error(1, "only 2d data is supported"); }
  int invalid_headers() { return set_error(2, "headers size must match number of columns"); }
  int bad_path(std::string const& path) { return set_error(3, "bad path " + path);}
  std::string path;
  std::vector<std::string> headers;
  std::string line_delim = "\n", field_delim = ",";
  unsigned int skip_rows = 0;
};

static pressio_register io_csv_plugin(io_plugins(), "csv",
                          []() { return compat::make_unique<csv_io>(); });

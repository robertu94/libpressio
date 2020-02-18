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

namespace {
  struct csv_printer {
    csv_printer(std::ofstream& outfile, size_t rows, size_t columns):
      rows(rows),
      columns(columns), 
      outfile(outfile)
    {}

    template<class T>
    int operator()(T* begin, T* end) {
      (void) begin;
      (void) end;
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < columns; ++col) {
          outfile << begin[row*columns + col];
          if(col != columns -1) outfile << ',';
          else outfile << '\n';
        }
      }
      return 0;
    }
    const size_t rows, columns;
    std::ofstream& outfile;
  };

  template <class Value>
  struct csv_builder {

    csv_builder()=default;
    ~csv_builder() {
      if(this->value) free(this->value);
    }
    csv_builder(csv_builder &&)=delete;
    csv_builder& operator=(csv_builder &&)=delete;
    csv_builder(csv_builder const&)=delete;
    csv_builder& operator=(csv_builder const&)=delete;
    
    void resize(size_t new_rows, size_t new_columns) {
      Value* tmp = static_cast<Value*>(malloc(sizeof(Value) * new_rows * new_columns));
      copy_resized(
          value, this->n_rows, this->n_columns,
          tmp, new_rows, new_columns,
          row_major
          );

      this->n_rows = new_rows;
      this->n_columns = new_columns;
      free(this->value);
      this->value = std::move(tmp);
    }

		void reserve(size_t rows, size_t columns, size_t)
		{
      auto new_rows = std::max(this->n_rows, rows);
      auto new_columns = std::max(this->n_columns, rows);
      if(new_rows == this->n_rows && new_columns == this->n_columns) return;
      else resize(rows, columns);
		}

		void set_entry(size_t row, size_t column, Value const& value)
		{
      if(row >= this->n_rows || column >= this->n_columns) {
        resize(row >= this->n_rows ? row + 1 : this->n_rows, column >= this->n_columns ? column + 1: this->n_columns);
      }
      if(row_major) {
        this->value[this->n_columns * row + column] = value;
      } else {
        this->value[this->n_rows * column + row] = value;
      }
		}

    Value* build() {
      auto tmp = this->value;
      this->value = nullptr;
      this->n_rows = 0;
      this->n_columns = 0;
      this->row_major = true;
      return tmp;
    }


    size_t n_rows=0, n_columns=0;
    bool row_major=true;
    private:
    Value* value=nullptr;

    /**
     * Copies or Zero initializes memory
     *
     * \param[in] old_values the memory to copy from
     * \param[in] old_rows the old number of rows
     * \param[in] old_columns the old number of columns
     * \param[in] new_values the memory to fill/allocate
     * \param[in] new_rows the new number of rows
     * \param[in] new_columns the new number of columns
     * \param[in] use_row_order -- the elements are in row-major order
     *
     * \tparam T the type of elements to copy
     */
    void copy_resized(Value* old_values, size_t old_rows, size_t old_columns,
                Value* new_values, size_t new_rows, size_t new_columns,
                bool use_row_order
                )
    {
      if(old_rows == new_rows && old_columns == new_columns) return;
      if(use_row_order)
      {
        for (size_t i = 0; i < std::min(new_rows,old_rows); ++i) {
          for (size_t j = 0; j < std::min(new_columns,old_columns); ++j) {
              new_values[i*new_columns + j] = old_values[i*old_columns + j];
          }
        }
        
        //zero initialize new values not set by previous code;
        for (size_t i = std::min(new_rows, old_rows); i < new_rows; ++i) {
          for (size_t j = std::min(new_columns, old_columns); j < new_columns; ++j) {
              new_values[i*new_columns + j] = 0;
          }
        }

      } else {
        for (size_t j = 0; j < std::min(new_columns,old_columns); ++j) {
          for (size_t i = 0; i < std::min(new_rows,old_rows); ++i) {
              new_values[i*new_rows + j] = old_values[i*old_rows + j];
          }
        }
        //zero initialize new values not set by previous code;
        for (size_t j = std::min(new_columns, old_columns); j < new_columns; ++j) {
          for (size_t i = std::min(new_rows, old_rows); i < new_rows; ++i) {
              new_values[i*new_rows + j] = 0;
          }
        }
      }

    }
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
    csv_builder<double> builder;
    for(std::string line; std::getline(in, line); sizes[0]++) {
      if(sizes[0] < skip_rows) continue;
      std::istringstream line_ss(line);
      size_t column = 0;
      for(std::string value; std::getline(line_ss, value,','); ++column) {
        builder.set_entry(sizes[0] - skip_rows, column, std::stold(value));
      }
      sizes[1] = column;
    }
    sizes[0] -= skip_rows;
    return pressio_data_new_move(
        pressio_double_dtype,
        builder.build(),
        2,
        sizes,
        pressio_data_libc_free_fn,
        nullptr
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
        outfile << headers[i] << ((i == headers.size()-1)? '\n': ',');
      }
    }
    size_t rows = pressio_data_get_dimension(data, 0), columns = pressio_data_get_dimension(data, 1);
    pressio_data_for_each<int>(*data, csv_printer{outfile, rows, columns});

    return 0;
  }
  
  virtual struct pressio_options get_configuration_impl() const override{
    return {
      {"pressio:thread_safe",  static_cast<int>(pressio_thread_safety_multiple)}
    };
  }

  virtual int set_options_impl(struct pressio_options const& opts) override{
    opts.get("io:path", &path);
    opts.get("csv:headers", &headers);
    opts.get("csv:skip_rows", &skip_rows);
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    return {
      {"io:path", path},
      {"csv:headers", headers},
      {"csv:skip_rows", skip_rows},
    };
  }

  int patch_version() const override{ 
    return 1;
  }

  virtual const char* version() const override{
    return "0.0.1";
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
  unsigned int skip_rows = 0;
};

static pressio_register X(io_plugins(), "csv",
                          []() { return compat::make_unique<csv_io>(); });

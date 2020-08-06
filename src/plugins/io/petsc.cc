#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/io/posix.h"
#include "pressio_compressor.h"
#include "pressio_data.h"
#include <mutex>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscviewer.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <vector>
#include "libpressio_ext/compat/memory.h"

namespace petsc {
std::mutex petsc_init_lock;

/** responsible for calling PETScInitalize() and PETScFinalize() if needed;
 *
 * PETSc is thread-safe except for calls to these functions which need to
 * happen before and after all other calls to the library respectively.  This
 * class ensures these functions are called
 */
class petsc_init {
public:
  petsc_init() {
    PetscInitialized(&did_init);
    if (!did_init) {
      PetscInitializeNoArguments();
    }
  }
  ~petsc_init() {
    if (!did_init) {
      PetscFinalize();
    }
  }

  static std::shared_ptr<petsc_init> get_library() {
    std::lock_guard<std::mutex> guard(petsc_init_lock);
    static std::weak_ptr<petsc_init> weak{};
    if (auto observed = weak.lock()) {
      return observed;
    } else {
      auto library = std::make_shared<petsc_init>();
      weak = library;
      return library;
    }
  }
  PetscBool did_init;
};

} // namespace

struct petsc_io : public libpressio_io_plugin {
  petsc_io(std::shared_ptr<petsc::petsc_init> &&library) : library(library) {}

  virtual struct pressio_data *read_impl(struct pressio_data *data) override {
    size_t sizes[2];
    PetscInt rows, columns;
    Mat mat = nullptr;
    pressio_data *ret;
    MatCreate(PETSC_COMM_SELF, &mat);
    MatSetType(mat, MATSEQDENSE);

    PetscViewer reader;
    PetscViewerCreate(PETSC_COMM_SELF, &reader);
    PetscViewerSetType(reader, viewer_type.c_str());
    PetscViewerPushFormat(reader, viewer_format);
    PetscViewerFileSetMode(reader, FILE_MODE_READ);
    PetscViewerFileSetName(reader, path.c_str());
    MatLoad(mat, reader);
    PetscViewerDestroy(&reader);

    //convert the matrix to SEQDENSE format if possible
    if(auto ec = MatConvert(mat, MATSEQDENSE, MAT_INPLACE_MATRIX, &mat)) {
      MatDestroy(&mat);
      petsc_error(ec);
      return nullptr;
    }

    // copy data to pressio_buffer
    MatGetSize(mat, &rows, &columns);
    sizes[0] = rows;
    sizes[1] = columns;
    PetscScalar *raw_data;
    MatDenseGetArray(mat, &raw_data);
    if (data != nullptr && pressio_data_dtype(data) == pressio_dtype_from_type<PetscScalar>() &&
        pressio_data_num_dimensions(data) == 2 && pressio_data_get_dimension(data, 0) == sizes[0] &&
        pressio_data_get_dimension(data, 1) == sizes[1] && pressio_data_has_data(data)) {
      ret = data;
      PetscScalar *data_ptr = static_cast<PetscScalar *>(pressio_data_ptr(data, nullptr));
      std::copy(raw_data, raw_data + (rows * columns), data_ptr);
    } else {
      pressio_data_free(data);
      ret = pressio_data_new_copy(pressio_dtype_from_type<PetscScalar>(), raw_data, 2, sizes);
    }
    MatDenseRestoreArray(mat, &raw_data);

    MatDestroy(&mat);

    return ret;
  }

  virtual int write_impl(struct pressio_data const *data) override {
    Mat mat = nullptr;
    if (pressio_data_num_dimensions(data) != 2)
      return unsupported_dimensions();

    size_t dims[2] = {pressio_data_get_dimension(data, 0), pressio_data_get_dimension(data, 1)};
    PetscScalar *data_ptr;
    pressio_data *tmp = nullptr;

    if (pressio_data_dtype(data) == pressio_dtype_from_type<PetscScalar>()) {
      // don't make a copy if we already have the correct type
      data_ptr = static_cast<PetscScalar *>(pressio_data_ptr(data, nullptr));
    } else {
      // otherwise convert the  type
      tmp = pressio_data_cast(data, pressio_dtype_from_type<PetscScalar>());
      data_ptr = static_cast<PetscScalar *>(pressio_data_ptr(tmp, nullptr));
    }
    MatCreateSeqDense(PETSC_COMM_SELF, dims[0], dims[1], data_ptr, &mat);

    // convert the matrix to the format specified if the matrix type matches,
    // this is a no-op in petsc else the matrix type does not match, the type is
    // converted "in-place" if the conversion fails, return an error code
    if (auto ec = MatConvert(mat, matrix_format.c_str(), MAT_INPLACE_MATRIX, &mat)) {
      MatDestroy(&mat);
      return petsc_error(ec);
    }

    PetscViewer writer;
    PetscViewerCreate(PETSC_COMM_SELF, &writer);
    PetscViewerSetType(writer, viewer_type.c_str());
    PetscViewerPushFormat(writer, viewer_format);
    PetscViewerFileSetMode(writer, FILE_MODE_WRITE);
    PetscViewerFileSetName(writer, path.c_str());

    MatView(mat, writer);

    PetscViewerDestroy(&writer);
    MatDestroy(&mat);
    pressio_data_free(tmp);
    return 0;
  }
  virtual struct pressio_options get_configuration_impl() const override {
    return {{"pressio:thread_safe", static_cast<int>(pressio_thread_safety_single)}};
  }

  virtual int set_options_impl(struct pressio_options const &opts) override {
    std::string viewer_format_str;
    get(opts, "io:path", &path);
    get(opts, "petsc:matrix_format", &matrix_format);
    get(opts, "petsc:viewer_type", &viewer_type);
    get(opts, "petsc:viewer_format", &viewer_format_str);

    auto viewer_format_it = map_name_to_viewer_format.find(viewer_format_str);
    if (viewer_format_it != map_name_to_viewer_format.end()) {
      viewer_format = viewer_format_it->second;
    } else {
      return invalid_viewer_format(viewer_format_str);
    }

    return 0;
  }
  virtual struct pressio_options get_options_impl() const override {
    pressio_options opts;
    set(opts, "io:path", path);
    set(opts, "petsc:matrix_format", matrix_format);
    set(opts, "petsc:viewer_format", map_viewer_format_to_name.at(viewer_format));
    set(opts, "petsc:viewer_type", viewer_type);
    return opts;
  }

  int patch_version() const override { return 1; }
  virtual const char *version() const override { return "0.0.1"; }
  const char *prefix() const override { return "petsc"; }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<petsc_io>(*this);
  }

private:
  int unsupported_dimensions() { return set_error(1, "only 2d data is supported"); }
  int petsc_error(PetscErrorCode ec) {
    const char *msg;
    PetscErrorMessage(ec, &msg, nullptr);
    return set_error(2, msg);
  }
  int invalid_viewer_format(std::string const &viewer_format_str) {
    using namespace std::literals;
    return set_error(3, "invalid viewer format: "s + viewer_format_str);
  }
  std::string path;
  std::string matrix_format = MATSEQDENSE;
  std::string viewer_type = PETSCVIEWERBINARY;
  PetscViewerFormat viewer_format = PETSC_VIEWER_DEFAULT;
  std::shared_ptr<petsc::petsc_init> library;
  static const std::map<std::string, PetscViewerFormat> map_name_to_viewer_format;
  static const std::map<PetscViewerFormat, std::string> map_viewer_format_to_name;
};

static std::vector<std::pair<std::string, PetscViewerFormat>> petsc_viewer_formats = []() {
  std::vector<std::pair<std::string, PetscViewerFormat>> vec;
  for (auto i : {PETSC_VIEWER_DEFAULT,
                 PETSC_VIEWER_ASCII_MATLAB,
                 PETSC_VIEWER_ASCII_MATHEMATICA,
                 PETSC_VIEWER_ASCII_IMPL,
                 PETSC_VIEWER_ASCII_INFO,
                 PETSC_VIEWER_ASCII_INFO_DETAIL,
                 PETSC_VIEWER_ASCII_COMMON,
                 PETSC_VIEWER_ASCII_SYMMODU,
                 PETSC_VIEWER_ASCII_INDEX,
                 PETSC_VIEWER_ASCII_DENSE,
                 PETSC_VIEWER_ASCII_MATRIXMARKET,
                 PETSC_VIEWER_ASCII_VTK,
                 PETSC_VIEWER_ASCII_VTK_CELL,
                 PETSC_VIEWER_ASCII_VTK_COORDS,
                 PETSC_VIEWER_ASCII_PCICE,
                 PETSC_VIEWER_ASCII_PYTHON,
                 PETSC_VIEWER_ASCII_FACTOR_INFO,
                 PETSC_VIEWER_ASCII_LATEX,
                 PETSC_VIEWER_ASCII_XML,
                 PETSC_VIEWER_ASCII_GLVIS,
                 PETSC_VIEWER_ASCII_CSV,
                 PETSC_VIEWER_DRAW_BASIC,
                 PETSC_VIEWER_DRAW_LG,
                 PETSC_VIEWER_DRAW_LG_XRANGE,
                 PETSC_VIEWER_DRAW_CONTOUR,
                 PETSC_VIEWER_DRAW_PORTS,
                 PETSC_VIEWER_VTK_VTS,
                 PETSC_VIEWER_VTK_VTR,
                 PETSC_VIEWER_VTK_VTU,
                 PETSC_VIEWER_BINARY_MATLAB,
                 PETSC_VIEWER_NATIVE,
                 PETSC_VIEWER_HDF5_PETSC,
                 PETSC_VIEWER_HDF5_VIZ,
                 PETSC_VIEWER_HDF5_XDMF,
                 PETSC_VIEWER_HDF5_MAT,
                 PETSC_VIEWER_NOFORMAT,
                 PETSC_VIEWER_LOAD_BALANCE}) {
    vec.emplace_back(PetscViewerFormats[i], i);
  }
  return vec;
}();

const std::map<PetscViewerFormat, std::string> petsc_io::map_viewer_format_to_name = []() {
  std::map<PetscViewerFormat, std::string> map;
  for (auto const &pair : petsc_viewer_formats) {
    map.emplace(pair.second, pair.first);
  }
  return map;
}();

const std::map<std::string, PetscViewerFormat> petsc_io::map_name_to_viewer_format = []() {
  std::map<std::string, PetscViewerFormat> map;
  for (auto const &pair : petsc_viewer_formats) {
    map.emplace(pair.first, pair.second);
  }
  return map;
}();

static pressio_register io_petsc_plugin(io_plugins(), "petsc", []() {
  return compat::make_unique<petsc_io>(petsc::petsc_init::get_library());
});

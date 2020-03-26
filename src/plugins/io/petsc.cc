#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <type_traits>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/io.h"

#include <petscsys.h>
#include <petscmat.h>

namespace {
std::mutex petsc_init_lock;

/**
 * responsible for calling PETScInitalize() and PETScFinalize() if needed;
 *
 * PETSc is thread-safe except for calls to these functions which need to happen before
 * and after all other calls to the library respectively.  This class ensures these functions are called
 */
class petsc_init {
  public:
  petsc_init() {
    PetscInitialized(&did_init);
    if(!did_init) {
      PetscInitializeNoArguments();
    }
  }
  ~petsc_init() {
    if(!did_init) {
      PetscFinalize();
    }
  }

  static std::shared_ptr<petsc_init> get_library() {
    std::lock_guard<std::mutex> guard(petsc_init_lock);
    static std::weak_ptr<petsc_init> weak{};
    if(auto observed = weak.lock())
    {
      return observed;
    } else {
      auto library = std::make_shared<petsc_init>();
      weak = library;
      return library;
    }
  }
  PetscBool did_init;
};


}

struct petsc_io : public libpressio_io_plugin {
  petsc_io(std::shared_ptr<petsc_init>&& library): library(library) {}

  virtual struct pressio_data* read_impl(struct pressio_data* data) override {
    size_t sizes[2];
    PetscInt rows, columns;
    Mat mat = nullptr;
    pressio_data* ret;
    MatCreate(PETSC_COMM_SELF, &mat);
    MatSetType(mat, MATSEQDENSE);

    PetscViewer reader;
    PetscViewerCreate(PETSC_COMM_SELF, &reader);
    PetscViewerSetType(reader, PETSCVIEWERBINARY);
    PetscViewerPushFormat(reader, PETSC_VIEWER_NATIVE);
    PetscViewerFileSetMode(reader, FILE_MODE_READ);
    PetscViewerFileSetName(reader, path.c_str());
    MatLoad(mat, reader);

    //copy data to pressio_buffer
    MatGetSize(mat, &rows, &columns);
    sizes[0] = rows;
    sizes[1] = columns;
    PetscScalar* raw_data;
    MatDenseGetArray(mat, &raw_data);
    if(data != nullptr \
        && pressio_data_dtype(data) == pressio_dtype_from_type<PetscScalar>()
        && pressio_data_num_dimensions(data) == 2
        && pressio_data_get_dimension(data, 0) == sizes[0]
        && pressio_data_get_dimension(data, 1) == sizes[1]
        && pressio_data_has_data(data)
        ) {
      ret = data;
      PetscScalar* data_ptr = static_cast<PetscScalar*>(pressio_data_ptr(data, nullptr));
      std::copy(raw_data, raw_data + (rows*columns), data_ptr);
    } else {
      pressio_data_free(data);
      ret = pressio_data_new_copy(
          pressio_dtype_from_type<PetscScalar>(),
          raw_data,
          2,
          sizes);
    }
    MatDenseRestoreArray(mat, &raw_data);

    PetscViewerDestroy(&reader);
    MatDestroy(&mat);

    return ret;
  }

  virtual int write_impl(struct pressio_data const* data) override{
    Mat mat = nullptr;
    size_t dims[2] = {
      pressio_data_get_dimension(data, 0),
      pressio_data_get_dimension(data, 1)
    };
    PetscScalar* data_ptr;
    pressio_data* tmp = nullptr;

    if(pressio_data_num_dimensions(data) != 2) return unsupported_dimensions();
    
    if(pressio_data_dtype(data) == pressio_dtype_from_type<PetscScalar>()) {
      //don't make a copy if we already have the correct type
      data_ptr = static_cast<PetscScalar*>(pressio_data_ptr(data, nullptr));
    } else {
      //otherwise convert the  type
      tmp = pressio_data_cast(data, pressio_dtype_from_type<PetscScalar>());
      data_ptr = static_cast<PetscScalar*>(pressio_data_ptr(tmp, nullptr));
    }
    MatCreateSeqDense(PETSC_COMM_SELF, dims[0], dims[1], data_ptr, &mat);

    PetscViewer writer;
    PetscViewerCreate(PETSC_COMM_SELF, &writer);
    PetscViewerSetType(writer, PETSCVIEWERBINARY);
    PetscViewerPushFormat(writer, PETSC_VIEWER_NATIVE);
    PetscViewerFileSetMode(writer, FILE_MODE_WRITE);
    PetscViewerFileSetName(writer, path.c_str());
    MatView(mat, writer);

    PetscViewerDestroy(&writer);
    MatDestroy(&mat);
    pressio_data_free(tmp);
    return 0;
  }
  virtual struct pressio_options get_configuration_impl() const override{
    return {
      {"pressio:thread_safe",  static_cast<int>(pressio_thread_safety_single)}
    };
  }

  virtual int set_options_impl(struct pressio_options const& opts) override{
    get(opts, "io:path", &path);
    return 0;
  }
  virtual struct pressio_options get_options_impl() const override{
    pressio_options opts;
    set(opts, "io:path", path);
    return opts;
  }

  int patch_version() const override{ 
    return 1;
  }
  virtual const char* version() const override{
    return "0.0.1";
  }
  const char* prefix() const override {
    return "petsc";
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<petsc_io>(*this);
  }

  private:
  int unsupported_dimensions() {
    return set_error(1, "only 2d data is supported");
  }
  std::string path;
  std::shared_ptr<petsc_init> library; 
};

static pressio_register X(io_plugins(), "petsc", [](){ return compat::make_unique<petsc_io>(petsc_init::get_library()); });

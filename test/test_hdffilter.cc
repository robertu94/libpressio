#include <gtest/gtest.h>
#include <std_compat/utility.h>
#include <std_compat/functional.h>
#include <libpressio_ext/cpp/options.h>
#include <functional>
#include <numeric>
#include <hdf5.h>
#include "libpressio_hdf5_filter.h"


struct cleanup {
    template <class Func>
    cleanup(Func&& f): f(f) {}
    cleanup(cleanup && rhs) : f(compat::exchange(rhs.f, []{})) {}
    cleanup& operator=(cleanup && rhs) {
        if (this == &rhs) return *this;
        f = compat::exchange(rhs.f, []{});
        return *this;
    }
    cleanup& operator=(cleanup const&)=delete;
    cleanup(cleanup const&)=delete;
    cleanup() : f([]{}) {}
    ~cleanup() {
        f();
    }

    std::function<void()> f;
};



TEST(hdffilter, filter_runs) {
    const hsize_t dims[] = {30,30,30};
    const auto vector_len = std::accumulate(std::begin(dims), std::end(dims), 1, compat::multiplies<>{});
    std::vector<float> v(vector_len);
    std::vector<float> v2(vector_len);

    if(H5Zregister(H5Z_LIBPRESSIO) < 0) {
        FAIL() << "failed to register plugin";
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if(fapl < 0) {
        FAIL() << "failed to create fapl";
    }
    cleanup fapl_cleanup ([&fapl]{ H5Pclose(fapl);});
    if(H5Pset_fapl_core(fapl, /*increment*/1024, /*backingstore*/false)) {
        FAIL() << "failed to set core driver";
    }

    hid_t file = H5Fcreate("hdf5filter", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if(file < 0) {
        FAIL() << "failed to create in-memory file";
    }
    cleanup file_cleanup([&file]{H5Fclose(file);});

    hid_t space = H5Screate_simple(3, dims, nullptr);
    if(space < 0) {
        FAIL() << "failed to create dataspace";
    }
    cleanup space_cleanup([&space]{H5Sclose(space);});

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if(dcpl < 0) {
        FAIL() << "failed to create dataset creation property list";
    }
    cleanup cleanup_dcpl([&dcpl]{H5Pclose(dcpl);});

    if(H5Pset_chunk(dcpl, 3, dims) < 0 ) {
        FAIL() << "failure to configure chunk size";
    }

    const char* compressor_id = "sz";
    pressio_options options {
        {"pressio:abs", 1e-4},
    };

    if(H5Pset_libpressio(dcpl, compressor_id, &options) < 0) {
      FAIL() << "failed to configure libpressio settings";
    }


    {
      hid_t dataset = H5Dcreate(file, "testing", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
      if(dataset < 0) {
          FAIL() << "failed to create dataset";
      }
      cleanup dataset_cleanup([&dataset]{ H5Dclose(dataset);});

      std::iota(v.begin(), v.end(), 1.0f);

      H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    }

    //force a flush to ensure the data is written and we don't just get a cached copy
    {
      hid_t dataset = H5Dopen(file, "testing", H5P_DEFAULT);
      if(dataset < 0) {
          FAIL() << "failed to open dataset";
      }
      cleanup dataset_cleanup([&dataset]{ H5Dclose(dataset);});

      H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, v2.data());
    }

    for (size_t i = 0; i < vector_len; ++i) {
      EXPECT_LE(std::abs(v[i] - v2[i]) , 1e-4);
    }

    
}

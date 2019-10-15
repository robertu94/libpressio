/**
 * \file 
 * \brief IO functions for simple HDF5 files
 */

#ifdef __cplusplus
extern "C" {
#endif 

#ifndef PRESSIO_HDF5_IO
#define PRESSIO_HDF5_IO

struct pressio_data;

/**
 * reads files that use simple datatypes and dataspaces into a pressio_data
 *
 * The following are not supported:
 * + complex datatypes or dataspaces
 * + unlimited dimension datasets.
 *
 * \param[in] file_name name of the file to read
 * \param[in] dataset_name name of the dataset to read from the file
 *
 * \returns a pressio_data structure or nullptr if there was an error
 */
struct pressio_data*
pressio_io_data_path_h5read(const char* file_name, const char* dataset_name);

/**
 * writes files that use simple datatypes and dataspaces from a pressio_data
 *
 * if the file already exists, it will be opened.  if the file does not exist, it will be created.
 * if the dataset already exists, it will be overwritten.  if the dataset does not exist, it will be created.
 *
 * \param[in] data the data to be written
 * \param[in] file_name name of the file to written
 * \param[in] dataset_name name of the dataset to written to the file
 *
 * \returns a 0 if successful, nonzero on error
 */
int
pressio_io_data_path_h5write(struct pressio_data const* data, const char* file_name, const char* dataset_name);

#endif

#ifdef __cplusplus
}
#endif

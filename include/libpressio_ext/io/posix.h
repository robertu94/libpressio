#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif 

  struct pressio_data;

  /** \file
   *  \brief IO functions for POSIX compatible systems
   *
   *  NOTE attempting to read files written by different machines has undefined behavior!!
   */

#ifndef PRESSIO_POSIX_IO
#define PRESSIO_POSIX_IO

  /** read in a file from a POSIX FILE pointer
   *
   *  NOTE attempting to read files written by different machines has undefined behavior!!
   *
   * \param[in,out] dims description of the dimension of the data if it known or NULL.  
   *            If dims is NULL, the remainder of the file will be read and the size of the resulting pointer will be a 1d pressio_byte_dtype of appropriate length
   *            If dims is not null, The user SHOULD assume that the memory pointed to by this pointer has been "moved" in a C++11 sense and the user MUST not rely on its contents.
   *            The implementation MAY return this pointer and reuse the underlying space if pressio_data_has_data(dims) returns true.
   * \param[in,out] in_file an file open for reading seeked to the beginning of the data to read in.
   * \returns a pointer to a (possibly new) pressio data structure.
   *
   */
  struct pressio_data* pressio_io_data_fread(struct pressio_data* dims, FILE* in_file); 


  /** read in a file from a POSIX file descriptor
   *
   *  NOTE attempting to read files written by different machines has undefined behavior!!
   *
   * \param[in,out] dims description of the dimension of the data if it known or NULL.  
   *            If dims is NULL, the remainder of the file will be read and the size of the resulting pointer will be a 1d pressio_byte_dtype of appropriate length
   *            If dims is not null, The user SHOULD assume that the memory pointed to by this pointer has been "moved" in a C++11 sense and the user MUST not rely on its contents.
   *            The implementation MAY return this pointer and reuse the underlying space if pressio_data_has_data(ptr) returns true.
   * \param[in,out] in_filedes an file open for reading seeked to the beginning of the data to read in.  If dims is not null, only pressio_data_get_bytes(dims) bytes are read.
   * \returns a pointer to a (possibly new) pressio data structure.
   *
   */
  struct pressio_data* pressio_io_data_read(struct pressio_data* dims, int in_filedes); 

  /** read in a file at a specifed location on the file-system
   *
   *  NOTE attempting to read files written by different machines has undefined behavior!!
   *
   * \param[in,out] dims description of the dimension of the data if it known or NULL.  
   *            If dims is NULL, the remainder of the file will be read and the size of the resulting pointer will be a 1d pressio_byte_dtype of appropriate length
   *            If dims is not null, The user SHOULD assume that the memory pointed to by this pointer has been "moved" in a C++11 sense and the user MUST not rely on its contents.
   *            The implementation MAY return this pointer and reuse the underlying space if pressio_data_has_data(dims) returns true.
   * \param[in,out] out_file an file open for reading seeked to the beginning of the data to read in.
   * \returns a pointer to a (possibly new) pressio data structure.
   *
   */
  struct pressio_data* pressio_io_data_path_read(struct pressio_data* dims, const char* out_file); 

  /** write in a file to the specified POSIX FILE pointer
   *
   * \param[in] data the data to be written.
   * \param[in,out] out_file 
   * \returns the number of bytes written
   *
   */
  size_t pressio_io_data_fwrite(struct pressio_data const* data, FILE* out_file); 

  /** write in a file to the specified POSIX file descriptor
   *
   * \param[in] data the data to be written.
   * \param[in,out] out_filedes the file descriptor to write to
   * \returns the number of bytes written
   *
   */
  size_t pressio_io_data_write(struct pressio_data const* data, int out_filedes); 

  /** write in a file to the specified path on the file-system
   *
   * \param[in] data the data to be written.
   * \param[in,out] path the path to write to
   * \returns the number of bytes written
   *
   */
  size_t pressio_io_data_path_write(struct pressio_data const* data, const char* path); 


#endif /*PRESSIO_POSIX_IO*/

#ifdef __cplusplus
}
#endif

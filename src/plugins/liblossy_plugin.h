#ifndef LIBLOSSY_COMPRESSOR_IMPL_H
#define LIBLOSSY_COMPRESSOR_IMPL_H
#include <string>

/*!\file 
 * \brief an implementation-only header for adding plugins to liblossy
 */

struct lossy_data;
struct lossy_options;

class liblossy_plugin {
  public:
  /** compressor should free their global memory in the destructor */
  virtual ~liblossy_plugin();
  /** get a set of options available for the compressor.
   *
   * The compressor should set a value if they have been set as default
   * The compressor should set a "reset" value if they are required to be set, but don't have a meaningful default
   *
   * \see lossy_compressor_get_options for the semantics this function should obey
   * \see lossy_options_clear to set a "reset" value
   * \see lossy_options_set_integer to set an integer value
   * \see lossy_options_set_double to set an double value
   * \see lossy_options_set_userptr to set an data value, include an \c include/ext/\<my_plugin\>.h to define the structure used
   * \see lossy_options_set_string to set a string value
   */
  virtual struct lossy_options* get_options() const=0;
  /** sets a set of options for the compressor 
   * \param[in] options to set for configuration of the compressor
   * \see lossy_compressor_set_options for the semantics this function should obey
   */
  virtual int set_options(struct lossy_options const* options)=0;
  /** compresses a lossy_data buffer
   * \see lossy_compressor_compress for the semantics this function should obey
   */
  virtual int compress(struct lossy_data* input, struct lossy_data** output)=0;
  /** decompress a lossy_data buffer
   * \see lossy_compressor_decompress for the semantics this function should obey
   */
  virtual int decompress(struct lossy_data* input, struct lossy_data** output)=0;
  /** get a version string for the compressor
   * \see lossy_compressor_version for the semantics this function should obey
   */
  virtual const char* version() const=0;
  /** get the major version, default version returns 0
   * \see lossy_compressor_major_version for the semantics this function should obey
   */
  virtual int major_version() const;
  /** get the minor version, default version returns 0
   * \see lossy_compressor_minor_version for the semantics this function should obey
   */
  virtual int minor_version() const;
  /** get the patch version, default version returns 0
   * \see lossy_compressor_patch_version for the semantics this function should obey
   */
  virtual int patch_version() const;

  /** get the error message for the last error
   * \returns an implementation specific c-style error message for the last error
   */
  const char* error_msg() const;
  /** get the error code for the last error
   * \returns an implementation specific integer error code for the last error, 0 is reserved for no error
   */
  int error_code() const;
  protected:
  /**
   * Should be used by implementing plug-ins to provide error codes
   * \param[in] code a implementation specific code for the last error
   * \param[in] msg a implementation specific message for the last error
   */
  void set_error(int code, std::string const& msg);

  private:
  struct {
    int code;
    std::string msg;
  } error;
};

#endif

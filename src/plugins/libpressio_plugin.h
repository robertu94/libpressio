#ifndef LIBPRESSIO_COMPRESSOR_IMPL_H
#define LIBPRESSIO_COMPRESSOR_IMPL_H
#include <string>

/*!\file 
 * \brief an implementation-only header for adding plugins to libpressio
 */

struct pressio_data;
struct pressio_options;

class libpressio_plugin {
  public:
  /** compressor should free their global memory in the destructor */
  virtual ~libpressio_plugin();
  /** get a set of options available for the compressor.
   *
   * The compressor should set a value if they have been set as default
   * The compressor should set a "reset" value if they are required to be set, but don't have a meaningful default
   *
   * \see pressio_compressor_get_options for the semantics this function should obey
   * \see pressio_options_clear to set a "reset" value
   * \see pressio_options_set_integer to set an integer value
   * \see pressio_options_set_double to set an double value
   * \see pressio_options_set_userptr to set an data value, include an \c include/ext/\<my_plugin\>.h to define the structure used
   * \see pressio_options_set_string to set a string value
   */
  virtual struct pressio_options* get_options() const=0;
  /** sets a set of options for the compressor 
   * \param[in] options to set for configuration of the compressor
   * \see pressio_compressor_set_options for the semantics this function should obey
   */
  virtual int set_options(struct pressio_options const* options)=0;
  /** compresses a pressio_data buffer
   * \see pressio_compressor_compress for the semantics this function should obey
   */
  virtual int compress(struct pressio_data* input, struct pressio_data** output)=0;
  /** decompress a pressio_data buffer
   * \see pressio_compressor_decompress for the semantics this function should obey
   */
  virtual int decompress(struct pressio_data* input, struct pressio_data** output)=0;
  /** get a version string for the compressor
   * \see pressio_compressor_version for the semantics this function should obey
   */
  virtual const char* version() const=0;
  /** get the major version, default version returns 0
   * \see pressio_compressor_major_version for the semantics this function should obey
   */
  virtual int major_version() const;
  /** get the minor version, default version returns 0
   * \see pressio_compressor_minor_version for the semantics this function should obey
   */
  virtual int minor_version() const;
  /** get the patch version, default version returns 0
   * \see pressio_compressor_patch_version for the semantics this function should obey
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

  /** checks for extra arguments set for the compressor.
   * the default verison just checks for unknown options passed in.
   * overriding implementations SHOULD call this version and return any errors it provides FIRST.
   *
   * \see pressio_compressor_check_options for semantics this function obeys
   * */
  virtual int check_options(struct pressio_options const*);

  protected:
  /**
   * Should be used by implementing plug-ins to provide error codes
   * \param[in] code a implementation specific code for the last error
   * \param[in] msg a implementation specific message for the last error
   * \returns the value passed to code
   */
  int set_error(int code, std::string const& msg);

  private:
  struct {
    int code;
    std::string msg;
  } error;
};

#endif

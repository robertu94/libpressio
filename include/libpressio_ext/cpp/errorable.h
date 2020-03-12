#ifndef LIBPRESSIO_ERRORS_H
#define LIBPRESSIO_ERRORS_H
#include <string>

/**
 * class that indicates that a pressio object handles errors
 */
class pressio_errorable {
  public:

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
   * \returns the value passed to code
   */
  int set_error(int code, std::string const& msg);

  private:
  struct {
    int code;
    std::string msg;
  } error;
};

#endif /* end of include guard: LIBPRESSIO_ERRORS_H */

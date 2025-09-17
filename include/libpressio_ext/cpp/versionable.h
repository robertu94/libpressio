#ifndef LIBPRESSIO_VERSIONABLE_H
#define LIBPRESSIO_VERSIONABLE_H
#include <cstdint>

/** \file 
 * \brief interface for versionable types */

namespace libpressio {
/**
 * interface for objects which have version information in libpressio
 */
class pressio_versionable {
  public:
  /**
   * virtual default destructor
   */
  virtual ~pressio_versionable()=default;

  /** get a implementation specific version string for the io module
   * \see pressio_io_version for the semantics this function should obey
   */
  virtual const char* version() const=0;

  /** a LibPressio version specific version number incremented whenever the major version does not reliably
   * reflect backwards incompatability, it is never reset to a lower value.  It defaults to 0.
   * \see pressio_io_major_version for the semantics this function should obey
   */
  virtual uint64_t epoch_version() const;

  /** get the major version, default version returns 0
   * \see pressio_io_major_version for the semantics this function should obey
   */
  virtual int major_version() const;
  /** get the minor version, default version returns 0
   * \see pressio_io_minor_version for the semantics this function should obey
   */
  virtual int minor_version() const;
  /** get the patch version, default version returns 0
   * \see pressio_io_patch_version for the semantics this function should obey
   */
  virtual int patch_version() const;
};
}

#endif /* end of include guard: VERSIONABLE_H_AMFI0DYW */

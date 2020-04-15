#ifndef LIBPRESSIO_VERSIONABLE_H
#define LIBPRESSIO_VERSIONABLE_H

/** \file 
 * \brief interface for versionable types */

/**
 * interface for objects which have version information in libpressio
 */
class pressio_versionable {
  public:
  /** get a implementation specific version string for the io module
   * \see pressio_io_version for the semantics this function should obey
   */
  virtual const char* version() const=0;
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


#endif /* end of include guard: VERSIONABLE_H_AMFI0DYW */

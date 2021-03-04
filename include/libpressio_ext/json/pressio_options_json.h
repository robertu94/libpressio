#ifdef __cplusplus
extern "C" {
#endif

  /**
   * \file 
   * \brief JSON serialization and deserialization for options structures
   */

  /**
   * Converts JSON to a pressio_options structure.  Intended for serialization and deserialization and the format
   * is unstable.  boolean entries in the JSON are not supported.
   *
   * \param[in] library optional argument, if the parse fails, the error message is stored in the library object
   * \param[in] json c-style string containing json to be deserialized to a pressio_options type
   * \return a pressio_options that needs to be freed with pressio_options_free
   */
 struct pressio_options* pressio_options_new_json(struct pressio* library, const char* json);

  /**
   * Converts pressio_options to JSON structure.  Intended for serialization and deserialization and the format
   * is unstable.  Entries of type pressio_option_userptr_type are omitted.
   *
   * \param[in] library optional argument, if the parse fails, the error message is stored in the library object
   * \param[in] options the options structure to serialize
   * \return a string that needs to be freed with free()
   */
 char* pressio_options_to_json(struct pressio* library, struct pressio_options const* options);
  
#ifdef __cplusplus
}
#endif

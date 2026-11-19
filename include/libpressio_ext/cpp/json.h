#ifndef LIBPRESSIO_CPP_JSON_H
#define LIBPRESSIO_CPP_JSON_H
/**
 * \file
 * \brief C++ convert libpressio data objects to/from JSON 
 */

#include <nlohmann/json_fwd.hpp>
struct pressio_data;
struct pressio_option;
struct pressio_options;

/** Serialize a `pressio_data` object into JSON. */
void to_json(nlohmann::json& j, pressio_data const& data);
/** Serialize a `pressio_option` object into JSON. */
void to_json(nlohmann::json& j, pressio_option const& option);
/** Serialize a `pressio_options` object into JSON. */
void to_json(nlohmann::json& j, pressio_options const& options);
/** Deserialize a `pressio_data` object from JSON. */
void from_json(nlohmann::json const& j, pressio_data& data);
/** Deserialize a `pressio_option` object from JSON. */
void from_json(nlohmann::json const& j, pressio_option& option);
/** Deserialize a `pressio_options` object from JSON. */
void from_json(nlohmann::json const& j, pressio_options& options);
#endif /* end of include guard: LIBPRESSIO_CPP_JSON_H */

#include <nlohmann/json_fwd.hpp>
struct pressio_data;
struct pressio_option;
struct pressio_options;

void to_json(nlohmann::json& j, pressio_data const& data);
void to_json(nlohmann::json& j, pressio_option const& option);
void to_json(nlohmann::json& j, pressio_options const& options);
void from_json(nlohmann::json const& j, pressio_data& data);
void from_json(nlohmann::json const& j, pressio_option& option);
void from_json(nlohmann::json const& j, pressio_options& options);

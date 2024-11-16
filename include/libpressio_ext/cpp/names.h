#include <vector>
#include <string>
#include "std_compat/string_view.h"

namespace libpressio { namespace names {

std::vector<compat::string_view> search(compat::string_view const& value);
std::string format_name(std::string const& name, std::string const& key);

}}

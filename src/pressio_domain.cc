#include "libpressio_ext/cpp/domain.h"

void pressio_domain::set_name(std::string const& name) {
    this->name = name;
    set_name_impl(name);
}

bool is_accessible(pressio_domain const& lhs, pressio_domain const& rhs) {
    if (lhs == rhs) return true;
    std::vector<std::string> domains;
    auto lhs_props = lhs.get_configuration();
    if(get(lhs_props, "domains:accessible", domains) == domain_option_key_status::key_set) {
        if(std::find(domains.begin(), domains.end(), rhs.prefix()) != domains.end()) {
            return true;
        }
    }
    auto rhs_props = rhs.get_configuration();
    if(get(rhs_props, "domains:accessible", domains) == domain_option_key_status::key_set) {
        if(std::find(domains.begin(), domains.end(), lhs.prefix()) != domains.end()) {
            return true;
        }
    }
    return false;
}

namespace detail {
std::vector<compat::string_view> search(compat::string_view const& value) {
  std::vector<compat::string_view> order;
  //normalize the string
  auto size = value.size();
  const unsigned int has_leading_slash = !value.empty() && value.front() == '/';
  const unsigned int has_training_slash = !value.empty() && value.back() == '/';
  if(size >= 2) {
    if(has_leading_slash) --size;
    if(has_training_slash) --size;
  } else if(size == 1){
    if(has_leading_slash) --size;
  }
  const auto normalized = value.substr(has_leading_slash, size);

  
  //special case empty string
  if(normalized.empty()) {
    order.emplace_back("");
    return order;
  }

  order.reserve(std::count(std::begin(normalized), std::end(normalized), '/') + 2);
  bool done = false;
  auto len = std::string::npos;
  while(!done) {
    order.emplace_back(normalized.substr(0, len));

    len = normalized.rfind('/', len - 1);
    done = (len == std::string::npos);
  }
  order.emplace_back("");

  return order;
}
}

#include <iomanip>
#include <sstream>
#include <algorithm>
#include "libpressio_ext/cpp/domain.h"

namespace libpressio { namespace domains { namespace detail {
    template<class... Ts>
    struct overloaded : Ts... { using Ts::operator()...; };
    // explicit deduction guide (not needed as of C++20)
    template<class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;
}

std::string to_string(domain_option const& op) {
    using s = std::ostream&;
    std::stringstream ss;
    std::visit(
            libpressio::domains::detail::overloaded {
                [&](std::monostate const&) -> s {
                    return ss << "<null>";
                },
                [&](std::any const&) -> s {
                    return ss << "<ptr>";
                },
                [&](std::string const& x) -> s {
                    return ss << std::quoted(x);
                },
                [&](std::vector<std::string> const& x) -> s {
                    ss << '{';
                    for(auto i : x) {
                        ss << std::quoted(i) << ',';
                    }
                    ss << '}';
                    return ss;
                },
                [&](auto const& x) -> s {
                    return ss << x;
                },
            }, op);
    return ss.str();
}
std::string to_string(domain_options const& op) {
    std::stringstream ss;
    for(auto const& e: op) {
        ss << e.first << '=' << to_string(e.second);
    }
    return ss.str();
}

void pressio_domain::set_name(std::string const& name) {
    this->name = name;
    set_name_impl(name);
}

bool is_accessible(pressio_domain const& lhs, pressio_domain const& rhs) {
    if (lhs.domain_id() == rhs.domain_id()) return true;
    std::vector<std::string> lhs_domains;
    auto lhs_props = lhs.get_configuration();
    if(get(lhs_props, "domains:accessible", lhs_domains) == domain_option_key_status::key_set) {
        if(std::find(lhs_domains.begin(), lhs_domains.end(), rhs.domain_id()) != lhs_domains.end()) {
            return true;
        }
    }
    std::vector<std::string> rhs_domains;
    auto rhs_props = rhs.get_configuration();
    if(get(rhs_props, "domains:accessible", rhs_domains) == domain_option_key_status::key_set) {
        if(std::find(rhs_domains.begin(), rhs_domains.end(), lhs.domain_id()) != rhs_domains.end()) {
            return true;
        }
    }

    return false;
}
}}
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

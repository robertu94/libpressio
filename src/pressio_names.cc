#include "std_compat/std_compat.h"
#include "std_compat/string_view.h"
#include <vector>

namespace libpressio { namespace names {

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

std::string format_name(std::string const& name, std::string const& key) {
    if(name == "") return key;
    else return '/' + name + ':' + key;
  }

} }

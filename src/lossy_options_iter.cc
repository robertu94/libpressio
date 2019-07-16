#include <map>
#include <string>
#include "lossy_options_impl.h"

struct lossy_options_iter {
  private:
  using iterator_t =  std::map<std::string, option_type>::iterator;
  public:
  lossy_options_iter(iterator_t current, iterator_t end): current(current), end(end) {}
  std::map<std::string, option_type>::iterator current;
  std::map<std::string, option_type>::iterator end;
};

extern "C" {
struct lossy_options_iter* lossy_options_get_iter(struct lossy_options* options) {
  return new struct lossy_options_iter(options->begin(), options->end());
}
bool lossy_options_iter_has_value(struct lossy_options_iter* iter) {
  return iter->current != iter->end;
}
char const* lossy_options_iter_get_key(struct lossy_options_iter* const iter) {
  return iter->current->first.c_str();
}
struct lossy_option* lossy_options_iter_get_value(struct lossy_options_iter* const iter) {
  return new lossy_option(iter->current->second);
}
void lossy_options_iter_free(struct lossy_options_iter* const iter) {
  delete iter;
}

void lossy_options_iter_next(struct lossy_options_iter* iter) {
  iter->current++;
}
}

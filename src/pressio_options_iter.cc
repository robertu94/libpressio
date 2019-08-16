#include <map>
#include <string>
#include "libpressio_ext/cpp/options.h"

struct pressio_options_iter {
  private:
  using iterator_t =  std::map<std::string, pressio_option>::const_iterator;
  public:
  pressio_options_iter(iterator_t current, iterator_t end): current(current), end(end) {}
  iterator_t current;
  iterator_t end;
};

extern "C" {
struct pressio_options_iter* pressio_options_get_iter(struct pressio_options const* options) {
  return new struct pressio_options_iter(options->begin(), options->end());
}
bool pressio_options_iter_has_value(struct pressio_options_iter* iter) {
  return iter->current != iter->end;
}
char const* pressio_options_iter_get_key(struct pressio_options_iter* const iter) {
  return iter->current->first.c_str();
}
struct pressio_option* pressio_options_iter_get_value(struct pressio_options_iter* const iter) {
  return new pressio_option(iter->current->second);
}
void pressio_options_iter_free(struct pressio_options_iter* const iter) {
  delete iter;
}

void pressio_options_iter_next(struct pressio_options_iter* iter) {
  iter->current++;
}
}

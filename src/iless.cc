#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <cstdlib>

#include "iless.h"

bool iless::operator()(std::string lhs, std::string rhs) const {
	std::transform(std::begin(lhs), std::end(lhs), std::begin(lhs), [](unsigned char c){return std::tolower(c);});
	std::transform(std::begin(rhs), std::end(rhs), std::begin(rhs), [](unsigned char c){return std::tolower(c);});
	return lhs < rhs;
}

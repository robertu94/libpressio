#include <string>

struct iless {
    bool operator()(std::string lhs, std::string rhs) const;
};  

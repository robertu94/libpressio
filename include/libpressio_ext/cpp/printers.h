#include <string>
#include <ostream>
#include "options.h"

/** \file 
 *  \brief C++ stream compatible IO functions
 *  */

/**
 * human readable debugging IO function for pressio_option_type, the format is unspecified
 * \param[in] out the stream to write to
 * \param[in] type the type to print
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, enum pressio_option_type type) {
    switch(type)
    {
      case pressio_option_int32_type:
        return out << "int32";
      case pressio_option_uint32_type:
        return out << "uint32";
      case pressio_option_float_type:
        return out << "float";
      case pressio_option_double_type:
        return out << "double";
      case pressio_option_charptr_type:
        return out << "char*";
      case pressio_option_userptr_type:
        return out << "void*";
      case pressio_option_unset_type:
        return out << "unset";
    }
}

/**
 * human readable debugging IO function for pressio_option, the format is unspecified
 * \param[in] out the stream to write to
 * \param[in] option the option to print
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, pressio_option const& option) {
  auto type = option.type();
  out << "<" << type << "> = ";
  if (option.has_value()) {
    switch(type)
    {
      case pressio_option_int32_type:
        return out << option.get_value<int>();
      case pressio_option_uint32_type:
        return out << option.get_value<unsigned int>();
      case pressio_option_float_type:
        return out << option.get_value<float>();
      case pressio_option_double_type:
        return out << option.get_value<double>();
      case pressio_option_charptr_type:
        return out << "\"" << option.get_value<std::string>() << "\"";
      case pressio_option_userptr_type:
        return out << option.get_value<void*>();
      case pressio_option_unset_type:
        return out << "<empty>";
    }
  } else { 
    return out << "<empty>" ;
  }
  

}

/**
 * human readable debugging IO function for pressio_options, the format is unspecified
 * \param[in] out the stream to write to
 * \param[in] options the options to print
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, pressio_options const& options) {
  for (auto const& option : options) {
    out << option.first << " " << option.second << "\n" ;
  }
  return out;
}


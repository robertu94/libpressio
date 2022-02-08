#ifndef LIBPRESSIO_PRINTERS_H
#define LIBPRESSIO_PRINTERS_H



#include <iterator>
#include <string>
#include <ostream>
#include "options.h"
#include "data.h"
#include "compressor.h"
#include "pressio_dtype.h"
#include "pressio_compressor.h"

/** \file 
 *  \brief C++ stream compatible IO functions
 *  */

/**
 * print the thread safety entries
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, pressio_thread_safety const& safety) {
  switch(safety) {
    case pressio_thread_safety_single: return out << "single";
    case pressio_thread_safety_multiple: return out << "multiple";
    case pressio_thread_safety_serialized: return out << "serialized";
    default: throw std::logic_error("invalid thread safety");
  }
}


/**
 * print the thread safety entries
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, pressio_options_key_status const& safety) {
  switch(safety) {
    case pressio_options_key_set: return out << "set";
    case pressio_options_key_exists: return out << "exists";
    case pressio_options_key_does_not_exist: return out << "does not exist";
    default: throw std::logic_error("invalid key status");
  }
}

/**
 * print the elements of an iterable container
 * \internal
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
struct print_elements_helper{
  template <class T>
    /**
     * prints the underlying container
     */
  int operator()(T const* begin,  T const* end) {
    out << '[';
    std::copy( begin, end, std::ostream_iterator<T>(out, ", "));
    out << ']';
    return 0;
  }
  /** the stream to output to */
  std::basic_ostream<CharT, Traits>& out;
};

/**
 * helper to construct the print_elements_helper for printing iterable collections
 * \param[in] out the stream to write to
 * \internal
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
print_elements_helper<CharT, Traits> print_elements(std::basic_ostream<CharT, Traits> &out) {
  return print_elements_helper<CharT, Traits>{out};
}

/**
 * human readable debugging IO function for pressio_data, the format is unspecified
 * \param[in] out  the output stream to print to
 * \param[in] data the data struct to print 
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, pressio_data const& data) {
  
  out << "data{ type=" << data.dtype();
  out << " dims={";
  for (auto const& dim : data.dimensions()) {
    out << dim << ", ";  
  }
  out << "} has_data=" ;
  if(data.has_data() &&  data.num_elements() < 10 ) {
    pressio_data_for_each<int>(data, print_elements(out));
    out << '}';
  } else {
    out << std::boolalpha << data.has_data() << "}";
  }

  return out;
}

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
      case pressio_option_int8_type:
        return out << "int8";
      case pressio_option_uint8_type:
        return out << "uint8";
      case pressio_option_int16_type:
        return out << "int16";
      case pressio_option_uint16_type:
        return out << "uint16";
      case pressio_option_int32_type:
        return out << "int32";
      case pressio_option_uint32_type:
        return out << "uint32";
      case pressio_option_int64_type:
        return out << "int64";
      case pressio_option_uint64_type:
        return out << "uint64";
      case pressio_option_float_type:
        return out << "float";
      case pressio_option_double_type:
        return out << "double";
      case pressio_option_bool_type:
        return out << "bool";
      case pressio_option_charptr_type:
        return out << "char*";
      case pressio_option_userptr_type:
        return out << "void*";
      case pressio_option_charptr_array_type:
        return out << "char*[]";
      case pressio_option_data_type:
        return out << "data";
      default:
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
      case pressio_option_int8_type:
        return out << int16_t{option.get_value<int8_t>()};
      case pressio_option_uint8_type:
        return out << uint16_t{option.get_value<uint8_t>()};
      case pressio_option_bool_type:
        return out << std::boolalpha << option.get_value<bool>();
      case pressio_option_int16_type:
        return out << option.get_value<int16_t>();
      case pressio_option_uint16_type:
        return out << option.get_value<uint16_t>();
      case pressio_option_int32_type:
        return out << option.get_value<int32_t>();
      case pressio_option_uint32_type:
        return out << option.get_value<uint32_t>();
      case pressio_option_int64_type:
        return out << option.get_value<int64_t>();
      case pressio_option_uint64_type:
        return out << option.get_value<uint64_t>();
      case pressio_option_float_type:
        return out << option.get_value<float>();
      case pressio_option_double_type:
        return out << option.get_value<double>();
      case pressio_option_charptr_type:
        return out << "\"" << option.get_value<std::string>() << "\"";
      case pressio_option_userptr_type:
        return out << option.get_value<void*>();
      case pressio_option_charptr_array_type:
        {
          auto const& values = option.get_value<std::vector<std::string>>();
          out << "{";
          for (auto const& value : values) {
            out << value << ", ";
          }
          return out << "}";
        }
      case pressio_option_data_type:
        return out << option.get_value<pressio_data>();
      case pressio_option_unset_type:
        return out << "<empty>";
      default:
        return out << "<unsupported>";
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

/**
 * human readable debugging IO function for pressio_dtype, the format is unspecified
 * \param[in] out the stream to write to
 * \param[in] type the type to print
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, enum pressio_dtype type)
{
  switch (type) {
    case pressio_double_dtype:
      return out << "double";
    case pressio_float_dtype:
      return out << "float";
    case pressio_uint8_dtype:
      return out << "uint8_t";
    case pressio_uint16_dtype:
      return out << "uint16_t";
    case pressio_uint32_dtype:
      return out << "uint32_t";
    case pressio_uint64_dtype:
      return out << "uint64_t";
    case pressio_int8_dtype:
      return out << "int8_t";
    case pressio_int16_dtype:
      return out << "int16_t";
    case pressio_int32_dtype:
      return out << "int32_t";
    case pressio_int64_dtype:
      return out << "int64_t";
    case pressio_bool_dtype:
      return out << "bool";
    default:
    case pressio_byte_dtype:
      return out << "byte";
  }
}

/**
 * human readable debugging IO function for libpressio_compressor_plugin, the format is unspecified
 * \param[in] out the stream to write to
 * \param[in] comp the type to print
 */
template <class CharT = char, class Traits = std::char_traits<CharT>>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, libpressio_compressor_plugin const& comp) {
	return out << comp.prefix();
}

#endif /* end of include guard: LIBPRESSIO_PRINTERS_H */

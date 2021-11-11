#include <memory>
#include <sstream>
#include <cassert>
#include <mutex>

#include <Magick++.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "pressio_option.h"
#include "pressio_version.h"
#include "std_compat/memory.h"

namespace libpressio { namespace magick{

std::mutex magick_init_lock;

/**
 * responsible for calling Magick::IntializeMagic and Magick::TerminateMagick;
 *
 * ImageMagick is thread-safe except for calls to these functions which need to happen before
 * and after all other calls to the library respectively.  This class ensures these functions are called
 */
class magick_init {
  public:
    magick_init() { Magick::InitializeMagick(NULL); }
    ~magick_init() { Magick::TerminateMagick();}
  static std::shared_ptr<magick_init> get_library() {
    std::lock_guard<std::mutex> guard(magick_init_lock);
    static std::weak_ptr<magick_init> weak{};
    if(auto observed = weak.lock())
    {
      return observed;
    } else {
      auto library = std::make_shared<magick_init>();
      weak = library;
      return library;
    }
  }
};


class magick_plugin: public libpressio_compressor_plugin {
  public:
  magick_plugin(std::shared_ptr<magick_init>&& init): init(init) {
  };


  struct pressio_options get_documentation_impl() const override {
    struct pressio_options options;
    set(options, "pressio:description", R"(ImageMagick is a robust library that preforms a wide array of image
      compression and manipulation. Only a fraction of its api is exposed. More information on ImageMagick
      can be found on its [project homepage](https://imagemagick.org/))");
    set(options, "magick:samples_magick", "the string for the samples format");
    set(options, "magick:compressed_magick", "the ImageMagick magick format");
    set(options, "magick:quality", "the quality parameter for lossy images");
    return options;
  }


  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "unstable");
    return options;
  }

  struct pressio_options get_options_impl() const override {
    struct pressio_options options;
    set(options, "magick:samples_magick", samples_magick);
    set(options, "magick:compressed_magick", compressed_magick);
    set(options, "magick:quality", quality);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override {
    get(options, "magick:samples_magick", &samples_magick);
    get(options, "magick:compressed_magick", &compressed_magick);
    get(options, "magick:quality", &quality);
    return 0;
  }

  /**
   * convert from pressio_data to image, store image in blob-format
   */
  int compress_impl(const pressio_data *input, struct pressio_data* output) override {
    auto apply_compression = [this](Magick::Image& image) {
      image.magick(this->compressed_magick);
      image.quality(this->quality);
    };
    if(input->num_dimensions() == 2) {
      //convert a single image
      //convert data to samples
      auto samples_image = data_to_samples_image(*input);
      if(not samples_image) {
        return invalid_type(input->dtype());
      }
      //convert samples image to compressed image
      apply_compression(*samples_image);


      //convert compressed image to blob and store output
      Magick::Blob blob;
      samples_image->write(&blob);
      *output = pressio_data::copy(pressio_byte_dtype, blob.data(),{blob.length()});
      return 0;
    } else if (input->num_dimensions() == 3) {
      //convert a set of images
      
      if(not Magick::CoderInfo(compressed_magick).isMultiFrame()) return invalid_codec(compressed_magick);

      std::vector<Magick::Image> images(input->get_dimension(0));
      size_t image_size_in_bytes = input->get_dimension(1) * input->get_dimension(2) * pressio_dtype_size(input->dtype());
      uint8_t const* pos = reinterpret_cast<uint8_t const*>(input->data());
      auto storage_type = data_type_to_storage_type(*input);
      if(storage_type == Magick::UndefinedPixel) return invalid_type(input->dtype());
      for (auto& image : images) {
        image.read(
            input->get_dimension(1),
            input->get_dimension(2),
            samples_magick,
            storage_type,
            pos
            ); 
        apply_compression(image);
        pos += image_size_in_bytes;
      }

      Magick::Blob output_blob;
      writeImages(images.begin(), images.end(), &output_blob);
      *output = pressio_data::copy(pressio_byte_dtype, output_blob.data(), {output_blob.length()});

      return 0;
    } else {
      return invalid_dimensions(input->num_dimensions());
    }
  }

  /**
   * convert from image blob to pressio_data, store data in output
   */
  int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
    if(output->num_dimensions() == 2) {
      //we are inputting a single image
      //convert data to compressed image
      Magick::Image samples = data_blob_to_image(input, output);
      auto storage_type = data_type_to_storage_type(*output);
      if(storage_type == Magick::UndefinedPixel) return invalid_type(output->dtype());

      if(output->has_data()) {
        std::vector<size_t> output_dims;
        output_dims.push_back(samples.rows());
        output_dims.push_back(samples.columns());
        *output = pressio_data::owning(output->dtype(), output_dims);
      }
      samples.write(
          0,0, /*start at the origin*/
          samples.rows(),
          samples.columns(),
          samples_magick,
          storage_type,
          output->data()
          );

      return 0;
    } else if (output->num_dimensions() == 3) {
      //we are inputting a set of images
      std::vector<Magick::Image> images;
      Magick::Blob input_blob(input->data(), input->size_in_bytes());
      Magick::readImages(&images, input_blob);
      auto storage_type = data_type_to_storage_type(*output);
      if(storage_type == Magick::UndefinedPixel) return invalid_type(output->dtype());

      if(output->has_data()) {
        std::vector<size_t> output_dims;
        output_dims.push_back(images.size());
        output_dims.push_back(images[0].rows());
        output_dims.push_back(images[0].columns());
        *output = pressio_data::owning(output->dtype(), output_dims);
      }

      size_t image_size_in_bytes = images[0].rows() * images[0].columns() * pressio_dtype_size(output->dtype());
      uint8_t* pos = reinterpret_cast<uint8_t*>(output->data());
      for (auto& image : images) {
        image.write(0,0,
            image.rows(),
            image.columns(),
            samples_magick,
            storage_type,
            pos
            );
        pos += image_size_in_bytes;
      }

      return 0;
    } else {
      return invalid_dimensions(output->num_dimensions());
    }
  }

  int major_version() const override {
    size_t version_number;
    MagickCore::GetMagickVersion(&version_number); 
    return version_number;
  }
  int minor_version() const override {
    return 0;
  }
  int patch_version() const override {
    return 0;
  }

  const char* version() const override {
    size_t unused;
    return MagickCore::GetMagickVersion(&unused); 
  }

  const char* prefix() const override {
    return "magick";
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<magick_plugin>(*this);
  }

  private:

  int invalid_dimensions(size_t n_dims) {
    std::stringstream ss;
    ss << "invalid dimensions for image magick plugin " << n_dims;
    return set_error(1, ss.str());
  }

  int invalid_type(pressio_dtype dtype) {
    std::stringstream ss;
    ss << "the passed dtype is not supportted by them magick plugin " << dtype;
    return set_error(2, ss.str());
  }

  int invalid_input() {
    return set_error(3, "floating point input must be scaled between 0 and 1 or 0..MaxRBG for other types, ");
  }

  int invalid_codec(std::string const& codec) {
    std::stringstream ss;
    ss << "the requested codec cannot output 3d images: " << codec;
    return set_error(4, ss.str());
  }

  /**
   * converts from an arbitrary image to samples and then to a pressio_data structure by converting
   * it to an image of samples
   */
  Magick::Image data_blob_to_image(pressio_data const* input, pressio_data const* output) {
    Magick::Blob blob(input->data(), input->size_in_bytes());
    Magick::Geometry geometry(output->get_dimension(0), output->get_dimension(1));
    Magick::Image image(blob, geometry, compressed_magick);
    return image;
  }

  /**
   * converts from a pressio_data structure to an image by treating it as
   * an image using samples.
   */
  compat::optional<Magick::Image> data_to_samples_image(pressio_data const& data) const {
    auto storage_type = data_type_to_storage_type(data);
    if(storage_type == Magick::UndefinedPixel) return {};
    Magick::Image image(
        data.get_dimension(0),
        data.get_dimension(1),
        samples_magick,
        storage_type,
        data.data()
        );
    return image;
  }

  Magick::StorageType data_type_to_storage_type(pressio_data const& data) const {
    if(data.dtype() == pressio_double_dtype) return Magick::DoublePixel;
    if(data.dtype() == pressio_float_dtype) return Magick::FloatPixel;
    if(data.dtype() == pressio_byte_dtype) return Magick::CharPixel;
    if(pressio_dtype_is_signed(data.dtype())) {
      size_t dtype_size = pressio_dtype_size(data.dtype());
      if(dtype_size == sizeof(unsigned short)) return Magick::ShortPixel;
#if LIBPRESSIO_COMPAT_HAS_IMAGEMAGICK_LONGLONG
      if(dtype_size == sizeof(unsigned int)) return Magick::LongPixel;
      if(dtype_size == sizeof(Magick::MagickSizeType)) return Magick::LongLongPixel;
#else
      if(dtype_size == sizeof(unsigned int)) return Magick::IntegerPixel;
#endif
      if(dtype_size == sizeof(Magick::Quantum)) return Magick::QuantumPixel;
    }
    return Magick::UndefinedPixel;

  };

  unsigned int quality = 100;
  std::string samples_magick = "G";
  std::string compressed_magick = "JPEG";
  std::shared_ptr<magick::magick_init> init;
};

static pressio_register comprssor_magick_plugin(compressor_plugins(), "magick", [](){
    return std::make_shared<magick_plugin>(magick::magick_init::get_library()); 
});
} }

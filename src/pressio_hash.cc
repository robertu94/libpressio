#include "libpressio_ext/hash/libpressio_hash.h"
#include <libpressio_ext/cpp/options.h>
#include <libpressio_ext/cpp/pressio.h>
#include <cstddef>
#include <openssl/evp.h>

class hasher {
    public:
    uint8_t* bytes(size_t* size){
        if(size==nullptr) {
            return nullptr;
        }

        auto ossl_b = (uint8_t*)OPENSSL_malloc(EVP_MD_size(EVP_md5()));
        if(ossl_b == nullptr) {
            return nullptr;
        }

        unsigned int digest_len;
        EVP_DigestFinal_ex(mdctx, (unsigned char*)ossl_b, &digest_len);
        *size = digest_len;

        EVP_MD_CTX_free(mdctx);
        mdctx=nullptr;
        uint8_t* b = (uint8_t*)malloc(*size);
        memcpy((char*)b, (char*)ossl_b, *size);
        OPENSSL_free(ossl_b);
        return b;
    }
    void operator()(std::string const& s) {
        if(EVP_DigestUpdate(mdctx, s.data(), s.size()) != 1) {
            throw std::runtime_error("failed to update hash");
        }
    }
    void operator()(pressio_option_type& s) {
        if(EVP_DigestUpdate(mdctx, &s, sizeof(s)) != 1) {
            throw std::runtime_error("failed to update hash");
        }
    }
    void operator()(bool b) {
        if(EVP_DigestUpdate(mdctx, &b, sizeof(b)) != 1) {
            throw std::runtime_error("failed to update hash");
        }
    }
    void operator()(const void* s, size_t size) {
        if(EVP_DigestUpdate(mdctx, s, size) != 1) {
            throw std::runtime_error("failed to update hash");
        }
    }
    ~hasher() {
        if(mdctx != nullptr) {
            EVP_MD_CTX_free(mdctx);
            mdctx=nullptr;
        }
    }

    hasher() : mdctx(EVP_MD_CTX_new()) {
        if(mdctx == nullptr) {
            throw std::runtime_error("failed to create OpenSSL EVP");
        }
        if(EVP_DigestInit_ex(mdctx, EVP_md5(), NULL) != 1) {
            throw std::runtime_error("failed to inialize OpenSSL EVP");
        }
    }
    hasher(hasher const&)=delete;
    hasher(hasher &&)=delete;
    hasher& operator=(hasher const&)=delete;
    hasher& operator=(hasher &&)=delete;

    private:
    EVP_MD_CTX* mdctx = nullptr;
};

template <class Func>
static uint8_t* libpressio_options_hashimpl(struct pressio* library, struct pressio_options const* options, size_t* output_size, Func func) {
    try {
        if(options == nullptr) throw std::runtime_error("options is nullptr");
        hasher h;

        for (auto const& i : *options) {
           func(h, i);
        }

        return h.bytes(output_size);
    } catch(std::runtime_error const& ex) { 
        if(library) {
            library->set_error(1, ex.what());
        }
        return nullptr;
    }
}

extern "C" {
    uint8_t* libpressio_options_hashkeys(struct pressio* library, struct pressio_options const* options, size_t* output_size) {
        return libpressio_options_hashimpl(library, options, output_size, [](hasher& h, pressio_options::value_type const& i){
                h(i.first);
        });
    }

    uint8_t* libpressio_options_hashentries(struct pressio* library, struct pressio_options const* options, size_t* output_size) {
        return libpressio_options_hashimpl(library, options, output_size, [](hasher& h, pressio_options::value_type const& i){
                h(i.first);
                h(i.second.type());
                h(i.second.has_value());
                if(i.second.has_value()) {
                    switch(i.second.type()) {
                        case pressio_option_userptr_type:
                        case pressio_option_unset_type:
                            return;
                        case pressio_option_data_type:
                            {
                                auto& data = i.second.get_value<pressio_data>();
                                h(data.dtype());
                                h(data.dimensions().data(), data.dimensions().size()*sizeof(size_t));
                                h(data.data(), data.size_in_bytes());
                                return;
                            }
                        case pressio_option_uint8_type:
                            {
                                auto& d = i.second.get_value<uint8_t>();
                                h(&d, sizeof(uint8_t));
                                return;
                            }
                        case pressio_option_uint16_type:
                            {
                                auto& d = i.second.get_value<uint16_t>();
                                h(&d, sizeof(uint16_t));
                                return;
                            }
                        case pressio_option_uint32_type:
                            {
                                auto& d = i.second.get_value<uint32_t>();
                                h(&d, sizeof(uint32_t));
                                return;
                            }
                        case pressio_option_uint64_type:
                            {
                                auto& d = i.second.get_value<uint64_t>();
                                h(&d, sizeof(uint64_t));
                                return;
                            }
                        case pressio_option_int8_type:
                            {
                                auto& d = i.second.get_value<int8_t>();
                                h(&d, sizeof(int8_t));
                                return;
                            }
                        case pressio_option_int16_type:
                            {
                                auto& d = i.second.get_value<int16_t>();
                                h(&d, sizeof(int16_t));
                                return;
                            }
                        case pressio_option_int32_type:
                            {
                                auto& d = i.second.get_value<int32_t>();
                                h(&d, sizeof(int32_t));
                                return;
                            }
                        case pressio_option_int64_type:
                            {
                                auto& d = i.second.get_value<int64_t>();
                                h(&d, sizeof(int64_t));
                                return;
                            }
                        case pressio_option_float_type:
                            {
                                auto& d = i.second.get_value<float>();
                                h(&d, sizeof(float));
                                return;
                            }
                        case pressio_option_double_type:
                            {
                                auto& d = i.second.get_value<double>();
                                h(&d, sizeof(double));
                                return;
                            }
                        case pressio_option_charptr_type:
                            {
                                auto& d = i.second.get_value<std::string>();
                                h(d);
                                return;
                            }
                        case pressio_option_charptr_array_type:
                            {
                                auto& d = i.second.get_value<std::vector<std::string>>();
                                h(d.size());
                                for (auto const& e : d) {
                                    h(e);
                                }
                                return;
                            }
                        case pressio_option_bool_type:
                            {
                                auto& d = i.second.get_value<bool>();
                                h(d);
                                return;
                            }
                        case pressio_option_dtype_type:
                            {
                                auto& d = i.second.get_value<pressio_dtype>();
                                h(d);
                                return;
                            }
                        case pressio_option_threadsafety_type:
                            {
                                auto& d = i.second.get_value<pressio_thread_safety>();
                                h(d);
                                return;
                            }

                    }
                }
        });
    }
}

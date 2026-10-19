#pragma once
#include <cmath>
#include <limits>
#include <type_traits>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "stage_registry.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

inline const std::map<std::string, fz::MemoryStrategy> MEMORY_STRATEGIES {
    {"preallocate", fz::MemoryStrategy::PREALLOCATE},
    {"minimal",     fz::MemoryStrategy::MINIMAL},
};

inline const std::map<std::string, fz::ErrorBoundMode> ERROR_BOUND_MODES {
    {"abs",  fz::ErrorBoundMode::ABS},
    {"rel",  fz::ErrorBoundMode::REL},
    {"noa",  fz::ErrorBoundMode::NOA},
    {"prel", fz::ErrorBoundMode::PREL},
};

inline const std::map<std::string, fz::SpecializationPolicy> SPECIALIZATION_POLICIES {
    {"off",  fz::SpecializationPolicy::Off},
    {"auto", fz::SpecializationPolicy::Auto},
};

template <class T>
inline std::vector<std::string> map_keys(std::map<std::string, T> const& m) {
    std::vector<std::string> keys;
    keys.reserve(m.size());
    for (auto const& kv : m) keys.push_back(kv.first);
    return keys;
}

struct PipelineOptions {
    double error_bound_abs = 1e-3;
    double error_bound_rel = 1e-3;
    std::string error_bound_mode = "abs";
    std::string memory_strategy = "minimal";
    // Python options require exact double/int64_t types, without numeric coercion.
    double memory_multiplier = 3.0;
    std::string specialization = "off";
    std::vector<size_t> dims = {0, 1, 1};
    int64_t num_streams = 1;
    std::vector<std::string> stages;
    std::vector<std::string> connections;
    // Empty selects the sole/first source; multi-source pipelines should set an id.
    std::string primary_source;
    std::string config_file;
    bool expose_stage_outputs = false;

    bool operator==(const PipelineOptions&) const = default;
};

// Owns options and every stage's parameters, including inactive handlers.
struct Configuration : PipelineOptions {
    StageRegistry stage_kinds;

    bool same_values(const Configuration& other) const {
        return static_cast<const PipelineOptions&>(*this) == other;
    }
    void swap(Configuration& other) noexcept {
        static_assert(std::is_nothrow_swappable_v<PipelineOptions>);
        using std::swap;
        swap(static_cast<PipelineOptions&>(*this), static_cast<PipelineOptions&>(other));
        stage_kinds.swap(other.stage_kinds);
    }

    template<typename F>
    void for_each_stage(F&& f) const {
        for(size_t i = 0; i < stages.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            if(const auto* kind = stage_kinds.find(stages[i])) f(*kind, sid, stages[i]);
        }
    }

    template<typename F>
    void for_each_stage(F&& f) {
        for(size_t i = 0; i < stages.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            if(auto* kind = stage_kinds.find(stages[i])) f(*kind, sid, stages[i]);
        }
    }

    // Mutates a candidate; returns whether stage parameters changed. Get preserves named option lookup.
    template<typename Get>
    bool read_options(const pressio_options& options, Get&& get) {
        auto read_option = [&](const auto& opts, const std::string& key, auto* value) {
            const auto status = get(opts, key, value);
            if(status == pressio_options_key_exists)
                throw std::invalid_argument(key + " has the wrong type or no value");
            return status;
        };
        read_option(options, "pressio:abs", &error_bound_abs);
        read_option(options, "pressio:rel", &error_bound_rel);

        read_option(options, "fzgpumodules:stages",         &stages);
        read_option(options, "fzgpumodules:connections",    &connections);
        read_option(options, "fzgpumodules:primary_source", &primary_source);

        read_option(options, "fzgpumodules:memory_strategy", &memory_strategy);
        if(!MEMORY_STRATEGIES.contains(memory_strategy)) {
            auto invalid = memory_strategy;
            throw std::invalid_argument("Invalid fzgpumodules:memory_strategy: '" + invalid +
                                "'. Valid: preallocate, minimal");
        }

        read_option(options, "fzgpumodules:memory_multiplier", &memory_multiplier);

        read_option(options, "fzgpumodules:error_bound_mode", &error_bound_mode);
        if(!ERROR_BOUND_MODES.contains(error_bound_mode)) {
            auto invalid = error_bound_mode;
            throw std::invalid_argument("Invalid fzgpumodules:error_bound_mode: '" + invalid +
                                "'. Valid: abs, rel, noa, prel");
        }

        pressio_data dims_data;
        if(read_option(options, "fzgpumodules:dims", &dims_data) == pressio_options_key_set) {
            std::vector<size_t> v;
            switch(dims_data.dtype()) {
                case pressio_uint8_dtype: case pressio_uint16_dtype:
                case pressio_uint32_dtype: case pressio_uint64_dtype:
                    for(auto dim : dims_data.to_vector<uint64_t>()) {
                        if(dim > std::numeric_limits<size_t>::max())
                            throw std::invalid_argument("dims exceed the host size range");
                        v.push_back(static_cast<size_t>(dim));
                    }
                    break;
                case pressio_int8_dtype: case pressio_int16_dtype:
                case pressio_int32_dtype: case pressio_int64_dtype:
                    for(auto dim : dims_data.to_vector<int64_t>()) {
                        if(dim < 0 || uint64_t(dim) > std::numeric_limits<size_t>::max())
                            throw std::invalid_argument("dims must be nonnegative host sizes");
                        v.push_back(static_cast<size_t>(dim));
                    }
                    break;
                default: throw std::invalid_argument("dims must contain integer values");
            }
            if(v.empty() || v.size() > 3)
                throw std::invalid_argument("dims must contain one to three values");
            dims.assign(3, size_t(1));
            for(size_t i = 0; i < v.size(); ++i) {
                if(i > 0 && v[i] == 0)
                    throw std::invalid_argument("only the first dimension may be zero (automatic)");
                dims[i] = v[i];
            }
        }

        read_option(options, "fzgpumodules:num_streams", &num_streams);

        bool removed_graph_mode;
        if (get(options, "fzgpumodules:graph_mode", &removed_graph_mode) != pressio_options_key_does_not_exist)
            throw std::invalid_argument("fzgpumodules:graph_mode was removed; CUDA Graph capture is not supported by this plugin");

        std::string removed_fusion;
        if (get(options, "fzgpumodules:fusion", &removed_fusion) != pressio_options_key_does_not_exist)
            throw std::invalid_argument("fzgpumodules:fusion was removed; use fzgpumodules:specialization");
        std::string requested_specialization = specialization;
        auto status = get(options, "fzgpumodules:specialization", &requested_specialization);
        if (status == pressio_options_key_exists)
            throw std::invalid_argument("fzgpumodules:specialization must be a string");
        if (!SPECIALIZATION_POLICIES.contains(requested_specialization))
            throw std::invalid_argument("Invalid fzgpumodules:specialization: '" + requested_specialization +
                                "'. Valid: off, auto");
        specialization = requested_specialization;

        read_option(options, "fzgpumodules:config_file", &config_file);

        read_option(options, "fzgpumodules:expose_stage_outputs", &expose_stage_outputs);

        if(!std::isfinite(error_bound_abs) || error_bound_abs < 0 ||
           !std::isfinite(error_bound_rel) || error_bound_rel < 0)
            throw std::invalid_argument("error bounds must be finite and nonnegative");
        if(!std::isfinite(memory_multiplier) || memory_multiplier <= 0 ||
           memory_multiplier > std::numeric_limits<float>::max() ||
           static_cast<float>(memory_multiplier) == 0)
            throw std::invalid_argument("memory_multiplier must be a finite positive float");
        if(num_streams < 1 || num_streams > std::numeric_limits<int>::max())
            throw std::invalid_argument("num_streams must fit a positive int");

        bool stage_params_changed = false;
        for_each_stage([&](StageKind& sk, const std::string& sid, const std::string& tok) {
            pressio_options expected;
            sk.populate_options(expected, sid, tok);
            for(const auto& entry : expected) {
                if(options.key_status(entry.first) == pressio_options_key_exists ||
                   (options.key_status(entry.first) == pressio_options_key_set &&
                    options.get(entry.first).type() != entry.second.type()))
                    throw std::invalid_argument(entry.first + " has the wrong type");
            }
            stage_params_changed |= sk.read_options(options, sid, tok);
        });

        return stage_params_changed;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

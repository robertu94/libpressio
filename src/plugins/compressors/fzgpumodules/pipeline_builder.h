#pragma once
#include <array>
#include <regex>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "configuration.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct ParsedConnection { std::string to, from, port; };

inline ParsedConnection parse_connection(const std::string& s) {
    static const std::regex re(R"(^\s*(\S+)\s*<-\s*(\S+?)(?::(\S+))?\s*$)");
    std::smatch m;
    if(!std::regex_match(s, m, re))
        throw std::invalid_argument("Invalid connection (expected 'sN <- sM[:port]'): " + s);
    return {m[1].str(), m[2].str(), m[3].str()};
}

struct OutlierStageRef {
    fz::Stage*  ptr;
    std::string indices_key;  // output port whose size / 4 == outlier count
};

struct TerminalOutput {
    std::string   metric_key;  // "fzgpumodules:sN:output:<port>"
    pressio_dtype dtype;
};

struct PipelineState {
    std::unique_ptr<fz::Pipeline>              ptr;
    bool   dirty     = true;
    size_t last_size = 0;
    std::vector<std::pair<std::string, OutlierStageRef>> outlier_stages;   // sid → stage ref
    std::vector<TerminalOutput>                          terminal_outputs;  // ordered terminal ports

    PipelineState() = default;
    PipelineState(PipelineState&&) = default;
    PipelineState& operator=(PipelineState&&) = default;
    PipelineState(PipelineState const&) = delete;
    PipelineState& operator=(PipelineState const&) = delete;
};

// Apply common per-pipeline settings after construction.
// Pass dims=nullptr in config-file mode (the TOML sets dims internally).
inline void configure_pipeline_base(fz::Pipeline& p,
                                     const std::array<size_t,3>* dims,
                                     int num_streams,
                                     fz::SpecializationPolicy specialization_policy) {
    if (dims) p.setDims(*dims);
    if (num_streams > 1) p.setNumStreams(num_streams);
    p.setSpecializationPolicy(specialization_policy);
}

inline pressio_dtype fz_to_pressio_dtype(fz::DataType dt) {
    switch (dt) {
        case fz::DataType::UINT8:   return pressio_uint8_dtype;
        case fz::DataType::UINT16:  return pressio_uint16_dtype;
        case fz::DataType::UINT32:  return pressio_uint32_dtype;
        case fz::DataType::UINT64:  return pressio_uint64_dtype;
        case fz::DataType::INT8:    return pressio_int8_dtype;
        case fz::DataType::INT16:   return pressio_int16_dtype;
        case fz::DataType::INT32:   return pressio_int32_dtype;
        case fz::DataType::INT64:   return pressio_int64_dtype;
        case fz::DataType::FLOAT32: return pressio_float_dtype;
        case fz::DataType::FLOAT64: return pressio_double_dtype;
        default:                    return pressio_byte_dtype;
    }
}

inline PipelineState build_manual_pipeline(Configuration& config, size_t data_size) {
    PipelineState state;
    std::vector<ParsedConnection> connections;
    connections.reserve(config.connections.size());
    for(const auto& text : config.connections) connections.push_back(parse_connection(text));

    const auto strategy = MEMORY_STRATEGIES.at(config.memory_strategy);

    state.ptr = std::make_unique<fz::Pipeline>(data_size, strategy, static_cast<float>(config.memory_multiplier));
    const std::array<size_t,3> fz_dims = {config.dims[0], config.dims[1], config.dims[2]};
    configure_pipeline_base(*state.ptr, &fz_dims, static_cast<int>(config.num_streams), SPECIALIZATION_POLICIES.at(config.specialization));

    StageContext ctx{
        *state.ptr,
        ERROR_BOUND_MODES.at(config.error_bound_mode),
        (config.error_bound_mode == "rel") ? config.error_bound_rel : config.error_bound_abs
    };

    std::map<std::string, fz::Stage*> ptrs;
    for(size_t i = 0; i < config.stages.size(); i++) {
        std::string sid = "s" + std::to_string(i);
        auto* kind = config.stage_kinds.find(config.stages[i]);
        if(!kind) throw std::runtime_error("Unknown stage type: " + config.stages[i]);
        auto* stage = kind->add_stage(config.stages[i], sid, ctx);
        const auto indices_key = kind->outlier_indices_key(config.stages[i]);
        if(!indices_key.empty())
            state.outlier_stages.push_back({sid, {stage, indices_key}});
        ptrs[sid] = stage;
    }

    for(const auto& c : connections) {
        if(c.from == "__external__") {
            // Input-port order follows connection order, including external bindings.
            state.ptr->bindExternalInput(ptrs.at(c.to));
            continue;
        }
        if(c.port.empty())
            state.ptr->connect(ptrs.at(c.to), ptrs.at(c.from));
        else
            state.ptr->connect(ptrs.at(c.to), ptrs.at(c.from), c.port);
    }

    if(!config.primary_source.empty())
        state.ptr->setPrimarySource(ptrs.at(config.primary_source));

    // Build ordered list of terminal (unconnected) output ports for expose_stage_outputs.
    // A port is terminal if no connection names it as the 'from' source.
    // An empty port in a connection means the stage's first (index-0) port.
    if (config.expose_stage_outputs) {
        auto is_consumed = [&](const std::string& sid, const std::string& port, size_t idx) {
            for(const auto& c : connections) {
                if (c.from != sid) continue;
                if (c.port == port)              return true;  // explicit match
                if (c.port.empty() && idx == 0)  return true;  // default = first port
            }
            return false;
        };
        for (size_t i = 0; i < config.stages.size(); i++) {
            std::string sid = "s" + std::to_string(i);
            fz::Stage*  stage = ptrs[sid];
            auto names = stage->getOutputNames();
            if (names.empty()) names = {"output"};
            for (size_t j = 0; j < names.size(); j++) {
                if (!is_consumed(sid, names[j], j)) {
                    state.terminal_outputs.push_back({
                        "fzgpumodules:" + sid + ":output:" + names[j],
                        fz_to_pressio_dtype(static_cast<fz::DataType>(stage->getOutputDataType(j)))
                    });
                }
            }
        }
    }

    state.ptr->finalize();
    return state;
}

inline PipelineState build_config_pipeline(const Configuration& config) {
    PipelineState state;
    const auto strategy = MEMORY_STRATEGIES.at(config.memory_strategy);

    state.ptr = std::make_unique<fz::Pipeline>(0, strategy, static_cast<float>(config.memory_multiplier));
    configure_pipeline_base(*state.ptr, nullptr, static_cast<int>(config.num_streams), SPECIALIZATION_POLICIES.at(config.specialization));

    state.ptr->loadConfig(config.config_file);  // adds stages, wires, and finalizes
    return state;
}


// Publish only a fully constructed pipeline; failed construction leaves live state intact.
inline PipelineState build_encoder(Configuration& config, size_t data_size) {
    auto state = config.config_file.empty() ? build_manual_pipeline(config, data_size)
                                          : build_config_pipeline(config);
    state.dirty = false;
    state.last_size = data_size;
    return state;
}

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

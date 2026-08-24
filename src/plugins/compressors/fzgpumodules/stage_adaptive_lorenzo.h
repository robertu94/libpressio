#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct AdaptiveLorenzoParams {
    int64_t coder_block_size = 32;  // fixed at 32 by the cost model; don't change
    int64_t blocks_per_tile  = 8;   // tile length = coder_block_size * blocks_per_tile, in [1, 32]
    bool    enable_order2    = true;
    bool    enable_centering = true;
    bool operator==(const AdaptiveLorenzoParams& o) const {
        return coder_block_size == o.coder_block_size && blocks_per_tile == o.blocks_per_tile &&
               enable_order2 == o.enable_order2 && enable_centering == o.enable_centering;
    }
    bool operator!=(const AdaptiveLorenzoParams& o) const { return !(*this == o); }
};

// Forward: 1 input -> 3 outputs (output, modes, means). Inverse: 3 inputs -> 1
// output. Confirmed empirically that Pipeline::buildInverseDAG() reconciles
// this direction-dependent port count transparently — no plugin-side handling
// needed, unlike stages whose port count depends on a *config* setter (see
// GPULZ's split_mode, which does need Config-at-construction support the
// engine doesn't have).
class AdaptiveLorenzoStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "adaptive_lorenzo"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "int32";
        const auto& p = get_params(sid);

        // Only int16/int32 are explicitly instantiated in the compiled library
        // (modules/fused/adaptive_lorenzo/*.cu) despite the header documenting
        // int8/int64 support too.
        if(t == "int16") return make<int16_t>(p, ctx);
        if(t == "int32") return make<int32_t>(p, ctx);
        throw std::runtime_error("Unsupported adaptive_lorenzo type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":coder_block_size", p.coder_block_size);
        opts.set("fzgpumodules:" + sid + ":blocks_per_tile",  p.blocks_per_tile);
        opts.set("fzgpumodules:" + sid + ":enable_order2",    p.enable_order2);
        opts.set("fzgpumodules:" + sid + ":enable_centering", p.enable_centering);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = AdaptiveLorenzoParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":coder_block_size", &p.coder_block_size);
        opts.get("fzgpumodules:" + sid + ":blocks_per_tile",  &p.blocks_per_tile);
        opts.get("fzgpumodules:" + sid + ":enable_order2",    &p.enable_order2);
        opts.get("fzgpumodules:" + sid + ":enable_centering", &p.enable_centering);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":coder_block_size",
            std::string("Downstream coder block size — fixed at 32 by the cost model and the "
            "one-warp-per-block reduction; only AdaptiveBitpackStage packs each bit-plane "
            "into exactly one 32-bit word at this size (default 32)"));
        opts.set("fzgpumodules:" + sid + ":blocks_per_tile",
            std::string("Coder blocks per adaptation tile, in [1, 32] (tile size = "
            "coder_block_size * blocks_per_tile <= 1024). Longer tiles mean a longer "
            "prediction chain and cheaper per-tile metadata but coarser adaptation "
            "granularity (default 8)"));
        opts.set("fzgpumodules:" + sid + ":enable_order2",
            std::string("Include order-2 (LZ2) prediction variants in the per-tile search "
            "(default: true)"));
        opts.set("fzgpumodules:" + sid + ":enable_centering",
            std::string("Include centered prediction variants in the per-tile search "
            "(default: true)"));
    }

private:
    std::map<std::string, AdaptiveLorenzoParams> params_;
    const AdaptiveLorenzoParams defaults_{};

    const AdaptiveLorenzoParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename T>
    fz::Stage* make(const AdaptiveLorenzoParams& p, const StageContext& ctx) {
        typename fz::AdaptiveLorenzoStage<T>::Config config;
        config.coder_block_size = static_cast<uint32_t>(p.coder_block_size);
        config.blocks_per_tile  = static_cast<uint32_t>(p.blocks_per_tile);
        config.enable_order2    = p.enable_order2;
        config.enable_centering = p.enable_centering;
        return ctx.pipeline.addStage<fz::AdaptiveLorenzoStage<T>>(config);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct AdaptiveBitpackParams {
    int64_t block_size        = 32;     // elements per fixed-rate block, in [1, 1024]
    bool    outlier_selection = false;  // cuSZp2 per-block plain/outlier selection
    bool operator==(const AdaptiveBitpackParams& o) const {
        return block_size == o.block_size && outlier_selection == o.outlier_selection;
    }
    bool operator!=(const AdaptiveBitpackParams& o) const { return !(*this == o); }
};

class AdaptiveBitpackStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "adaptive_bitpack"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "int32";
        const auto& p = get_params(sid);

        if(t == "int16") return make<int16_t>(p, ctx);
        if(t == "int32") return make<int32_t>(p, ctx);
        throw std::runtime_error("Unsupported adaptive_bitpack type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":block_size",        p.block_size);
        opts.set("fzgpumodules:" + sid + ":outlier_selection", p.outlier_selection);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = AdaptiveBitpackParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":block_size",        &p.block_size);
        opts.get("fzgpumodules:" + sid + ":outlier_selection", &p.outlier_selection);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":block_size",
            std::string("Elements per fixed-rate block, in [1, 1024] (default 32; cuSZp uses 32). "
            "Match a TiledLorenzo tile_elems upstream to align blocks to tiles."));
        opts.set("fzgpumodules:" + sid + ":outlier_selection",
            std::string("Enable cuSZp2 per-block plain/outlier selection: store element 0 as a "
            "raw outlier and pack only the rest, whichever is smaller. Helps non-sparse, "
            "high-smoothness data (default: false)."));
    }

private:
    std::map<std::string, AdaptiveBitpackParams> params_;
    const AdaptiveBitpackParams defaults_{};

    const AdaptiveBitpackParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename T>
    fz::Stage* make(const AdaptiveBitpackParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::AdaptiveBitpackStage<T>>();
        s->setBlockSize(static_cast<uint32_t>(p.block_size));
        s->setOutlierSelection(p.outlier_selection);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

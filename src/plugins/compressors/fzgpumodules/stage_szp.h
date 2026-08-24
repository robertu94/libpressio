#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct SZpParams {
    int64_t block_size = 128;  // elements per block, in [1, 4096]
    bool operator==(const SZpParams& o) const { return block_size == o.block_size; }
    bool operator!=(const SZpParams& o) const { return !(*this == o); }
};

class SZpStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "szp"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "float";
        const auto& p = get_params(sid);

        if(t == "float")  return make<float> (p, ctx);
        if(t == "double") return make<double>(p, ctx);
        throw std::runtime_error("Unsupported szp type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":block_size", get_params(sid).block_size);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = SZpParams{};
        auto  old = params_[sid];
        opts.get("fzgpumodules:" + sid + ":block_size", &params_[sid].block_size);
        return params_[sid] != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":block_size",
            std::string("Elements per block, in [1, 4096] (default 128). Uses the pipeline's "
            "error_bound_mode/pressio:abs; only abs and noa are supported (rel/prel error)."));
    }

private:
    std::map<std::string, SZpParams> params_;
    const SZpParams defaults_{};

    const SZpParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    static fz::SZpErrorMode to_szp_mode(fz::ErrorBoundMode mode) {
        switch(mode) {
            case fz::ErrorBoundMode::ABS: return fz::SZpErrorMode::ABS;
            case fz::ErrorBoundMode::NOA: return fz::SZpErrorMode::NOA;
            default:
                throw std::runtime_error(
                    "szp: only fzgpumodules:error_bound_mode = abs or noa are supported");
        }
    }

    template<typename T>
    fz::Stage* make(const SZpParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::SZpStage<T>>();
        s->setBlockSize(static_cast<uint32_t>(p.block_size));
        s->setErrorMode(to_szp_mode(ctx.eb_mode));
        s->setErrorBound(ctx.eb);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

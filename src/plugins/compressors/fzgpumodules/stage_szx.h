#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct SZxParams {
    int64_t block_size = 128;  // elements per block, in [1, 4096]
    bool operator==(const SZxParams& o) const { return block_size == o.block_size; }
    bool operator!=(const SZxParams& o) const { return !(*this == o); }
};

class SZxStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "szx"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "float";
        const auto& p = get_params(sid);

        if(t == "float")  return make<float> (p, ctx);
        if(t == "double") return make<double>(p, ctx);
        throw std::runtime_error("Unsupported szx type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":block_size", get_params(sid).block_size);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = SZxParams{};
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
    std::map<std::string, SZxParams> params_;
    const SZxParams defaults_{};

    const SZxParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    static fz::SZxErrorMode to_szx_mode(fz::ErrorBoundMode mode) {
        switch(mode) {
            case fz::ErrorBoundMode::ABS: return fz::SZxErrorMode::ABS;
            case fz::ErrorBoundMode::NOA: return fz::SZxErrorMode::NOA;
            default:
                throw std::runtime_error(
                    "szx: only fzgpumodules:error_bound_mode = abs or noa are supported");
        }
    }

    template<typename T>
    fz::Stage* make(const SZxParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::SZxStage<T>>();
        s->setBlockSize(static_cast<uint32_t>(p.block_size));
        s->setErrorMode(to_szx_mode(ctx.eb_mode));
        s->setErrorBound(ctx.eb);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

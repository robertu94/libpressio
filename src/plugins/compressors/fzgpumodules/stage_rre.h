#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct RREParams {
    int64_t chunk_size = 16384;  // bytes; only 16384 is supported
    int64_t word_size  = 1;      // LC RRE word granularity: 1, 2, 4, or 8
    bool operator==(const RREParams& o) const {
        return chunk_size == o.chunk_size && word_size == o.word_size;
    }
    bool operator!=(const RREParams& o) const { return !(*this == o); }
};

class RREStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "rre"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::RREStage>();
        s->setChunkSize(static_cast<size_t>(p.chunk_size));
        s->setWordSize(static_cast<size_t>(p.word_size));
        return s;
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":chunk_size", p.chunk_size);
        opts.set("fzgpumodules:" + sid + ":word_size",  p.word_size);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = RREParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":chunk_size", &p.chunk_size);
        opts.get("fzgpumodules:" + sid + ":word_size",  &p.word_size);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":chunk_size",
            std::string("RRE chunk size in bytes (only 16384 is supported; default 16384)"));
        opts.set("fzgpumodules:" + sid + ":word_size",
            std::string("RRE word granularity 1/2/4/8 = LC RRE_1/2/4/8 (default 1)"));
    }

private:
    std::map<std::string, RREParams> params_;
    const RREParams defaults_{};

    const RREParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct RZEParams {
    // int64_t (not int): pressio_options::get() requires an exact type match
    // and libpressio's Python layer always boxes int->int64_t, so narrower
    // fields here would silently never be set.
    int64_t chunk_size = 16384;  // bytes; 4096, 8192, or 16384
    int64_t word_size  = 1;      // LC RZE_N word granularity: 1, 2, 4, or 8
    bool operator==(const RZEParams& o) const {
        return chunk_size == o.chunk_size && word_size == o.word_size;
    }
    bool operator!=(const RZEParams& o) const { return !(*this == o); }
};

class RZEStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "rze"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::RZEStage>();
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
        if(params_.count(sid) == 0) params_[sid] = RZEParams{};
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
            std::string("RZE chunk size in bytes (4096, 8192, or 16384; default 16384)"));
        opts.set("fzgpumodules:" + sid + ":word_size",
            std::string("RZE word granularity 1/2/4/8 = LC RZE_1/2/4/8 (default 1)"));
    }

private:
    std::map<std::string, RZEParams> params_;
    const RZEParams defaults_{};

    const RZEParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

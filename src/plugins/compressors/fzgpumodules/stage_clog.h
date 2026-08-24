#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct CLOGParams {
    int64_t chunk_size = 16384;  // bytes; 4096, 8192, or 16384
    int64_t word_size  = 1;      // LC CLOG word granularity: 1, 2, 4, or 8 (unsigned only)
    bool operator==(const CLOGParams& o) const {
        return chunk_size == o.chunk_size && word_size == o.word_size;
    }
    bool operator!=(const CLOGParams& o) const { return !(*this == o); }
};

class CLOGStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "clog"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::CLOGStage>();
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
        if(params_.count(sid) == 0) params_[sid] = CLOGParams{};
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
            std::string("CLOG chunk size in bytes (4096, 8192, or 16384; default 16384)"));
        opts.set("fzgpumodules:" + sid + ":word_size",
            std::string("CLOG word granularity 1/2/4/8, unsigned words only (default 1)"));
    }

private:
    std::map<std::string, CLOGParams> params_;
    const CLOGParams defaults_{};

    const CLOGParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

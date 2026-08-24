#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct TUPLParams {
    int64_t block_size = 16384;  // bytes; chunk over which the transpose runs
    int64_t word_size  = 1;      // bytes per tuple element
    int64_t dim        = 2;      // tuple width (elements per group)
    bool operator==(const TUPLParams& o) const {
        return block_size == o.block_size && word_size == o.word_size && dim == o.dim;
    }
    bool operator!=(const TUPLParams& o) const { return !(*this == o); }
};

class TUPLStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "tupl"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::TUPLStage>();
        s->setBlockSize(static_cast<size_t>(p.block_size));
        s->setWordSize(static_cast<size_t>(p.word_size));
        s->setDim(static_cast<size_t>(p.dim));
        return s;
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":block_size", p.block_size);
        opts.set("fzgpumodules:" + sid + ":word_size",  p.word_size);
        opts.set("fzgpumodules:" + sid + ":dim",        p.dim);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = TUPLParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":block_size", &p.block_size);
        opts.get("fzgpumodules:" + sid + ":word_size",  &p.word_size);
        opts.get("fzgpumodules:" + sid + ":dim",        &p.dim);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":block_size",
            std::string("Chunk size in bytes over which the AoS->SoA transpose runs (default 16384)"));
        opts.set("fzgpumodules:" + sid + ":word_size",
            std::string("Bytes per tuple element (default 1)"));
        opts.set("fzgpumodules:" + sid + ":dim",
            std::string("Tuple width: elements per interleaved group (default 2)"));
    }

private:
    std::map<std::string, TUPLParams> params_;
    const TUPLParams defaults_{};

    const TUPLParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

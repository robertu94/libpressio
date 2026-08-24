#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct GPULZParams {
    int64_t chunk_size  = 2048;  // bytes
    int64_t word_size   = 4;     // bytes
    int64_t match_level  = 1;    // 0 = near-window only, 1 = + hashed long-range table
    // split_mode is intentionally not exposed: GPULZStage has no Config-based
    // constructor, and it *increases* getNumOutputs() from 1 to 4, which
    // addRawStage() captures at addStage() time before this factory gets the
    // returned pointer back. Confirmed empirically — Pipeline tolerates a
    // stage's port count *decreasing* after addStage() (stale-buffer
    // cleanup in autoDetectUnconnectedOutputs()) but not increasing: setting
    // split_mode=true here throws "Missing pre-allocated buffer for output 1
    // of stage 'GPULZ' — this is an internal pipeline bug" at construction.
    bool operator==(const GPULZParams& o) const {
        return chunk_size == o.chunk_size && word_size == o.word_size &&
               match_level == o.match_level;
    }
    bool operator!=(const GPULZParams& o) const { return !(*this == o); }
};

class GPULZStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "gpulz"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::GPULZStage>();
        s->setChunkSize(static_cast<size_t>(p.chunk_size));
        s->setWordSize(static_cast<size_t>(p.word_size));
        s->setMatchLevel(static_cast<int>(p.match_level));
        return s;
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":chunk_size",  p.chunk_size);
        opts.set("fzgpumodules:" + sid + ":word_size",   p.word_size);
        opts.set("fzgpumodules:" + sid + ":match_level", p.match_level);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = GPULZParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":chunk_size",  &p.chunk_size);
        opts.get("fzgpumodules:" + sid + ":word_size",   &p.word_size);
        opts.get("fzgpumodules:" + sid + ":match_level", &p.match_level);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":chunk_size",
            std::string("GPULZ chunk size in bytes (default 2048)"));
        opts.set("fzgpumodules:" + sid + ":word_size",
            std::string("GPULZ match word size in bytes (default 4)"));
        opts.set("fzgpumodules:" + sid + ":match_level",
            std::string("Match-search effort, encode-side only: 0 = exact longest match over "
            "the near window; 1 = additionally consults a hashed long-range table "
            "(default 1, trades throughput for ratio)"));
    }

private:
    std::map<std::string, GPULZParams> params_;
    const GPULZParams defaults_{};

    const GPULZParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

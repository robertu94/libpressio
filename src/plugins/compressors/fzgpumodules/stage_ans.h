#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// No configurable params: the vendored dietGPU kernels only support
// prob_bits=10 in this build (ANSStage::setProbBits is reserved for future
// use and throws on any other value), so it is intentionally not exposed.
class ANSStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "ans"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        return ctx.pipeline.addStage<fz::ANSStage>();
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

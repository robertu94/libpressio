#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// No configurable params: BitplaneRZEStage is a fixed uint16_t-input fused
// stage (no template parameter, no setters besides setInverse).
class BitplaneRZEStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "bitplane_rze"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        return ctx.pipeline.addStage<fz::BitplaneRZEStage>();
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

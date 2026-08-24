#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

class ZigzagStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "zigzag"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "int32";

        if(t == "int8")  return ctx.pipeline.addStage<fz::ZigzagStage<int8_t>>();
        if(t == "int16") return ctx.pipeline.addStage<fz::ZigzagStage<int16_t>>();
        if(t == "int32") return ctx.pipeline.addStage<fz::ZigzagStage<int32_t>>();
        if(t == "int64") return ctx.pipeline.addStage<fz::ZigzagStage<int64_t>>();
        throw std::runtime_error("Unsupported zigzag type: " + t);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

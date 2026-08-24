#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

class NegabinaryStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "negabinary"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "int32";

        if(t == "int8")  return ctx.pipeline.addStage<fz::NegabinaryStage<int8_t>>();
        if(t == "int16") return ctx.pipeline.addStage<fz::NegabinaryStage<int16_t>>();
        if(t == "int32") return ctx.pipeline.addStage<fz::NegabinaryStage<int32_t>>();
        if(t == "int64") return ctx.pipeline.addStage<fz::NegabinaryStage<int64_t>>();
        throw std::runtime_error("Unsupported negabinary type: " + t);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

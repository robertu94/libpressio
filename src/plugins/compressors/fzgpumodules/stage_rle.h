#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

class RLEStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "rle"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "uint16";

        if(t == "uint8")  return ctx.pipeline.addStage<fz::RLEStage<uint8_t>>();
        if(t == "uint16") return ctx.pipeline.addStage<fz::RLEStage<uint16_t>>();
        if(t == "uint32") return ctx.pipeline.addStage<fz::RLEStage<uint32_t>>();
        if(t == "int32")  return ctx.pipeline.addStage<fz::RLEStage<int32_t>>();
        throw std::runtime_error("Unsupported rle type: " + t);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

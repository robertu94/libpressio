#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

class DiffStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "diff"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string  t  = (parts.size() > 1) ? parts[1] : "uint16";
        const std::string  t2 = (parts.size() > 2) ? parts[2] : "";

        // Negabinary-fused two-type variants: diff:intN:uintN
        if(!t2.empty()) {
            if(t == "int8"  && t2 == "uint8")  return ctx.pipeline.addStage<fz::DifferenceStage<int8_t,  uint8_t>>();
            if(t == "int16" && t2 == "uint16") return ctx.pipeline.addStage<fz::DifferenceStage<int16_t, uint16_t>>();
            if(t == "int32" && t2 == "uint32") return ctx.pipeline.addStage<fz::DifferenceStage<int32_t, uint32_t>>();
            if(t == "int64" && t2 == "uint64") return ctx.pipeline.addStage<fz::DifferenceStage<int64_t, uint64_t>>();
            throw std::runtime_error("Unsupported diff types: " + t + ":" + t2);
        }

        if(t == "float")  return ctx.pipeline.addStage<fz::DifferenceStage<float>>();
        if(t == "double") return ctx.pipeline.addStage<fz::DifferenceStage<double>>();
        if(t == "uint8")  return ctx.pipeline.addStage<fz::DifferenceStage<uint8_t>>();
        if(t == "uint16") return ctx.pipeline.addStage<fz::DifferenceStage<uint16_t>>();
        if(t == "uint32") return ctx.pipeline.addStage<fz::DifferenceStage<uint32_t>>();
        if(t == "int32")  return ctx.pipeline.addStage<fz::DifferenceStage<int32_t>>();
        if(t == "int64")  return ctx.pipeline.addStage<fz::DifferenceStage<int64_t>>();
        throw std::runtime_error("Unsupported diff type: " + t);
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

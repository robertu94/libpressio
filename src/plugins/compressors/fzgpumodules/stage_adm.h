#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// Element type is fixed at construction (ADMStage::setDtype must be called
// before finalize()), so it is selected via the token suffix rather than a
// runtime option: "adm:uint16" (default) or "adm:uint32".
//
// ADM assumes bounded quantization codes (small diffs from a per-block
// center) — see mapping_uint16.cu / mapping_uint32.cu. Input whose diffs
// exceed that capacity now raises a clean "local_bits overflow" exception
// instead of corrupting device memory (the overflow guard used to be
// compiled out in Release builds; fixed alongside a couple of related
// out-of-bounds writes discovered while wiring this handler).
class ADMStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "adm"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& /*sid*/,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "uint16";

        auto* s = ctx.pipeline.addStage<fz::ADMStage>();
        if(t == "uint16")      s->setDtype(fz::ADMDtype::U16);
        else if(t == "uint32") s->setDtype(fz::ADMDtype::U32);
        else throw std::runtime_error("Unsupported adm type: " + t);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct GInterpParams {
    int64_t quant_radius     = 0;      // 0 = auto-tune on first execute()
    double  outlier_capacity = 0.10;
    int64_t auto_tuning_mode = 0;      // 0 off, 1 cheap, 3 full, 4 full+sweep, 5 manual
    double  manual_alpha     = 0.0;
    double  manual_beta      = 0.0;
    bool operator==(const GInterpParams& o) const {
        return quant_radius == o.quant_radius && outlier_capacity == o.outlier_capacity &&
               auto_tuning_mode == o.auto_tuning_mode && manual_alpha == o.manual_alpha &&
               manual_beta == o.manual_beta;
    }
    bool operator!=(const GInterpParams& o) const { return !(*this == o); }
};

// cuSZ-Hi-style interpolation predictor. 2-D/3-D only (rejects 1-D dims).
// Forward: 1 input -> 4 outputs. Inverse: 4 inputs -> 1 output. Direction
// swap handled transparently by Pipeline::buildInverseDAG() — no plugin-side
// handling needed (confirmed empirically against AdaptiveLorenzoStage).
class GInterpStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "ginterp"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        typename fz::GInterpStage<float, uint16_t>::Config config;
        config.error_bound      = static_cast<float>(ctx.eb);
        config.eb_mode          = ctx.eb_mode;
        config.quant_radius     = static_cast<int>(p.quant_radius);
        config.outlier_capacity = static_cast<float>(p.outlier_capacity);
        config.auto_tuning_mode = static_cast<uint8_t>(p.auto_tuning_mode);
        config.manual_alpha     = p.manual_alpha;
        config.manual_beta      = p.manual_beta;
        return ctx.pipeline.addStage<fz::GInterpStage<float, uint16_t>>(config);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":quant_radius",     p.quant_radius);
        opts.set("fzgpumodules:" + sid + ":outlier_capacity", p.outlier_capacity);
        opts.set("fzgpumodules:" + sid + ":auto_tuning_mode", p.auto_tuning_mode);
        opts.set("fzgpumodules:" + sid + ":manual_alpha",     p.manual_alpha);
        opts.set("fzgpumodules:" + sid + ":manual_beta",      p.manual_beta);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = GInterpParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":quant_radius",     &p.quant_radius);
        opts.get("fzgpumodules:" + sid + ":outlier_capacity", &p.outlier_capacity);
        opts.get("fzgpumodules:" + sid + ":auto_tuning_mode", &p.auto_tuning_mode);
        opts.get("fzgpumodules:" + sid + ":manual_alpha",     &p.manual_alpha);
        opts.get("fzgpumodules:" + sid + ":manual_beta",      &p.manual_beta);
        return p != old;
    }

    std::string outlier_indices_key(const std::string& /*token*/) const override {
        return "outlier_idxs";
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":quant_radius",
            std::string("0 = auto-tune (default): on first execute() the stage scans the input "
            "min/max and picks the largest radius that fits the data range. >0 = manual: use "
            "this radius directly, skip the scan (required for CUDA graph capture)."));
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",
            std::string("Outlier buffer reserve as fraction of total elements (default 0.10)"));
        opts.set("fzgpumodules:" + sid + ":auto_tuning_mode",
            std::string("INTERPOLATION_PARAMS auto-tuning (cuSZ-Hi): 0 = off (default, "
            "deterministic baseline), 1 = cheap profiling (~1ms, 3-D only), 3 = full "
            "structural profiling (~5-15ms, 3-D only), 4 = full + alpha/beta sweep "
            "(~15-30ms, 3-D only), 5 = manual alpha/beta override (dim-agnostic, graph-safe). "
            "Modes 1/3/4 force a host-blocking D2H sync and are incompatible with CUDA graph "
            "capture; on 2-D input they fall back to baseline."));
        opts.set("fzgpumodules:" + sid + ":manual_alpha",
            std::string("Manual alpha for auto_tuning_mode=5; >0 uses it verbatim, 0.0 (default) "
            "defers to the cuSZ-Hi piecewise-linear schedule keyed on the relative error bound"));
        opts.set("fzgpumodules:" + sid + ":manual_beta",
            std::string("Manual beta for auto_tuning_mode=5; same convention as manual_alpha "
            "(default 0.0 -> beta=4.0)"));
    }

private:
    std::map<std::string, GInterpParams> params_;
    const GInterpParams defaults_{};

    const GInterpParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

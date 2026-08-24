#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct LogTransformParams {
    double threshold        = 0.0;
    double outlier_capacity = 0.05;
    bool operator==(const LogTransformParams& o) const {
        return threshold == o.threshold && outlier_capacity == o.outlier_capacity;
    }
    bool operator!=(const LogTransformParams& o) const { return !(*this == o); }
};

// Forward: 1 input -> 4 outputs. Inverse: 4 inputs -> 1 output. Direction
// swap handled transparently by Pipeline::buildInverseDAG() — no plugin-side
// handling needed (confirmed empirically against AdaptiveLorenzoStage).
class LogTransformStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "log_transform"; }

    fz::Stage* add_stage(const std::string& /*token*/,
                          const std::string& sid,
                          const StageContext& ctx) override {
        const auto& p = get_params(sid);
        fz::LogTransformStage<float>::Config config;
        config.error_bound      = static_cast<float>(ctx.eb);
        config.threshold        = static_cast<float>(p.threshold);
        config.outlier_capacity = static_cast<float>(p.outlier_capacity);
        return ctx.pipeline.addStage<fz::LogTransformStage<float>>(config);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":threshold",        p.threshold);
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",  p.outlier_capacity);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = LogTransformParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":threshold",       &p.threshold);
        opts.get("fzgpumodules:" + sid + ":outlier_capacity", &p.outlier_capacity);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":threshold",
            std::string("|x| < threshold => lossless outlier. 0 disables the threshold, so "
            "only zeros/denormals/inf/NaN are escalated. Raising it trades a larger outlier "
            "list for a narrower (more compressible) log range (default 0.0)"));
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",
            std::string("Fraction of the input element count reserved for outliers (default 0.05). "
            "Note: this stage always interprets the pipeline's error bound value as an exact "
            "point-wise relative bound (delta), regardless of fzgpumodules:error_bound_mode — "
            "set error_bound_mode=rel and pressio:rel so the value you set matches what the "
            "stage does with it."));
    }

private:
    std::map<std::string, LogTransformParams> params_;
    const LogTransformParams defaults_{};

    const LogTransformParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

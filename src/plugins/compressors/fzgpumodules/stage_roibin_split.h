#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct ROIBinSplitParams {
    std::string peaks_file;
    int64_t     roi_half_width = 4;
    int64_t     bin_factor     = 1;
    bool operator==(const ROIBinSplitParams& o) const {
        return peaks_file == o.peaks_file && roi_half_width == o.roi_half_width &&
               bin_factor == o.bin_factor;
    }
    bool operator!=(const ROIBinSplitParams& o) const { return !(*this == o); }
};

// Region-of-interest / background split. Forward: 1 input -> 3 outputs
// (roi, background, meta). Inverse: 3 inputs -> 1 output. Direction swap
// handled transparently by Pipeline::buildInverseDAG() — no plugin-side
// handling needed (confirmed empirically against AdaptiveLorenzoStage).
class ROIBinSplitStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "roibin_split"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "float";
        const auto& p = get_params(sid);

        if(t == "float")  return make<float> (p, ctx);
        if(t == "double") return make<double>(p, ctx);
        throw std::runtime_error("Unsupported roibin_split type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":peaks_file",     p.peaks_file);
        opts.set("fzgpumodules:" + sid + ":roi_half_width", p.roi_half_width);
        opts.set("fzgpumodules:" + sid + ":bin_factor",     p.bin_factor);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = ROIBinSplitParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":peaks_file",     &p.peaks_file);
        opts.get("fzgpumodules:" + sid + ":roi_half_width", &p.roi_half_width);
        opts.get("fzgpumodules:" + sid + ":bin_factor",     &p.bin_factor);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":peaks_file",
            std::string("Path to a .roi peak-list file (magic 'FZROI1', header "
            "nx/ny/nz/npeaks as little-endian uint32, then npeaks records of "
            "{z:uint32, x:uint16, y:uint16} = 8 bytes each). Required before compress; "
            "its geometry must match fzgpumodules:dims if both are set."));
        opts.set("fzgpumodules:" + sid + ":roi_half_width",
            std::string("ROI box half-width in pixels; the box is (2*hw+1)^2 (default 4 -> 9x9)"));
        opts.set("fzgpumodules:" + sid + ":bin_factor",
            std::string("Background binning factor; 1 disables binning (default 1)"));
    }

private:
    std::map<std::string, ROIBinSplitParams> params_;
    const ROIBinSplitParams defaults_{};

    const ROIBinSplitParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename T>
    fz::Stage* make(const ROIBinSplitParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::ROIBinSplitStage<T>>();
        s->setRoiHalfWidth(static_cast<uint32_t>(p.roi_half_width));
        s->setBinFactor(static_cast<uint32_t>(p.bin_factor));
        if(p.peaks_file.empty())
            throw std::runtime_error("roibin_split: fzgpumodules:sN:peaks_file must be set");
        s->setPeaksFile(p.peaks_file);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

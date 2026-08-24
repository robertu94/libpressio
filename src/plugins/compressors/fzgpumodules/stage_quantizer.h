#pragma once
#include <limits>
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct QuantizerParams {
    // int64_t/double (not int/float): pressio_options::get() requires an exact
    // type match and libpressio's Python layer always boxes int->int64_t,
    // float->double, so narrower fields here would silently never be set.
    int64_t quant_radius      = 32768;
    double  outlier_capacity  = 0.2;
    bool    zigzag_codes      = false;
    double  value_base        = 0.0;   // 0 = auto-scan; >0 = skip NOA scan
    double  outlier_threshold = std::numeric_limits<double>::infinity();  // |x| >= threshold → lossless
    bool    inplace_outliers  = false;  // requires zigzag_codes=true AND sizeof(TCode)==sizeof(TIn)
    bool    linear_mode      = false;  // no-outlier cuSZp-style raw signed codes; excl. w/ REL, inplace_outliers, zigzag_codes
    bool    dither            = false;  // dithered reconstruction; excl. w/ linear_mode, inplace_outliers
    int64_t dither_seed       = 0;      // Python ints box as int64_t; cast to uint64_t for the setter
    double  dither_strength   = 1.0;
    bool operator==(const QuantizerParams& o) const {
        return quant_radius == o.quant_radius &&
               outlier_capacity == o.outlier_capacity &&
               zigzag_codes == o.zigzag_codes &&
               value_base == o.value_base &&
               outlier_threshold == o.outlier_threshold &&
               inplace_outliers == o.inplace_outliers &&
               linear_mode == o.linear_mode &&
               dither == o.dither &&
               dither_seed == o.dither_seed &&
               dither_strength == o.dither_strength;
    }
    bool operator!=(const QuantizerParams& o) const { return !(*this == o); }
};

class QuantizerStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "quantizer"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& in_t  = (parts.size() > 1) ? parts[1] : "float";
        const std::string& cod_t = (parts.size() > 2) ? parts[2] : "uint16";

        if(in_t == "float"  && cod_t == "uint16") return make<float,  uint16_t>(sid, ctx);
        if(in_t == "float"  && cod_t == "uint32") return make<float,  uint32_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint16") return make<double, uint16_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint32") return make<double, uint32_t>(sid, ctx);
        throw std::runtime_error("Unsupported quantizer types: " + in_t + ":" + cod_t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":quant_radius",      p.quant_radius);
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",  p.outlier_capacity);
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",      p.zigzag_codes);
        opts.set("fzgpumodules:" + sid + ":value_base",        p.value_base);
        opts.set("fzgpumodules:" + sid + ":outlier_threshold", p.outlier_threshold);
        opts.set("fzgpumodules:" + sid + ":inplace_outliers",  p.inplace_outliers);
        opts.set("fzgpumodules:" + sid + ":linear_mode",       p.linear_mode);
        opts.set("fzgpumodules:" + sid + ":dither",            p.dither);
        opts.set("fzgpumodules:" + sid + ":dither_seed",       p.dither_seed);
        opts.set("fzgpumodules:" + sid + ":dither_strength",   p.dither_strength);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = QuantizerParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":quant_radius",      &p.quant_radius);
        opts.get("fzgpumodules:" + sid + ":outlier_capacity",  &p.outlier_capacity);
        opts.get("fzgpumodules:" + sid + ":zigzag_codes",      &p.zigzag_codes);
        opts.get("fzgpumodules:" + sid + ":value_base",        &p.value_base);
        opts.get("fzgpumodules:" + sid + ":outlier_threshold", &p.outlier_threshold);
        opts.get("fzgpumodules:" + sid + ":inplace_outliers",  &p.inplace_outliers);
        opts.get("fzgpumodules:" + sid + ":linear_mode",       &p.linear_mode);
        opts.get("fzgpumodules:" + sid + ":dither",            &p.dither);
        opts.get("fzgpumodules:" + sid + ":dither_seed",       &p.dither_seed);
        opts.get("fzgpumodules:" + sid + ":dither_strength",   &p.dither_strength);
        return p != old;
    }

    std::string outlier_indices_key(const std::string& /*token*/) const override {
        return "outlier_idxs";
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":quant_radius",
            std::string("Quantization radius (bin count / 2)"));
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",
            std::string("Outlier buffer reserve as fraction of total elements"));
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",
            std::string("Zigzag-encode codes before downstream storage"));
        opts.set("fzgpumodules:" + sid + ":value_base",
            std::string("Pre-computed value_range to skip the NOA data scan; 0 = auto. "
            "No effect outside NOA mode."));
        opts.set("fzgpumodules:" + sid + ":outlier_threshold",
            std::string("ABS/NOA: |x| >= threshold stored losslessly (default: infinity = disabled)"));
        opts.set("fzgpumodules:" + sid + ":inplace_outliers",
            std::string("Encode outliers in-place in codes array; requires zigzag_codes=true "
            "and sizeof(TCode)==sizeof(TIn) (default: false)"));
        opts.set("fzgpumodules:" + sid + ":linear_mode",
            std::string("ABS/NOA: no-outlier cuSZp-style raw signed codes, no radius clamp, no "
            "zigzag. Mutually exclusive with error_bound_mode=rel, inplace_outliers, and "
            "zigzag_codes (default: false). A value whose bin exceeds the signed code range "
            "raises a clean exception at compress time (rather than silently wrapping) — "
            "widen the code type, center the data first, or use an outlier-capable quantizer "
            "instead."));
        opts.set("fzgpumodules:" + sid + ":dither",
            std::string("Reconstruct to a deterministic pseudo-random point within the "
            "error-bound interval instead of the bin center, decorrelating reconstruction "
            "error from the signal. Elements that would violate the bound under dithering are "
            "escalated to lossless outliers. Mutually exclusive with linear_mode and "
            "inplace_outliers (default: false)"));
        opts.set("fzgpumodules:" + sid + ":dither_seed",
            std::string("Seed for the deterministic dither offset; persisted in the compressed "
            "header so decode reproduces identical offsets (default: 0)"));
        opts.set("fzgpumodules:" + sid + ":dither_strength",
            std::string("Dither offset amplitude as a fraction of the error bound, in (0, 1]. "
            "1.0 (default) spans the full bin and empirically escalates ~25% of elements to "
            "lossless outliers for smooth data; lower values trade decorrelation for fewer "
            "outliers; 0.0 disables the offset (bit-identical to dither=false)"));
    }

private:
    std::map<std::string, QuantizerParams> params_;
    const QuantizerParams defaults_{};

    const QuantizerParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename TIn, typename TCode>
    fz::Stage* make(const std::string& sid, const StageContext& ctx) {
        const auto& p = get_params(sid);
        auto* s = ctx.pipeline.addStage<fz::QuantizerStage<TIn, TCode>>();
        s->setErrorBound(static_cast<TIn>(ctx.eb));
        s->setErrorBoundMode(ctx.eb_mode);
        s->setQuantRadius(static_cast<int>(p.quant_radius));
        s->setOutlierCapacity(static_cast<float>(p.outlier_capacity));
        s->setZigzagCodes(p.zigzag_codes);
        if(p.value_base > 0.0) s->setValueBase(static_cast<float>(p.value_base));
        s->setOutlierThreshold(static_cast<float>(p.outlier_threshold));
        s->setInplaceOutliers(p.inplace_outliers);
        s->setLinearMode(p.linear_mode);
        s->setDither(p.dither);
        s->setDitherSeed(static_cast<uint64_t>(p.dither_seed));
        s->setDitherStrength(static_cast<float>(p.dither_strength));
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

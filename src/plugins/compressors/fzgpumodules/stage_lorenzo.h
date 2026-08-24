#pragma once
#include <limits>
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct LorenzoParams {
    // int64_t/double (not int/float): pressio_options::get() requires an exact
    // type match and libpressio's Python layer always boxes int->int64_t,
    // float->double, so narrower fields here would silently never be set.
    int64_t quant_radius     = 32768;
    double  outlier_capacity = 0.2;
    bool    zigzag_codes     = false;
    double  value_base       = 0.0;  // 0 = auto-scan; >0 = skip scan (NOA: value_range, PREL: max|data|)
    bool    centering        = false; // per-tile mean centering (FSZ); 1-D only
    bool operator==(const LorenzoParams& o) const {
        return quant_radius == o.quant_radius &&
               outlier_capacity == o.outlier_capacity &&
               zigzag_codes == o.zigzag_codes &&
               value_base == o.value_base &&
               centering == o.centering;
    }
    bool operator!=(const LorenzoParams& o) const { return !(*this == o); }
};

class LorenzoStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "lorenzo"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& in_t = (parts.size() > 1) ? parts[1] : "float";

        // Integer (lossless) Lorenzo: no configurable params
        if(in_t == "int8")  return ctx.pipeline.addStage<fz::LorenzoStage<int8_t>>();
        if(in_t == "int16") return ctx.pipeline.addStage<fz::LorenzoStage<int16_t>>();
        if(in_t == "int32") return ctx.pipeline.addStage<fz::LorenzoStage<int32_t>>();
        if(in_t == "int64") return ctx.pipeline.addStage<fz::LorenzoStage<int64_t>>();

        // Quantizing Lorenzo: error-bound-driven
        std::string cod_t = (parts.size() > 2 && !parts[2].empty()) ? parts[2] : "uint16";
        if(in_t == "float"  && cod_t == "uint16") return make<float,  uint16_t>(sid, ctx);
        if(in_t == "float"  && cod_t == "uint8")  return make<float,  uint8_t> (sid, ctx);
        if(in_t == "double" && cod_t == "uint16") return make<double, uint16_t>(sid, ctx);
        if(in_t == "double" && cod_t == "uint32") return make<double, uint32_t>(sid, ctx);
        throw std::runtime_error("Unsupported lorenzo types: " + in_t + ":" + cod_t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& token) const override {
        if(is_int(token)) return;
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":quant_radius",     p.quant_radius);
        opts.set("fzgpumodules:" + sid + ":outlier_capacity", p.outlier_capacity);
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",     p.zigzag_codes);
        opts.set("fzgpumodules:" + sid + ":value_base",       p.value_base);
        opts.set("fzgpumodules:" + sid + ":centering",        p.centering);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     token) override {
        if(is_int(token)) return false;
        if(params_.count(sid) == 0) params_[sid] = LorenzoParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":quant_radius",     &p.quant_radius);
        opts.get("fzgpumodules:" + sid + ":outlier_capacity", &p.outlier_capacity);
        opts.get("fzgpumodules:" + sid + ":zigzag_codes",     &p.zigzag_codes);
        opts.get("fzgpumodules:" + sid + ":value_base",       &p.value_base);
        opts.get("fzgpumodules:" + sid + ":centering",        &p.centering);
        return p != old;
    }

    std::string outlier_indices_key(const std::string& token) const override {
        return is_int(token) ? "" : "outlier_indices";
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& token) const override {
        if(is_int(token)) return;
        opts.set("fzgpumodules:" + sid + ":quant_radius",
            std::string("Quantization radius (bin count / 2)"));
        opts.set("fzgpumodules:" + sid + ":outlier_capacity",
            std::string("Outlier buffer reserve as fraction of total elements"));
        opts.set("fzgpumodules:" + sid + ":zigzag_codes",
            std::string("Zigzag-encode codes before downstream storage"));
        opts.set("fzgpumodules:" + sid + ":value_base",
            std::string("Pre-computed value_range (NOA) or max(|data|) (PREL) to skip data scan; 0 = auto"));
        opts.set("fzgpumodules:" + sid + ":centering",
            std::string("Per-tile mean centering (FSZ adaptive centering): predict each 1024-element "
                         "tile's first element from the tile mean instead of 0. Helps fields with a "
                         "large constant offset (e.g. temperature in Kelvin). 1-D only — rejected by "
                         "the engine if combined with 2-D/3-D dims. Default: false."));
    }

private:
    std::map<std::string, LorenzoParams> params_;
    const LorenzoParams defaults_{};

    static bool is_int(const std::string& token) {
        auto parts = split_str(token, ':');
        if(parts.size() < 2) return false;
        const auto& t = parts[1];
        return t == "int8" || t == "int16" || t == "int32" || t == "int64";
    }

    const LorenzoParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename TIn, typename TCode>
    fz::Stage* make(const std::string& sid, const StageContext& ctx) {
        const auto& p = get_params(sid);
        // Centering adds a "means" output port that addStage() captures at
        // construction time, so it (and everything else) is passed via Config
        // rather than set on the stage pointer after the fact.
        typename fz::LorenzoQuantStage<TIn, TCode>::Config config;
        config.error_bound       = static_cast<float>(ctx.eb);
        config.quant_radius      = static_cast<int>(p.quant_radius);
        config.outlier_capacity  = static_cast<float>(p.outlier_capacity);
        config.zigzag_codes      = p.zigzag_codes;
        config.centering         = p.centering;
        if(p.value_base > 0.0) config.precomputed_value_base = static_cast<float>(p.value_base);
        auto* s = ctx.pipeline.addStage<fz::LorenzoQuantStage<TIn, TCode>>(config);
        // eb_mode goes through the setter so the REL->PREL deprecation-alias
        // warning (resolveApproxRelMode) still fires when applicable.
        s->setErrorBoundMode(ctx.eb_mode);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

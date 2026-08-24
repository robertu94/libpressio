#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct HuffmanParams {
    int64_t bklen                 = 0;      // 0 = keep the type's built-in default
    bool    device_resident       = false;  // false = HostCoordinated (default)
    bool    validate_symbol_range = true;
    // book_source is intentionally limited to PerBlock/Adaptive: Fixed
    // requires a caller-supplied frequency table or model (setFixedBookFromFreq
    // / setFixedBookFromModel) that has no scalar pressio_options equivalent.
    bool    adaptive_book         = false;  // false = PerBlock (default), true = Adaptive
    bool operator==(const HuffmanParams& o) const {
        return bklen == o.bklen && device_resident == o.device_resident &&
               validate_symbol_range == o.validate_symbol_range &&
               adaptive_book == o.adaptive_book;
    }
    bool operator!=(const HuffmanParams& o) const { return !(*this == o); }
};

class HuffmanStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "huffman"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "uint16";
        const auto& p = get_params(sid);

        if(t == "uint8")  return make<uint8_t> (p, ctx);
        if(t == "uint16") return make<uint16_t>(p, ctx);
        if(t == "uint32") return make<uint32_t>(p, ctx);
        throw std::runtime_error("Unsupported huffman type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":bklen",                 p.bklen);
        opts.set("fzgpumodules:" + sid + ":device_resident",       p.device_resident);
        opts.set("fzgpumodules:" + sid + ":validate_symbol_range", p.validate_symbol_range);
        opts.set("fzgpumodules:" + sid + ":adaptive_book",         p.adaptive_book);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = HuffmanParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":bklen",                 &p.bklen);
        opts.get("fzgpumodules:" + sid + ":device_resident",       &p.device_resident);
        opts.get("fzgpumodules:" + sid + ":validate_symbol_range", &p.validate_symbol_range);
        opts.get("fzgpumodules:" + sid + ":adaptive_book",         &p.adaptive_book);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":bklen",
            std::string("Codebook length (rounded up so sizeof(T)*bklen is a multiple of 4); "
            "0 = keep the type's built-in default. KNOWN ENGINE BUG: values close to 65536 "
            "currently crash the GPU kernel (huffman_stage.cu narrows bklen to uint16_t "
            "before the launch, so 65536 itself silently wraps to 0) — keep bklen well under "
            "65536, or size quant_radius on the upstream stage so codes fit the default bklen "
            "(1024 for uint16) instead of raising this."));
        opts.set("fzgpumodules:" + sid + ":device_resident",
            std::string("Forward execution path: false = HostCoordinated (cuSZ coarse path, "
            "default), true = DeviceResident (scan/header assembly stays on GPU)"));
        opts.set("fzgpumodules:" + sid + ":validate_symbol_range",
            std::string("Verify every symbol is in [0, bklen) on the GPU when a codebook is "
            "pinned (default: true). Safe to disable only when the symbol range is guaranteed "
            "by construction upstream (e.g. LorenzoQuantStage zigzag codes with "
            "bklen == 2*quant_radius)."));
        opts.set("fzgpumodules:" + sid + ":adaptive_book",
            std::string("Codebook source: false = PerBlock, a fresh histogram+book on every "
            "forward call (default); true = Adaptive, histogram the first call only and "
            "reuse that codebook forever. (Fixed is not exposed here — it needs a "
            "caller-supplied frequency table.)"));
    }

private:
    std::map<std::string, HuffmanParams> params_;
    const HuffmanParams defaults_{};

    const HuffmanParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename T>
    fz::Stage* make(const HuffmanParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::HuffmanStage<T>>();
        if(p.bklen > 0) s->setBklen(static_cast<uint32_t>(p.bklen));
        s->setExecutionMode(p.device_resident ? fz::HuffmanExecutionMode::DeviceResident
                                               : fz::HuffmanExecutionMode::HostCoordinated);
        s->setBookSource(p.adaptive_book ? fz::HuffmanBookSource::Adaptive
                                          : fz::HuffmanBookSource::PerBlock);
        s->setValidateSymbolRange(p.validate_symbol_range);
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

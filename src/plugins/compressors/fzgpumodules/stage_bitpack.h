#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct BitpackParams {
    // int64_t (not int): pressio_options::get() requires an exact type match
    // and libpressio's Python layer always boxes int->int64_t, so a narrower
    // field here would silently never be set.
    int64_t nbits = 0;  // 0 = stage default (full width); set to power-of-2 to save bits
    bool operator==(const BitpackParams& o) const { return nbits == o.nbits; }
    bool operator!=(const BitpackParams& o) const { return !(*this == o); }
};

class BitpackStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "bitpack"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string t = (parts.size() > 1) ? parts[1] : "uint16";
        const auto& p = get_params(sid);

        if(t == "uint8") {
            auto* s = ctx.pipeline.addStage<fz::BitpackStage<uint8_t>>();
            if(p.nbits > 0) s->setNBits(static_cast<uint8_t>(p.nbits));
            return s;
        }
        if(t == "uint16") {
            auto* s = ctx.pipeline.addStage<fz::BitpackStage<uint16_t>>();
            if(p.nbits > 0) s->setNBits(static_cast<uint8_t>(p.nbits));
            return s;
        }
        if(t == "uint32") {
            auto* s = ctx.pipeline.addStage<fz::BitpackStage<uint32_t>>();
            if(p.nbits > 0) s->setNBits(static_cast<uint8_t>(p.nbits));
            return s;
        }
        throw std::runtime_error("Unsupported bitpack type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":nbits", get_params(sid).nbits);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = BitpackParams{};
        auto  old = params_[sid];
        opts.get("fzgpumodules:" + sid + ":nbits", &params_[sid].nbits);
        return params_[sid] != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":nbits",
            std::string("Bits per element (power of 2; 0 = use stage default i.e. full width). "
            "uint8: 1/2/4/8; uint16: 1/2/4/8/16; uint32: 1/2/4/8/16/32."));
    }

private:
    std::map<std::string, BitpackParams> params_;
    const BitpackParams defaults_{};

    const BitpackParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

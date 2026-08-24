#pragma once
#include "stage_kind.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

struct TiledLorenzoParams {
    // 0 = keep the ndim-derived default (2-D 8x8, 3-D 4x4x4, 1-D 64).
    int64_t tile_x = 0;
    int64_t tile_y = 0;
    int64_t tile_z = 0;
    bool operator==(const TiledLorenzoParams& o) const {
        return tile_x == o.tile_x && tile_y == o.tile_y && tile_z == o.tile_z;
    }
    bool operator!=(const TiledLorenzoParams& o) const { return !(*this == o); }
};

class TiledLorenzoStageKind : public StageKind {
public:
    bool matches(std::string_view kind) const override { return kind == "tiled_lorenzo"; }

    fz::Stage* add_stage(const std::string& token,
                          const std::string& sid,
                          const StageContext& ctx) override {
        auto parts = split_str(token, ':');
        const std::string& t = (parts.size() > 1) ? parts[1] : "int32";
        const auto& p = get_params(sid);

        if(t == "int16") return make<int16_t>(p, ctx);
        if(t == "int32") return make<int32_t>(p, ctx);
        throw std::runtime_error("Unsupported tiled_lorenzo type: " + t);
    }

    void populate_options(pressio_options&   opts,
                           const std::string& sid,
                           const std::string& /*token*/) const override {
        const auto& p = get_params(sid);
        opts.set("fzgpumodules:" + sid + ":tile_x", p.tile_x);
        opts.set("fzgpumodules:" + sid + ":tile_y", p.tile_y);
        opts.set("fzgpumodules:" + sid + ":tile_z", p.tile_z);
    }

    bool read_options(const pressio_options& opts,
                       const std::string&     sid,
                       const std::string&     /*token*/) override {
        if(params_.count(sid) == 0) params_[sid] = TiledLorenzoParams{};
        auto  old = params_[sid];
        auto& p   = params_[sid];
        opts.get("fzgpumodules:" + sid + ":tile_x", &p.tile_x);
        opts.get("fzgpumodules:" + sid + ":tile_y", &p.tile_y);
        opts.get("fzgpumodules:" + sid + ":tile_z", &p.tile_z);
        return p != old;
    }

    void populate_documentation(pressio_options&   opts,
                                 const std::string& sid,
                                 const std::string& /*token*/) const override {
        opts.set("fzgpumodules:" + sid + ":tile_x",
            std::string("Tile extent in x (fast dim); 0 = ndim-derived default. "
            "tile_x*tile_y*tile_z must be in [1, 1024], each extent in [1, 255]."));
        opts.set("fzgpumodules:" + sid + ":tile_y",
            std::string("Tile extent in y; 0 = ndim-derived default (1 for 1-D)."));
        opts.set("fzgpumodules:" + sid + ":tile_z",
            std::string("Tile extent in z; 0 = ndim-derived default (1 for 1-D/2-D)."));
    }

private:
    std::map<std::string, TiledLorenzoParams> params_;
    const TiledLorenzoParams defaults_{};

    const TiledLorenzoParams& get_params(const std::string& sid) const {
        auto it = params_.find(sid);
        return it != params_.end() ? it->second : defaults_;
    }

    template<typename T>
    fz::Stage* make(const TiledLorenzoParams& p, const StageContext& ctx) {
        auto* s = ctx.pipeline.addStage<fz::TiledLorenzoStage<T>>();
        if(p.tile_x > 0 || p.tile_y > 0 || p.tile_z > 0)
            s->setTileShape(static_cast<uint32_t>(p.tile_x),
                             static_cast<uint32_t>(p.tile_y),
                             static_cast<uint32_t>(p.tile_z));
        return s;
    }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

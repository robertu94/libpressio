#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <stdexcept>
#include "libpressio_ext/cpp/options.h"
#include "fzgpumodules.h"
#include "predictors/lorenzo/lorenzo_stage.h"
#include "coders/bitpack/bitpack_stage.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

// =============================================================================
// Shared utilities
// =============================================================================

inline std::vector<std::string> split_str(std::string_view s, char delim) {
    std::vector<std::string> parts;
    std::string_view::size_type start = 0, end;
    while ((end = s.find(delim, start)) != std::string_view::npos) {
        parts.emplace_back(s.substr(start, end - start));
        start = end + 1;
    }
    parts.emplace_back(s.substr(start));
    return parts;
}

// Returns the first colon-delimited token (e.g. "lorenzo" from "lorenzo:float:uint16")
inline std::string_view token_kind(std::string_view token) {
    auto pos = token.find(':');
    return pos == std::string_view::npos ? token : token.substr(0, pos);
}

// =============================================================================
// StageContext — data the factory needs from the pipeline-level plugin
// =============================================================================

struct StageContext {
    fz::Pipeline&      pipeline;
    fz::ErrorBoundMode eb_mode;
    double             eb;     // the actual bound value (abs or rel already selected)
};

// =============================================================================
// StageKind — abstract base for each stage family
//
// Adding a new stage family:
//   1. Create stage_<name>.h with a concrete StageKind subclass.
//   2. #include it in fzgpumodules.cc.
//   3. Push a new instance in fzgpumodules_plugin::make_stage_kinds().
//   Nothing else needs to change.
// =============================================================================

struct StageKind {
    virtual ~StageKind() = default;

    // Returns true if this handler owns tokens with the given kind prefix.
    virtual bool matches(std::string_view kind) const = 0;

    // Create the stage and add it to ctx.pipeline; return the Stage*.
    virtual fz::Stage* add_stage(const std::string& token,
                                  const std::string& sid,
                                  const StageContext& ctx) = 0;

    // Write current param values for sid into opts (called from get_options_impl).
    virtual void populate_options(pressio_options& /*opts*/,
                                   const std::string& /*sid*/,
                                   const std::string& /*token*/) const {}

    // Read param values for sid from opts; return true if anything changed.
    virtual bool read_options(const pressio_options& /*opts*/,
                               const std::string&     /*sid*/,
                               const std::string&     /*token*/) { return false; }

    // Write documentation strings for sid into opts.
    virtual void populate_documentation(pressio_options& /*opts*/,
                                         const std::string& /*sid*/,
                                         const std::string& /*token*/) const {}

    // Returns the name of the outlier-indices output port used to derive the
    // post-compress outlier count (size / sizeof(uint32_t) == count).
    // Returns "" for stages that produce no outlier output.
    virtual std::string outlier_indices_key(const std::string& /*token*/) const { return ""; }
};

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

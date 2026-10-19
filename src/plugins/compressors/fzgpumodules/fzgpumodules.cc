#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <chrono>
#include "std_compat/memory.h"
#include "std_compat/utility.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

#include "pipeline_builder.h"
#include "archive.h"

namespace libpressio { namespace fzgpumodules { namespace fzgpumodules_ns {

using namespace libpressio::compressors;  // brings libpressio_compressor_plugin, compressor_plugins, etc.

// =============================================================================
// RAII helpers
// =============================================================================

struct CudaStream {
    cudaStream_t s = nullptr;
    CudaStream() = default;
    ~CudaStream() { if(s) cudaStreamDestroy(s); }
    CudaStream(CudaStream&& o) noexcept : s(std::exchange(o.s, nullptr)) {}
    CudaStream& operator=(CudaStream&& o) noexcept {
        if(this != &o) { if(s) cudaStreamDestroy(s); s = std::exchange(o.s, nullptr); }
        return *this;
    }
    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;
    operator cudaStream_t() {
        if(!s) {
            const auto error = cudaStreamCreate(&s);
            if(error != cudaSuccess) {
                s = nullptr;
                throw std::runtime_error(std::string("CUDA stream creation failed: ") +
                                         cudaGetErrorString(error));
            }
        }
        return s;
    }
};

// =============================================================================
// Main plugin class
// =============================================================================

class fzgpumodules_plugin : public libpressio_compressor_plugin {
public:

    fzgpumodules_plugin() = default;

    // =========================================================================
    // Options interface
    // =========================================================================

    struct pressio_options get_options_impl() const override {
        struct pressio_options options;

        set(options, "pressio:abs", config_.error_bound_abs);
        set(options, "pressio:rel", config_.error_bound_rel);
        set(options, "fzgpumodules:memory_strategy",   config_.memory_strategy);
        set(options, "fzgpumodules:memory_multiplier", config_.memory_multiplier);
        set(options, "fzgpumodules:error_bound_mode",  config_.error_bound_mode);
        set(options, "fzgpumodules:dims", pressio_data(config_.dims.begin(), config_.dims.end()));
        set(options, "fzgpumodules:num_streams", config_.num_streams);
        set(options, "fzgpumodules:specialization", config_.specialization);
        set(options, "fzgpumodules:stages",               config_.stages);
        set(options, "fzgpumodules:connections",          config_.connections);
        set(options, "fzgpumodules:primary_source",       config_.primary_source);
        set(options, "fzgpumodules:config_file",          config_.config_file);
        set(options, "fzgpumodules:expose_stage_outputs", config_.expose_stage_outputs);

        if (config_.config_file.empty()) {
            config_.for_each_stage([&](const StageKind& sk, const std::string& sid, const std::string& tok) {
                sk.populate_options(options, sid, tok);
            });
        }

        return options;
    }

    struct pressio_options get_configuration_impl() const override {
        struct pressio_options options;
        // Serialize calls across all FZGM instances, not just each individual object.
        set(options, "pressio:thread_safe",  pressio_thread_safety_serialized);
        set(options, "pressio:stability",    "experimental");
        set(options, "fzgpumodules:memory_strategy",  map_keys(MEMORY_STRATEGIES));
        set(options, "fzgpumodules:error_bound_mode", map_keys(ERROR_BOUND_MODES));
        set(options, "fzgpumodules:specialization", map_keys(SPECIALIZATION_POLICIES));
        set(options, "predictors:error_dependent",
            std::vector<std::string>{"fzgpumodules:stages", "fzgpumodules:connections"});
        set(options, "predictors:error_agnostic",
            std::vector<std::string>{"fzgpumodules:stages", "fzgpumodules:connections"});
        set(options, "predictors:runtime",
            std::vector<std::string>{"fzgpumodules:memory_strategy", "fzgpumodules:num_streams",
                                      "fzgpumodules:specialization"});
        return options;
    }

    struct pressio_options get_documentation_impl() const override {
        struct pressio_options options;

        set(options, "pressio:description",
            "FZGPUModules GPU compressor with user-composable pipeline stages.\n\n"
            "Configure the pipeline by setting `fzgpumodules:stages` (ordered list of stage tokens) "
            "and `fzgpumodules:connections` (list of wiring strings), or use a TOML config file. "
            "Encoding requires an explicit pipeline; the default error bound is 1e-3. "
            "Set `fzgpumodules:dims` for multi-dimensional data (default: infer 1-D from input size). "
            "Use one GPU with the standard CUDA runtime primary context. Serialize calls across "
            "all FZGM instances; do not switch devices or reset the context while objects are alive. "
            "Host inputs produce host archives; GPU inputs preserve device residency. "
            "A fresh decoder reconstructs its pipeline from the archive metadata.");
        set(options, "pressio:abs",
            "Absolute error bound. Also holds the bound fraction for fzgpumodules:error_bound_mode "
            "'noa' (value-range fraction) and 'prel' (fraction of max|data|).");
        set(options, "pressio:rel",
            "Point-wise relative error bound (used when fzgpumodules:error_bound_mode is 'rel')");
        set(options, "fzgpumodules:error_bound_mode",
            "Error bound mode: abs (default), rel, noa, or prel.\n"
            "  abs  — absolute error bound (pressio:abs).\n"
            "  rel  — point-wise relative bound (pressio:rel). Exact for QuantizerStage-backed "
            "stages. Stages that quantize a fused prediction residual against one global "
            "tolerance (e.g. the lorenzo:float/double tokens, which build LorenzoQuantStage) "
            "cannot honor an exact point-wise bound: they treat 'rel' as a deprecated alias "
            "for 'prel' and log a warning. Use 'prel' explicitly on those stages to silence it.\n"
            "  noa  — value-range relative bound (pressio:abs as a fraction of the data's value range).\n"
            "  prel — pseudo-relative: abs_eb = pressio:abs * max(|data|), applied as a single "
            "absolute bound. This is the honest name for what 'rel' silently did on fused "
            "prediction stages before the REL/PREL split.");
        set(options, "fzgpumodules:memory_strategy",
            "GPU memory allocation strategy: preallocate or minimal (default)");
        set(options, "fzgpumodules:memory_multiplier",
            "Memory allocation multiplier (e.g. 3.0 = 3x input size)");
        set(options, "fzgpumodules:dims",
            "Spatial dimensions [x, y, z] (x = fast). Default {0,1,1} infers 1-D from input size.");
        set(options, "fzgpumodules:num_streams",
            "Number of parallel CUDA streams (default: 1)");
        set(options, "fzgpumodules:specialization",
            "Pipeline Specialization policy: off (default) or auto. Auto installs admitted "
            "implementations for supported stage combinations and otherwise uses staged execution. "
            "It is not a per-input profitability predictor. "
            "FZ_SPECIALIZE overrides the policy at pipeline-build time. "
            "Inspect config_.specialization* metrics after encode/decode to confirm installation. "
            "Use fzgpumodules:specialization in Python and C++ configurations.");
        set(options, "fzgpumodules:stages",
            "Ordered list of stage tokens.\n"
            "Quantizing:        lorenzo:float:uint8/uint16, lorenzo:double:uint16/uint32,\n"
            "                   quantizer:float:uint16/uint32, quantizer:double:uint16/uint32.\n"
            "Integer Lorenzo:   lorenzo:int8, lorenzo:int16, lorenzo:int32, lorenzo:int64.\n"
            "Tiled predictor:   tiled_lorenzo:int16, tiled_lorenzo:int32 (dimension-aware,\n"
            "                   separable Lorenzo; tile-major output for AdaptiveBitpack).\n"
            "Adaptive Lorenzo:  adaptive_lorenzo:int16/32 (lossless, 1 in/3 out fwd).\n"
            "Interpolation:     ginterp (cuSZ-Hi interpolation, 2-D/3-D only, 1 in/4 out fwd).\n"
            "Transforms:        zigzag:int8/16/32/64, negabinary:int8/16/32/64,\n"
            "                   log_transform (float, exact point-wise rel bound, 1 in/4 out fwd).\n"
            "Difference:        diff:float, diff:double, diff:uint8/16/32, diff:int32/64,\n"
            "                   diff:int8:uint8, diff:int16:uint16, diff:int32:uint32, diff:int64:uint64.\n"
            "Entropy coders:    rle:uint8/16/32, bitpack:uint8/16/32, adaptive_bitpack:int16/32,\n"
            "                   huffman:uint8/16/32, ans.\n"
            "LC byte reducers:  rze, rre, rare, raze, clog, hclog (all take chunk_size/word_size).\n"
            "Byte-level:        bitshuffle, gpulz, tupl, bitplane_rze (fixed uint16 input).\n"
            "Structural:        roibin_split:float/double (needs fzgpumodules:sN:peaks_file,\n"
            "                   1 in/3 out fwd).\n"
            "The stages marked 'N in/M out fwd' above have a port count that swaps with\n"
            "direction (M in/N out on decompress) — Pipeline::buildInverseDAG() reconciles\n"
            "this transparently, nothing special to configure.\n"
            "See each stage's fzgpumodules:sN:<param> documentation (get_documentation) for "
            "per-stage tuning knobs, or the TOML config_file path for stages not listed here "
            "(Merge, GPULZ split_mode — these need the stage's forward port count to *grow* "
            "after construction based on a config value, which addRawStage() doesn't support "
            "since it captures port counts once, right after addStage() returns).");
        set(options, "fzgpumodules:connections",
            "Wiring strings of the form 'sN <- sM' or 'sN <- sM:port'. The reserved source name "
            "'__external__' binds the pipeline's raw input directly to sN's next input port "
            "(Pipeline::bindExternalInput()) instead of another stage's output — e.g. "
            "'s2 <- __external__' — for a stage that needs the untouched original input on one "
            "port while another port connects normally. See "
            "fzgpumodules:primary_source when this creates more than one source stage.");
        set(options, "fzgpumodules:primary_source",
            "Stage id (e.g. 's2') whose inverse output decompress() returns, for pipelines with "
            "more than one source stage (see the '__external__' connection above). Empty string "
            "(default) uses the sole/first-discovered source, which is almost never what you want "
            "once a second source exists — set this explicitly whenever fzgpumodules:connections "
            "contains a '__external__' entry alongside a stage that also has a real connection.");
        set(options, "fzgpumodules:config_file",
            "Path to a TOML pipeline config file. When non-empty, the file is the source of "
            "truth for stages, connections, dims, and per-stage error bounds. Options "
            "fzgpumodules:stages, fzgpumodules:connections, fzgpumodules:dims, and pressio:abs "
            "are ignored. fzgpumodules:num_streams and "
            "fzgpumodules:memory_multiplier may still be set via libpressio options. "
            "Per-stage outlier_count metrics are not available in config-file mode. "
            "Set to empty string to revert to the stages/connections API. "
            "See FZGPUModules docs/config_file.md for the TOML format.");
        set(options, "fzgpumodules:expose_stage_outputs",
            "When true, each unconnected stage output port is exposed as a pressio_data "
            "metric under 'fzgpumodules:sN:output:<port>' after compress(). "
            "For a single-stage pipeline this lets you inspect the raw data on both "
            "sides of the transformation. Not available in config_file mode.");
        set(options, "fzgpumodules:peak_memory",
            "[Output] Peak core pipeline pool usage from last compression (bytes); excludes host and external input/archive allocations.");
        set(options, "fzgpumodules:execution_time_us",
            "[Output] Wall time around the core compression call (microseconds), including "
            "its host dispatch and synchronization. Excludes pipeline construction, input upload, "
            "archive assembly, and host output transfer; not CUDA-event kernel timing or "
            "end-to-end Python latency. See fzgpumodules:rebuilt for build status.");
        set(options, "fzgpumodules:rebuilt",
            "[Output] True if the last compress() call (re)built the pipeline (first call, "
            "dirty options, or — in manual-pipeline mode — a changed input size) rather than "
            "reusing the cached one. execution_time_us excludes build cost either way, so a "
            "timing loop should discard measurements where this is true, or warm up first.");

        if (config_.config_file.empty()) {
            config_.for_each_stage([&](const StageKind& sk, const std::string& sid, const std::string& tok) {
                sk.populate_documentation(options, sid, tok);
                if(!sk.outlier_indices_key(tok).empty())
                    set(options, "fzgpumodules:" + sid + ":outlier_count",
                        std::string("[Output] Actual outlier count from last compress"));
            });
        }

        return options;
    }

    int set_options_impl(const pressio_options& options) override {
        try {
            auto candidate = config_;
            const bool stage_changed = candidate.read_options(options,
                [this](const auto& opts, const std::string& key, auto* value) {
                    return get(opts, key, value);
                });
            const bool changed = stage_changed || !config_.same_values(candidate);
            config_.swap(candidate);
            state_.dirty |= changed;
            return 0;
        } catch(const std::exception& error) {
            return set_error(1, std::string("Invalid FZGM configuration: ") + error.what());
        }
    }

    // =========================================================================
    // Compression / decompression
    // =========================================================================

    int compress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            if(config_.stages.empty() && config_.config_file.empty())
                return set_error(1, "Encoding requires an explicit pipeline: set "
                                    "fzgpumodules:stages or fzgpumodules:config_file");
            const auto& types = archive_dtypes();
            if(std::find(types.begin(), types.end(), input->dtype()) == types.end())
                return set_error(1, "Unsupported data type for FZGpuModules compression");
            static_cast<cudaStream_t>(stream_); // Allocate inside the libpressio error boundary.
            return compress_device(input, output);
        } catch(std::exception const& ex) {
            return set_error(2, std::string("Compression exception: ") + ex.what());
        }
    }

    int decompress_impl(const pressio_data* input, struct pressio_data* output) override {
        try {
            static_cast<cudaStream_t>(stream_);
            return decompress_archive(input, output);
        } catch(std::exception const& ex) {
            return set_error(2, std::string("Decompression exception: ") + ex.what());
        }
    }

    // =========================================================================
    // Metadata
    // =========================================================================

    int major_version() const override { return 0; }
    int minor_version() const override { return 1; }
    int patch_version() const override { return 0; }
    const char* version() const override { return "0.1.0"; }
    const char* prefix() const override { return "fzgpumodules"; }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
        auto p = compat::make_unique<fzgpumodules_plugin>();
        p->set_options(get_options());
        return p;
    }

    pressio_options get_metrics_results_impl() const override {
        pressio_options metrics;
        set(metrics, "fzgpumodules:peak_memory",       last_peak_memory_);
        set(metrics, "fzgpumodules:execution_time_us", last_execution_time_us_);
        set(metrics, "fzgpumodules:rebuilt",           last_rebuilt_);
        // Inverse decisions become available after the first decode.
        uint64_t forward_groups = 0, inverse_groups = 0;
        std::vector<std::string> forward_impls, inverse_impls;
        std::string fallback = "pipeline_not_built";
        if (state_.ptr) {
            const auto& info = state_.ptr->getSpecializationInfo();
            forward_groups = info.installed_groups.size();
            inverse_groups = info.installed_inverse_groups.size();
            for (const auto& group : info.installed_groups)
                forward_impls.push_back(group.implementation);
            for (const auto& group : info.installed_inverse_groups)
                inverse_impls.push_back(group.implementation);
            fallback = info.fallback_reason;
        }
        set(metrics, "fzgpumodules:specialization_forward_groups", forward_groups);
        set(metrics, "fzgpumodules:specialization_inverse_groups", inverse_groups);
        set(metrics, "fzgpumodules:specialization_forward_implementations", forward_impls);
        set(metrics, "fzgpumodules:specialization_inverse_implementations", inverse_impls);
        set(metrics, "fzgpumodules:specialization_fallback_reason", fallback);
        for(auto& [sid, count] : last_outlier_counts_)
            set(metrics, "fzgpumodules:" + sid + ":outlier_count", count);
        for(auto& [key, val] : last_stage_outputs_)
            set(metrics, key, val);
        return metrics;
    }

private:
    // =========================================================================
    // Archive compression / decompression
    // =========================================================================

    int compress_device(const pressio_data* real_input, struct pressio_data* output) {
        pressio_data gpu_input = domain_manager().make_readable(
            domain_plugins().build("cudamalloc"), *real_input);
        const void* d_input = gpu_input.data();
        size_t   data_size = gpu_input.size_in_bytes();

        // Config-file pipelines are sized by the TOML's input_size; don't rebuild
        // on data_size change. Manual pipelines must rebuild when data size changes.
        bool need_rebuild = state_.dirty || !state_.ptr ||
                            (config_.config_file.empty() && data_size != state_.last_size);
        last_rebuilt_ = need_rebuild;
        if(need_rebuild) {
            try {
                state_ = build_encoder(config_, data_size);
            } catch(std::exception const& ex) {
                return set_error(3, std::string("Pipeline construction failed: ") + ex.what());
            }
        }

        // compress() returns a BorrowedDeviceBuffer — pool-owned, valid until the
        // next compress()/reset(). We copy it out below into a pressio-owned buffer.
        fz::BorrowedDeviceBuffer comp;

        auto start = std::chrono::high_resolution_clock::now();
        try {
            comp = state_.ptr->compress({d_input, data_size}, stream_);
        } catch(std::exception const& ex) {
            return set_error(4, std::string("Compression execution failed: ") + ex.what());
        }
        const size_t output_size = comp.bytes();
        auto end = std::chrono::high_resolution_clock::now();
        last_execution_time_us_ =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        last_peak_memory_ = state_.ptr->getPeakMemoryUsage();

        last_outlier_counts_.clear();
        for(auto& [sid, ref] : state_.outlier_stages) {
            auto sizes = ref.ptr->getActualOutputSizesByName();
            auto it    = sizes.find(ref.indices_key);
            last_outlier_counts_[sid] = (it != sizes.end())
                ? static_cast<int64_t>(it->second / sizeof(uint32_t)) : 0;
        }

        last_stage_outputs_.clear();
        if (config_.expose_stage_outputs && !state_.terminal_outputs.empty()) {
            std::vector<uint8_t> h_buf(output_size);
            cudaMemcpy(h_buf.data(), comp.data(), output_size, cudaMemcpyDeviceToHost);
            parse_stage_outputs(h_buf.data(), output_size);
        }

        auto metadata = make_archive_metadata(*real_input,
            state_.ptr->serializeHeaderToMemory(), output_size);
        const size_t archive_size = metadata.size() + output_size;
        auto archive = pressio_data::owning(pressio_byte_dtype, {archive_size},
                                           domain_plugins().build("cudamalloc"));
        auto check_cuda = [](cudaError_t error) {
            if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        };
        check_cuda(cudaMemcpyAsync(archive.data(), metadata.data(), metadata.size(),
                                   cudaMemcpyHostToDevice, stream_));
        check_cuda(cudaMemcpyAsync(static_cast<uint8_t*>(archive.data()) + metadata.size(),
                                   comp.data(), output_size, cudaMemcpyDeviceToDevice, stream_));
        check_cuda(cudaStreamSynchronize(stream_));
        const bool wants_device = real_input->domain()->domain_id() == "cudamalloc" ||
                                  output->domain()->domain_id() == "cudamalloc";
        *output = wants_device ? std::move(archive)
            : domain_manager().make_readable(domain_plugins().build("malloc"), std::move(archive));
        return 0;
    }

    int decompress_archive(const pressio_data* input, pressio_data* output) {
        const bool wants_device = output->domain()->domain_id() == "cudamalloc";
        auto compressed = domain_manager().make_readable(domain_plugins().build("cudamalloc"), *input);
        const size_t total = compressed.size_in_bytes();
        if(total < ArchiveMetadata::prefix_size)
            throw std::runtime_error("Truncated FZGM archive prefix");
        auto check_cuda = [](cudaError_t error) {
            if(error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        };
        std::vector<uint8_t> metadata(ArchiveMetadata::prefix_size);
        check_cuda(cudaMemcpy(metadata.data(), compressed.data(), metadata.size(), cudaMemcpyDeviceToHost));
        const size_t metadata_size = archive_metadata_size(metadata.data(), total);
        metadata.resize(metadata_size);
        check_cuda(cudaMemcpy(metadata.data(), compressed.data(), metadata.size(), cudaMemcpyDeviceToHost));
        const auto archive = parse_archive_metadata(metadata.data(), total);
        if(output->num_elements() != 0 &&
           (output->dtype() != archive.dtype || output->dimensions() != archive.dims))
            throw std::runtime_error("FZGM archive dtype/shape does not match requested output");

        if(archive.segment_sizes.size() > 1) {
            const size_t concat_size = 4 + 8*archive.segment_sizes.size();
            if(concat_size > archive.payload_size)
                throw std::runtime_error("Truncated FZGM concat header");
            std::vector<uint8_t> concat(concat_size);
            check_cuda(cudaMemcpy(concat.data(), static_cast<uint8_t*>(compressed.data()) +
                                   archive.payload_offset, concat.size(), cudaMemcpyDeviceToHost));
            if(archive_read(concat.data(), 4) != archive.segment_sizes.size())
                throw std::runtime_error("FZGM concat count disagrees with archive metadata");
            for(size_t i = 0; i < archive.segment_sizes.size(); ++i)
                if(archive_read(concat.data()+4+8*i, 8) != archive.segment_sizes[i])
                    throw std::runtime_error("FZGM concat length disagrees with archive metadata");
        }
        // Reconstruct from this payload, never from the encoder's most recent state.
        auto decoder = fz::Pipeline::fromHeader(archive.header.data(), archive.header.size(),
            MEMORY_STRATEGIES.at(config_.memory_strategy), SPECIALIZATION_POLICIES.at(config_.specialization));
        auto restored = decoder->decompressOwned(
            {static_cast<uint8_t*>(compressed.data()) + archive.payload_offset, archive.payload_size}, stream_);
        if(restored.bytes() != archive.output_size)
            throw std::runtime_error("FZGM decoded size does not match archive shape");
        auto gpu_output = pressio_data::move(archive.dtype, restored.release(), archive.dims,
                                            domain_plugins().build("cudamalloc"));
        *output = wants_device ? std::move(gpu_output)
            : domain_manager().make_readable(domain_plugins().build("malloc"), std::move(gpu_output));
        state_.ptr = std::move(decoder);
        state_.dirty = true; // A later encode must reconstruct its configured forward pipeline.
        state_.outlier_stages.clear();
        state_.terminal_outputs.clear();
        return 0;
    }

    // =========================================================================
    // parse_stage_outputs — maps the compressed blob to per-port pressio_data.
    //
    // Single terminal buffer: pipeline omits the concat header and returns raw
    // stage output directly.  Multi-buffer: concat format from writeConcatBuffer():
    //   [n_bufs: u32][sz_0: u64]...[sz_N-1: u64]  ← padded to 16-byte boundary
    //   [data_0: align16(sz_0) bytes] [data_1: ...] ...
    // =========================================================================

    void parse_stage_outputs(const uint8_t* buf, size_t total) {
        last_stage_outputs_.clear();
        if (state_.terminal_outputs.empty()) return;

        if (state_.terminal_outputs.size() == 1) {
            auto& t = state_.terminal_outputs[0];
            size_t esz = static_cast<size_t>(pressio_dtype_size(t.dtype));
            if (esz == 0) esz = 1;
            last_stage_outputs_[t.metric_key] =
                pressio_data::copy(t.dtype, buf, {total / esz});
            return;
        }

        uint32_t n_bufs = 0;
        std::memcpy(&n_bufs, buf, sizeof(uint32_t));

        std::vector<uint64_t> sizes(n_bufs);
        const uint8_t* hdr = buf + sizeof(uint32_t);
        for (uint32_t i = 0; i < n_bufs; i++) {
            std::memcpy(&sizes[i], hdr, sizeof(uint64_t));
            hdr += sizeof(uint64_t);
        }

        size_t hdr_raw    = sizeof(uint32_t) + n_bufs * sizeof(uint64_t);
        size_t hdr_padded = (hdr_raw + 15u) & ~15u;
        const uint8_t* data_ptr = buf + hdr_padded;

        for (uint32_t i = 0; i < n_bufs && i < static_cast<uint32_t>(state_.terminal_outputs.size()); i++) {
            uint64_t sz = sizes[i];
            auto& t = state_.terminal_outputs[i];
            size_t esz = static_cast<size_t>(pressio_dtype_size(t.dtype));
            if (esz == 0) esz = 1;
            if (sz > 0) {
                last_stage_outputs_[t.metric_key] =
                    pressio_data::copy(t.dtype, data_ptr, {sz / esz});
            } else {
                last_stage_outputs_[t.metric_key] =
                    pressio_data::owning(t.dtype, {0});
            }
            data_ptr += (sz + 15u) & ~15u;
        }
    }

    // =========================================================================
    // Member variables
    // =========================================================================

    CudaStream    stream_;
    PipelineState state_;

    Configuration config_;

    size_t  last_peak_memory_       = 0;
    int64_t last_execution_time_us_ = 0;
    bool    last_rebuilt_           = false;  // did the last compress() rebuild the pipeline?
    std::map<std::string, int64_t>    last_outlier_counts_;
    std::map<std::string, pressio_data> last_stage_outputs_;
};

// =============================================================================
// Plugin registration
// =============================================================================

pressio_register registration(
    compressor_plugins(),
    "fzgpumodules",
    []() { return compat::make_unique<fzgpumodules_plugin>(); }
);

}}} // namespace libpressio::fzgpumodules::fzgpumodules_ns

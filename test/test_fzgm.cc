// test_fzgm.cc -- FZGpuModules plugin tests
//
// Pipeline topology:
//   BasicLinearPipeline  : lorenzo:float:uint16 → diff:uint16
//   ParallelDAGPipeline  : same with small quant_radius/outlier_capacity
// Config file:
//   ConfigFileSimpleLorenzoRLE, ConfigFilePFPL
// Dimensionality:
//   Dims1D_Default, Dims2D_Explicit, Dims3D_Explicit
// Error bound modes:
//   ErrorBoundMode_Abs, ErrorBoundMode_Noa, ErrorBoundMode_Rel
// Pipeline options:
//   NumStreams2, ZigzagCodes, GraphModeMultipleCompress
// Stage types:
//   QuantizerFloatUint16, QuantizerFloatUint32, NewStageTokensCompileCheck, IntegerLorenzoLossless
// Per-stage options:
//   LorenzoValueBaseNoa, QuantizerValueBaseNoa, QuantizerOutlierThreshold, QuantizerInplaceOutliers
// Complex pipelines:
//   BitpackPipeline, BitshuffleRzePipeline
// Single-stage with expose_stage_outputs:
//   SingleStageRle, SingleStageLorenzo
// Regression:
//   RepeatedCompress, TiledLorenzoNoExplicitDims, HuffmanBklenTooLargeRejectsCleanly,
//   HuffmanBklenModerateOverflowRejectsCleanly, AdmOverflowRejectsCleanly
// Validation:
//   InvalidErrorBoundMode, InvalidMemoryStrategy
// Fusion:
//   FusionOff, FusionAuto, InvalidFusion
// Error bound modes (new):
//   ErrorBoundMode_Prel, LorenzoQuantRelAliasMatchesPrel
// Per-stage options (new):
//   LorenzoCentering, QuantizerDither, QuantizerLinearMode
// New stage types:
//   TiledLorenzoLossless, AdaptiveBitpackPipeline, RREPipeline, CLOGPipeline,
//   GPULZPipeline, TUPLPipeline, BitplaneRZEPipeline, ANSPipeline,
//   AdmUint16BoundedInput, AdmUint32BoundedInput, SZxStandaloneAbs, SZpStandaloneAbs,
//   HuffmanWithZigzagCodes
// Direction-dependent port count stages:
//   AdaptiveLorenzoLossless, LogTransformStage, GInterpPipeline, ROIBinSplitPipeline
//
// Control via environment variables:
//   LIBPRESSIO_TEST_VERBOSE  -- if set, print options/metrics to stdout

#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

#include "libpressio.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"

// ── helpers ───────────────────────────────────────────────────────────────────

static bool verbose() { return std::getenv("LIBPRESSIO_TEST_VERBOSE") != nullptr; }

static std::string write_temp_toml(const std::string& content, const std::string& name) {
    std::string path = "/tmp/" + name + ".toml";
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    f << content;
    return path;
}

// Data generators

static pressio_data make_1d(size_t n = 65536) {
    pressio_data d = pressio_data::owning(pressio_float_dtype, {n});
    float* p = static_cast<float*>(d.data());
    for (size_t i = 0; i < n; ++i)
        p[i] = std::sin(2.0f * float(M_PI) * i / n) + 0.5f * std::cos(6.0f * float(M_PI) * i / n);
    return d;
}

static pressio_data make_2d(size_t cols = 256, size_t rows = 256) {
    pressio_data d = pressio_data::owning(pressio_float_dtype, {cols, rows});
    float* p = static_cast<float*>(d.data());
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            *p++ = std::sin(2.0f * float(M_PI) * r / rows)
                 + std::cos(4.0f * float(M_PI) * c / cols)
                 + 0.5f * std::sin(float(M_PI) * (r + c) / (rows + cols));
    return d;
}

static pressio_data make_3d(size_t nx = 64, size_t ny = 64, size_t nz = 16) {
    pressio_data d = pressio_data::owning(pressio_float_dtype, {nx, ny, nz});
    float* p = static_cast<float*>(d.data());
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x)
                *p++ = std::sin(2.0f * float(M_PI) * x / nx)
                     + std::cos(3.0f * float(M_PI) * y / ny)
                     + 0.5f * std::sin(float(M_PI) * z / nz);
    return d;
}

static pressio_metrics make_metrics(pressio& lib) {
    std::vector<std::string> names = {"error_stat", "size"};
    pressio probe;
    for (const char* id : {"time", "memory"}) {
        if (probe.get_metric(id)) names.push_back(id);
    }
    return lib.get_metrics(names.begin(), names.end());
}

// round-trip result
struct RunResult {
    double   compression_ratio = 0;
    double   max_error         = 0;
    double   psnr              = 0;
    double   mse               = 0;
    double   compress_time_ms  = 0;
    double   throughput_mb_s   = 0;
    uint64_t memory_compress   = 0;
};

// compress+decompress round-trip
static RunResult run_pipeline(pressio&               lib,
                              const pressio_options&  opts,
                              pressio_data&           input) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    if (!comp) {
        ADD_FAILURE() << "could not load fzgpumodules: " << lib.err_msg();
        return {};
    }

    auto m = make_metrics(lib);
    if (!m) {
        ADD_FAILURE() << "could not build composite metrics: " << lib.err_msg();
        return {};
    }
    comp->set_metrics(std::move(m));

    if (comp->set_options(opts)) {
        ADD_FAILURE() << "set_options failed: " << comp->error_msg();
        return {};
    }

    if (verbose()) {
        auto cur = comp->get_options();
        std::vector<std::string> stages, conns;
        std::string strat; float mult = 0;
        cur.get("fzgpumodules:stages",           &stages);
        cur.get("fzgpumodules:connections",       &conns);
        cur.get("fzgpumodules:memory_strategy",   &strat);
        cur.get("fzgpumodules:memory_multiplier", &mult);
        std::cout << "  stages:   ";
        for (auto& s : stages) std::cout << s << "  ";
        std::cout << "\n  conns:    ";
        for (auto& c : conns)  std::cout << c << "  ";
        std::cout << "\n  memory:   " << strat << " x" << mult << "\n";
    }

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    if (comp->compress(&input, &compressed)) {
        ADD_FAILURE() << "compress failed: " << comp->error_msg();
        return {};
    }

    pressio_data output = pressio_data::owning(pressio_float_dtype, input.dimensions());
    if (comp->decompress(&compressed, &output)) {
        ADD_FAILURE() << "decompress failed: " << comp->error_msg();
        return {};
    }

    auto results = comp->get_metrics_results();
    if (verbose()) {
        double cr = 0, max_err = 0, psnr = 0;
        uint64_t comp_sz = 0, uncomp_sz = 0;
        results.get("size:compression_ratio", &cr);
        results.get("size:compressed_size",   &comp_sz);
        results.get("size:uncompressed_size", &uncomp_sz);
        results.get("error_stat:max_error",   &max_err);
        results.get("error_stat:psnr",        &psnr);
        std::cout << "  ratio:    " << cr
                  << "  (" << uncomp_sz << " -> " << comp_sz << " bytes)\n"
                  << "  max_err:  " << max_err << "\n"
                  << "  psnr:     " << psnr << " dB\n";
    }

    RunResult r;
    double   cr = 0, max_err = 0, psnr = 0, mse = 0, t_ms = 0;
    uint64_t mem = 0;
    results.get("size:compression_ratio", &cr);
    results.get("error_stat:max_error",   &max_err);
    results.get("error_stat:psnr",        &psnr);
    results.get("error_stat:mse",         &mse);
    results.get("time:compress",          &t_ms);
    results.get("memory:compress",        &mem);

    r.compression_ratio = cr;
    r.max_error         = max_err;
    r.psnr              = psnr;
    r.mse               = mse;
    r.compress_time_ms  = t_ms;
    r.memory_compress   = mem;
    if (t_ms > 0)
        r.throughput_mb_s =
            (input.size_in_bytes() / (1024.0 * 1024.0)) / (t_ms / 1000.0);
    return r;
}

// simple round-trip returning {max_error, compression_ratio}
static std::pair<double, double> roundtrip(pressio&               lib,
                                           const pressio_options&  opts,
                                           pressio_data&           input) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    if (!comp) {
        ADD_FAILURE() << "could not load fzgpumodules: " << lib.err_msg();
        return {-1, -1};
    }

    std::vector<std::string> metric_names{"error_stat", "size"};
    auto m = lib.get_metrics(metric_names.begin(), metric_names.end());
    comp->set_metrics(std::move(m));

    if (comp->set_options(opts)) {
        ADD_FAILURE() << "set_options failed: " << comp->error_msg();
        return {-1, -1};
    }

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    if (comp->compress(&input, &compressed)) {
        ADD_FAILURE() << "compress failed: " << comp->error_msg();
        return {-1, -1};
    }

    pressio_data output = pressio_data::owning(input.dtype(), input.dimensions());
    if (comp->decompress(&compressed, &output)) {
        ADD_FAILURE() << "decompress failed: " << comp->error_msg();
        return {-1, -1};
    }

    auto results = comp->get_metrics_results();
    double max_err = 0, cr = 0;
    results.get("error_stat:max_error",   &max_err);
    results.get("size:compression_ratio", &cr);

    if (verbose()) {
        double psnr = 0; uint64_t comp_sz = 0, uncomp_sz = 0;
        results.get("error_stat:psnr",        &psnr);
        results.get("size:compressed_size",   &comp_sz);
        results.get("size:uncompressed_size", &uncomp_sz);
        std::cout << "  max_err=" << max_err << "  psnr=" << psnr
                  << "  ratio=" << cr
                  << "  (" << uncomp_sz << "->" << comp_sz << ")\n";
    }
    return {max_err, cr};
}

// ── fixture ───────────────────────────────────────────────────────────────────

class FzgpuModulesTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!lib.get_compressor("fzgpumodules"))
            GTEST_SKIP() << "fzgpumodules not available in this build";
        input = make_2d();
    }

    pressio      lib;
    pressio_data input;
};

// ── Pipeline topology ─────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, BasicLinearPipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages", std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s0:quant_radius", static_cast<int64_t>(512));
    opts.set("fzgpumodules:s0:outlier_capacity", 0.15);
    opts.set("fzgpumodules:memory_strategy", std::string("minimal"));
    opts.set("fzgpumodules:memory_multiplier", 3.0);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_LE(r.max_error, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, ParallelDAGPipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages", std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s0:quant_radius", static_cast<int64_t>(32));
    opts.set("fzgpumodules:s0:outlier_capacity", 0.05);
    opts.set("fzgpumodules:memory_strategy", std::string("minimal"));
    opts.set("fzgpumodules:memory_multiplier", 3.0);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_LE(r.max_error, eb + 1e-5);
}

// ── Config file ───────────────────────────────────────────────────────────────

static const char* SIMPLE_LORENZO_RLE_TOML = R"toml(
[pipeline]
dims = [256, 256, 1]
input_size = 262144

[[stage]]
name = "lorz"
type = "LorenzoQuant"
input_type = "float32"
code_type = "uint16"
error_bound = 1e-3
error_bound_mode = "ABS"
quant_radius = 512
outlier_capacity = 0.15

[[stage]]
name = "rle"
type = "RLE"
inputs = [{from = "lorz", port = "codes"}]
)toml";

TEST_F(FzgpuModulesTest, ConfigFileSimpleLorenzoRLE) {
    std::string toml_path;
    ASSERT_NO_THROW(toml_path = write_temp_toml(SIMPLE_LORENZO_RLE_TOML, "fzgm_simple"));

    pressio_options opts;
    opts.set("fzgpumodules:config_file", toml_path);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_GT(r.psnr, 20.0);
}

static const char* PFPL_TOML = R"toml(
[pipeline]
memory_strategy = "PREALLOCATE"
input_size = 262144

[[stage]]
name = "quant"
type = "Quantizer"
input_type = "float32"
code_type = "uint32"
error_bound = 1e-3
error_bound_mode = "REL"
quant_radius = 32768
outlier_capacity = 0.1
zigzag_codes = true

[[stage]]
name = "diff"
type = "Difference"
input_type = "int32"
output_type = "uint32"
chunk_size = 16384
inputs = [{from = "quant", port = "codes"}]

[[stage]]
name = "bshuf"
type = "Bitshuffle"
element_width = 2
block_size = 16384
inputs = [{from = "diff", port = "output"}]

[[stage]]
name = "rze"
type = "RZE"
levels = 4
inputs = [{from = "bshuf", port = "output"}]
)toml";

TEST_F(FzgpuModulesTest, ConfigFilePFPL) {
    std::string toml_path;
    ASSERT_NO_THROW(toml_path = write_temp_toml(PFPL_TOML, "fzgm_pfpl"));

    pressio_options opts;
    opts.set("fzgpumodules:config_file",       toml_path);
    opts.set("fzgpumodules:memory_multiplier", 3.0);

    auto r = run_pipeline(lib, opts, input);

    EXPECT_GT(r.compression_ratio, 1.0);
    EXPECT_GT(r.psnr, 10.0);
}

// ── Dimensionality ────────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, Dims1D_Default) {
    const double eb = 1e-3;
    auto data = make_1d();

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});

    auto [max_err, cr] = roundtrip(lib, opts, data);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, Dims2D_Explicit) {
    const double eb = 1e-3;
    const size_t cols = 256, rows = 256;
    auto data = make_2d(cols, rows);

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{cols, rows, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, data);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, Dims3D_Explicit) {
    const double eb = 1e-3;
    const size_t nx = 64, ny = 64, nz = 16;
    auto data = make_3d(nx, ny, nz);

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{nx, ny, nz}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, data);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// ── Error bound modes ─────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, ErrorBoundMode_Abs) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:error_bound_mode", std::string("abs"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// NOA: abs_eb = user_eb × value_range
TEST_F(FzgpuModulesTest, ErrorBoundMode_Noa) {
    pressio_options opts;
    opts.set("pressio:abs", 0.01);
    opts.set("fzgpumodules:error_bound_mode", std::string("noa"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_GT(max_err, 0.0);
}

TEST_F(FzgpuModulesTest, ErrorBoundMode_Rel) {
    pressio_options opts;
    opts.set("pressio:rel", 0.01);
    opts.set("fzgpumodules:error_bound_mode", std::string("rel"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_GT(max_err, 0.0);
}

// ── Pipeline options ──────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, NumStreams2) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:num_streams", static_cast<int64_t>(2));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, ZigzagCodes) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",              std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections",         std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:zigzag_codes", true);

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// Compress N times without decompressing (CUDA graph captured on first call),
// then decompress the last result and verify the error bound.
TEST_F(FzgpuModulesTest, GraphModeMultipleCompress) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rle:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:graph_mode",  true);
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(comp->compress(&input, &compressed), 0)
            << "compress failed at call " << i << ": " << comp->error_msg();
    }

    pressio_data output = pressio_data::owning(pressio_float_dtype, input.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const float*>(input.data());
    const auto* recon = static_cast<const float*>(output.data());
    float max_err = 0;
    for (size_t i = 0; i < input.num_elements(); ++i)
        max_err = std::max(max_err, std::abs(orig[i] - recon[i]));
    EXPECT_LE(max_err, static_cast<float>(eb + 1e-5));
}

// ── Stage types ───────────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, QuantizerFloatUint16) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// uint32 codes match float word size so CR ~ 1 without an entropy coder; just verify round-trip
TEST_F(FzgpuModulesTest, QuantizerFloatUint32) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 0.5);
    EXPECT_LE(max_err, eb + 1e-5);
}

// Verify all stage token strings are accepted by set_options without error.
// Integer-input stages can't run the float compress path, so we confirm token
// acceptance here and rely on BitshuffleRzePipeline for byte-level end-to-end coverage.
TEST_F(FzgpuModulesTest, NewStageTokensCompileCheck) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    struct TokenCase { std::vector<std::string> stages, conns; };
    std::vector<TokenCase> cases = {
        {{"lorenzo:int32"}, {}},
        {{"lorenzo:int8"},  {}},
        {{"lorenzo:int64"}, {}},
        {{"zigzag:int32"},  {}},
        {{"zigzag:int16"},  {}},
        {{"negabinary:int32"}, {}},
        {{"negabinary:int16"}, {}},
        {{"diff:int32:uint32"}, {}},
        {{"diff:int16:uint16"}, {}},
        {{"diff:int64:uint64"}, {}},
        {{"bitshuffle"}, {}},
        {{"rze"},         {}},
        {{"bitpack:uint16"}, {}},
        {{"bitpack:uint8"},  {}},
        {{"bitpack:uint32"}, {}},
        {{"lorenzo:float:uint8"},    {}},
        {{"lorenzo:double:uint32"},  {}},
        {{"quantizer:double:uint16"}, {}},
        {{"quantizer:double:uint32"}, {}},
        {{"tiled_lorenzo:int16"},     {}},
        {{"tiled_lorenzo:int32"},     {}},
        {{"adaptive_bitpack:int16"},  {}},
        {{"adaptive_bitpack:int32"},  {}},
        {{"rre"},  {}}, {{"rare"}, {}}, {{"raze"}, {}}, {{"clog"}, {}}, {{"hclog"}, {}},
        {{"gpulz"}, {}}, {{"tupl"}, {}}, {{"bitplane_rze"}, {}}, {{"ans"}, {}},
        {{"adm:uint16"}, {}}, {{"adm:uint32"}, {}},
        {{"huffman:uint8"}, {}}, {{"huffman:uint16"}, {}}, {{"huffman:uint32"}, {}},
        {{"szx:float"}, {}}, {{"szx:double"}, {}}, {{"szp:float"}, {}}, {{"szp:double"}, {}},
        {{"adaptive_lorenzo:int16"}, {}}, {{"adaptive_lorenzo:int32"}, {}},
        {{"log_transform"}, {}},
        {{"ginterp"}, {}},
        {{"roibin_split:float"}, {}}, {{"roibin_split:double"}, {}},
    };

    for (auto& tc : cases) {
        pressio_options opts;
        opts.set("pressio:abs", 1e-3);
        opts.set("fzgpumodules:stages",      tc.stages);
        opts.set("fzgpumodules:connections", tc.conns);
        EXPECT_EQ(comp->set_options(opts), 0) << "set_options failed for stage: " << tc.stages[0];
    }
}

// lorenzo:int32 → zigzag:int32 → bitpack:uint32 must be bit-exact
TEST_F(FzgpuModulesTest, IntegerLorenzoLossless) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {65536});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < 65536; ++i)
        p[i] = (i % 4 == 0) ? static_cast<int32_t>(i % 1000) : 0;

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{"lorenzo:int32", "zigzag:int32", "bitpack:uint32"});
    opts.set("fzgpumodules:connections",
             std::vector<std::string>{"s1 <- s0", "s2 <- s1"});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();

    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const int32_t*>(data.data());
    const auto* recon = static_cast<const int32_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "integer Lorenzo round-trip is not bit-exact";
}

// ── Per-stage options ─────────────────────────────────────────────────────────

// Pre-supplying value_base (NOA mode) skips the device data scan
TEST_F(FzgpuModulesTest, LorenzoValueBaseNoa) {
    const float* p = static_cast<const float*>(input.data());
    size_t n = input.num_elements();
    float vmin = *std::min_element(p, p + n);
    float vmax = *std::max_element(p, p + n);
    float value_range = vmax - vmin;
    const double user_eb = 0.01;

    pressio_options opts;
    opts.set("pressio:abs", user_eb);
    opts.set("fzgpumodules:error_bound_mode", std::string("noa"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:value_base", static_cast<double>(value_range));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, user_eb * value_range + 1e-5);
}

TEST_F(FzgpuModulesTest, QuantizerValueBaseNoa) {
    const float* p = static_cast<const float*>(input.data());
    size_t n = input.num_elements();
    float vmin = *std::min_element(p, p + n);
    float vmax = *std::max_element(p, p + n);
    float value_range = vmax - vmin;
    const double user_eb = 0.01;

    pressio_options opts;
    opts.set("pressio:abs", user_eb);
    opts.set("fzgpumodules:error_bound_mode", std::string("noa"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:value_base", static_cast<double>(value_range));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, user_eb * value_range + 1e-5);
}

// outlier_threshold=1.8 affects only values >= 1.8 (near the data max ~2.5);
// lossless outliers have zero error, quantized elements satisfy the ABS bound
TEST_F(FzgpuModulesTest, QuantizerOutlierThreshold) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:outlier_threshold", 1.8);

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_LE(max_err, eb + 1e-5);
}

// inplace_outliers requires zigzag_codes=true and sizeof(TCode)==sizeof(TInput),
// so float input must use uint32 codes
TEST_F(FzgpuModulesTest, QuantizerInplaceOutliers) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:inplace_outliers", true);
    opts.set("fzgpumodules:s0:zigzag_codes",     true);

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_LE(max_err, eb + 1e-5);
}

// ── Complex pipelines ─────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, BitpackPipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{"lorenzo:float:uint16", "bitpack:uint16"});
    opts.set("fzgpumodules:connections",
             std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s1:nbits", static_cast<int64_t>(0));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// quantizer:float:uint16 → bitshuffle (element_width=2) → rze
TEST_F(FzgpuModulesTest, BitshuffleRzePipeline) {
    const double eb = 1e-3;

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{"quantizer:float:uint16", "bitshuffle", "rze"});
    opts.set("fzgpumodules:connections",
             std::vector<std::string>{"s1 <- s0:codes", "s2 <- s1"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s1:element_width", static_cast<int64_t>(2));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// ── Single-stage via expose_stage_outputs ─────────────────────────────────────

TEST_F(FzgpuModulesTest, SingleStageRle) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    pressio_data data = pressio_data::owning(pressio_uint16_dtype, {65536});
    auto* p = static_cast<uint16_t*>(data.data());
    for (size_t i = 0; i < 65536; ++i)
        p[i] = (i % 8 == 0) ? static_cast<uint16_t>(i % 60000) : uint16_t(0);

    pressio_options opts;
    opts.set("fzgpumodules:stages",               std::vector<std::string>{"rle:uint16"});
    opts.set("fzgpumodules:connections",          std::vector<std::string>{});
    opts.set("fzgpumodules:expose_stage_outputs", true);
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();

    pressio_data output = pressio_data::owning(pressio_uint16_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const uint16_t*>(data.data());
    const auto* recon = static_cast<const uint16_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "rle:uint16 round-trip is not exact";

    auto results = comp->get_metrics_results();
    pressio_data out_metric;
    EXPECT_EQ(results.get("fzgpumodules:s0:output:output", &out_metric),
              pressio_options_key_set)
        << "fzgpumodules:s0:output:output metric missing";
    EXPECT_GT(out_metric.num_elements(), size_t(0));
}

TEST_F(FzgpuModulesTest, SingleStageLorenzo) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    pressio_options opts;
    opts.set("fzgpumodules:stages",               std::vector<std::string>{"lorenzo:float:uint16"});
    opts.set("fzgpumodules:connections",          std::vector<std::string>{});
    opts.set("fzgpumodules:expose_stage_outputs", true);
    opts.set("pressio:abs", 1e-3);
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&input, &compressed), 0) << comp->error_msg();

    pressio_data output = pressio_data::owning(pressio_float_dtype, input.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    auto results = comp->get_metrics_results();
    pressio_data codes;
    EXPECT_EQ(results.get("fzgpumodules:s0:output:codes", &codes),
              pressio_options_key_set)
        << "fzgpumodules:s0:output:codes metric missing";
    EXPECT_EQ(codes.dtype(), pressio_uint16_dtype);
    EXPECT_GT(codes.num_elements(), size_t(0));

    const double eb = 1e-3;
    const auto* orig  = static_cast<const float*>(input.data());
    const auto* recon = static_cast<const float*>(output.data());
    float max_err = 0;
    for (size_t i = 0; i < input.num_elements(); ++i)
        max_err = std::max(max_err, std::abs(orig[i] - recon[i]));
    EXPECT_LE(max_err, static_cast<float>(eb + 1e-5));
}

// ── Regression ────────────────────────────────────────────────────────────────

// 5 full cycles on the same compressor object; verifies internal pool buffers
// are not corrupted between calls
TEST_F(FzgpuModulesTest, RepeatedCompress) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    std::vector<std::string> metric_names{"error_stat", "size"};
    auto m = lib.get_metrics(metric_names.begin(), metric_names.end());
    comp->set_metrics(std::move(m));

    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rle:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    for (int cycle = 0; cycle < 5; ++cycle) {
        pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
        ASSERT_EQ(comp->compress(&input, &compressed), 0)
            << "cycle " << cycle << ": " << comp->error_msg();

        pressio_data output = pressio_data::owning(pressio_float_dtype, input.dimensions());
        ASSERT_EQ(comp->decompress(&compressed, &output), 0)
            << "cycle " << cycle << ": " << comp->error_msg();

        double max_err = 0;
        comp->get_metrics_results().get("error_stat:max_error", &max_err);
        EXPECT_LE(max_err, eb + 1e-5) << "error bound violated at cycle " << cycle;
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, InvalidErrorBoundMode) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    pressio_options opts;
    opts.set("fzgpumodules:error_bound_mode", std::string("bogus"));
    EXPECT_NE(comp->set_options(opts), 0);
}

TEST_F(FzgpuModulesTest, InvalidMemoryStrategy) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);

    pressio_options opts;
    opts.set("fzgpumodules:memory_strategy", std::string("pipeline"));
    EXPECT_NE(comp->set_options(opts), 0);
}

// ── Fusion ────────────────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, FusionOff) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rze"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:fusion",      std::string("off"));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, FusionAuto) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rze"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:fusion",      std::string("auto"));

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, InvalidFusion) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:fusion", std::string("bogus"));
    EXPECT_NE(comp->set_options(opts), 0);
}

// ── Error bound modes (new) ─────────────────────────────────────────────────────

// PREL: abs_eb = eb * max(|data|), applied as a single ABS bound on LorenzoQuantStage
TEST_F(FzgpuModulesTest, ErrorBoundMode_Prel) {
    pressio_options opts;
    opts.set("pressio:abs", 0.01);
    opts.set("fzgpumodules:error_bound_mode", std::string("prel"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_GT(max_err, 0.0);
}

// LorenzoQuantStage cannot honour an exact point-wise bound: REL is a deprecated
// alias that resolves to PREL (with a warning). abs=0.01, rel=0.01 should therefore
// produce identical results on this stage.
TEST_F(FzgpuModulesTest, LorenzoQuantRelAliasMatchesPrel) {
    std::vector<size_t> dims{256, 256, 1};

    pressio_options prel_opts;
    prel_opts.set("pressio:abs", 0.01);
    prel_opts.set("fzgpumodules:error_bound_mode", std::string("prel"));
    prel_opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    prel_opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    prel_opts.set("fzgpumodules:dims", pressio_data(dims.begin(), dims.end()));
    auto [prel_err, prel_cr] = roundtrip(lib, prel_opts, input);

    pressio_options rel_opts;
    rel_opts.set("pressio:rel", 0.01);
    rel_opts.set("fzgpumodules:error_bound_mode", std::string("rel"));
    rel_opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "diff:uint16"});
    rel_opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    rel_opts.set("fzgpumodules:dims", pressio_data(dims.begin(), dims.end()));
    auto [rel_err, rel_cr] = roundtrip(lib, rel_opts, input);

    EXPECT_DOUBLE_EQ(prel_err, rel_err);
    EXPECT_DOUBLE_EQ(prel_cr, rel_cr);
}

// ── Per-stage options (new) ──────────────────────────────────────────────────────

// FSZ-style per-tile mean centering; helps fields with a large constant offset. 1-D only.
TEST_F(FzgpuModulesTest, LorenzoCentering) {
    const double eb = 1e-3;
    pressio_data data = make_1d();
    float* p = static_cast<float*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i) p[i] += 2000.0f;  // large constant offset

    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rze"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s0:centering", true);

    auto [max_err, cr] = roundtrip(lib, opts, data);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, QuantizerDither) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    // dither commonly escalates ~25% of elements at strength=1.0; raise capacity to fit.
    // NOTE: outlier_capacity/dither_strength are double (not float) fields —
    // pressio_options::get() requires an exact type match, so a `0.35f` literal
    // here would silently no-op instead of setting the value (the same class of
    // bug the int64_t/double widening in this plugin was fixed for on the Python
    // side; it bites C++ callers too if the literal suffix doesn't match).
    opts.set("fzgpumodules:s0:outlier_capacity", 0.35);
    opts.set("fzgpumodules:s0:dither",           true);
    opts.set("fzgpumodules:s0:dither_seed",      static_cast<int64_t>(42));
    opts.set("fzgpumodules:s0:dither_strength",  0.5);

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 0.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, QuantizerLinearMode) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"quantizer:float:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    opts.set("fzgpumodules:s0:linear_mode", true);

    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 0.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// ── New stage types ───────────────────────────────────────────────────────────

TEST_F(FzgpuModulesTest, TiledLorenzoLossless) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {256, 256});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<int32_t>((i * 37) % 1000) - 500;

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"tiled_lorenzo:int32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const int32_t*>(data.data());
    const auto* recon = static_cast<const int32_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "tiled_lorenzo:int32 round-trip is not bit-exact";
}

// TiledLorenzoStage regression: the default dims [0,1,1] (1-D inference) used to
// under-size the forward output buffer and crash — see paddedElems() in
// tiled_lorenzo_stage.h. This exercises that path with dims left at the default.
TEST_F(FzgpuModulesTest, TiledLorenzoNoExplicitDims) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {65536});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<int32_t>((i * 37) % 1000) - 500;

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"tiled_lorenzo:int32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();
}

TEST_F(FzgpuModulesTest, AdaptiveBitpackPipeline) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {256, 256});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<int32_t>((i * 37) % 1000) - 500;

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",
             std::vector<std::string>{"tiled_lorenzo:int32", "adaptive_bitpack:int32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0"});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const int32_t*>(data.data());
    const auto* recon = static_cast<const int32_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact);
}

TEST_F(FzgpuModulesTest, RREPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "rre"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, CLOGPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "clog"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, GPULZPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "gpulz"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, TUPLPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "tupl"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 0.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, BitplaneRZEPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "bitplane_rze"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, ANSPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "ans"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, HuffmanWithZigzagCodes) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "huffman:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s0:zigzag_codes",     true);
    opts.set("fzgpumodules:s0:quant_radius",     static_cast<int64_t>(512));
    // Outliers get an escape code outside the [0, 2*quant_radius) in-range zigzag
    // span, so bklen needs headroom beyond 2*radius, not just exactly 2*radius.
    opts.set("fzgpumodules:s0:outlier_capacity", 0.2);
    opts.set("fzgpumodules:s1:bklen",            static_cast<int64_t>(1088));
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// Regression: bklen_ is uint16_t end to end (serialized header, histogram kernel
// API); 65536 used to silently wrap to 0 in a narrowing cast. setBklen() now
// validates and throws instead.
TEST_F(FzgpuModulesTest, HuffmanBklenTooLargeRejectsCleanly) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "huffman:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s1:bklen",    static_cast<int64_t>(65536));
    EXPECT_EQ(comp->set_options(opts), 0) << comp->error_msg();  // set_options is lazy; the
                                                                  // throw happens at build time

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    EXPECT_NE(comp->compress(&input, &compressed), 0)
        << "bklen=65536 should have been rejected";
}

// Regression: a hist_len whose privatized histogram doesn't fit in shared memory
// (even one replica) used to divide by zero on-device (repeat=0 in the p2013
// histogram kernel). The optimizer now validates and throws before launch.
TEST_F(FzgpuModulesTest, HuffmanBklenModerateOverflowRejectsCleanly) {
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"lorenzo:float:uint16", "huffman:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{"s1 <- s0:codes"});
    opts.set("fzgpumodules:s1:bklen",    static_cast<int64_t>(65534));
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    EXPECT_NE(comp->compress(&input, &compressed), 0)
        << "bklen=65534 should have been rejected on this device";
}

TEST_F(FzgpuModulesTest, AdmUint16BoundedInput) {
    pressio_data data = pressio_data::owning(pressio_uint16_dtype, {256, 256});
    auto* p = static_cast<uint16_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<uint16_t>(10000 + (i % 100) - 50);

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"adm:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_uint16_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const uint16_t*>(data.data());
    const auto* recon = static_cast<const uint16_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "adm:uint16 round-trip is not exact";
}

TEST_F(FzgpuModulesTest, AdmUint32BoundedInput) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {256, 256});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<int32_t>(10000 + (i % 100) - 50);

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"adm:uint32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const int32_t*>(data.data());
    const auto* recon = static_cast<const int32_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "adm:uint32 round-trip is not exact";
}

// Regression: ADM's overflow guard used to be compiled out in Release builds
// (#ifndef NDEBUG), so out-of-spec input (diffs exceeding algorithm capacity)
// silently corrupted device memory instead of raising. Confirms it now raises.
TEST_F(FzgpuModulesTest, AdmOverflowRejectsCleanly) {
    pressio_data data = pressio_data::owning(pressio_uint16_dtype, {256, 256});
    auto* p = static_cast<uint16_t*>(data.data());
    // Alternating extremes: adjacent-element diffs from the block center are
    // near the full uint16 range, far beyond ADM's bounded-diff assumption.
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = (i % 2 == 0) ? uint16_t(0) : uint16_t(65535);

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"adm:uint16"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    EXPECT_NE(comp->compress(&data, &compressed), 0)
        << "out-of-spec ADM input should have been rejected, not silently corrupted";
}

TEST_F(FzgpuModulesTest, SZxStandaloneAbs) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"szx:float"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, SZpStandaloneAbs) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"szp:float"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

// ── Direction-dependent port count stages ────────────────────────────────────────
//
// These stages' getNumOutputs()/getNumInputs() swap with direction (more ports
// forward/compress, fewer inverse/decompress, or vice versa). Confirmed
// Pipeline::buildInverseDAG() reconciles this transparently — nothing
// stage-count-related for the plugin to do beyond the normal addStage() call.

TEST_F(FzgpuModulesTest, AdaptiveLorenzoLossless) {
    pressio_data data = pressio_data::owning(pressio_int32_dtype, {65536});
    auto* p = static_cast<int32_t*>(data.data());
    for (size_t i = 0; i < data.num_elements(); ++i)
        p[i] = static_cast<int32_t>((i * 37) % 1000) - 500;

    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"adaptive_lorenzo:int32"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_int32_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const int32_t*>(data.data());
    const auto* recon = static_cast<const int32_t*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "adaptive_lorenzo:int32 round-trip is not bit-exact";
}

TEST_F(FzgpuModulesTest, LogTransformStage) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:rel", eb);
    opts.set("fzgpumodules:error_bound_mode", std::string("rel"));
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"log_transform"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 0.0);
    EXPECT_GT(max_err, 0.0);
}

TEST_F(FzgpuModulesTest, GInterpPipeline) {
    const double eb = 1e-3;
    pressio_options opts;
    opts.set("pressio:abs", eb);
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"ginterp"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    { auto d = std::vector<size_t>{256, 256, 1}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    auto [max_err, cr] = roundtrip(lib, opts, input);
    EXPECT_GT(cr, 1.0);
    EXPECT_LE(max_err, eb + 1e-5);
}

TEST_F(FzgpuModulesTest, ROIBinSplitPipeline) {
    const size_t nx = 64, ny = 64, nz = 1;
    std::string roi_path = "/tmp/fzgm_test.roi";
    {
        std::ofstream f(roi_path, std::ios::binary);
        ASSERT_TRUE(!!f);
        f.write("FZROI1\x00\x00", 8);
        uint32_t hdr[4] = {static_cast<uint32_t>(nx), static_cast<uint32_t>(ny),
                           static_cast<uint32_t>(nz), 2u};
        f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        struct { uint32_t z; uint16_t x, y; } peaks[2] = {{0, 20, 20}, {0, 40, 40}};
        f.write(reinterpret_cast<const char*>(peaks), sizeof(peaks));
    }

    pressio_data data = make_2d(nx, ny);
    pressio_compressor comp = lib.get_compressor("fzgpumodules");
    ASSERT_TRUE(!!comp);
    pressio_options opts;
    opts.set("fzgpumodules:stages",      std::vector<std::string>{"roibin_split:float"});
    opts.set("fzgpumodules:connections", std::vector<std::string>{});
    opts.set("fzgpumodules:s0:peaks_file", roi_path);
    { auto d = std::vector<size_t>{nx, ny, nz}; opts.set("fzgpumodules:dims", pressio_data(d.begin(), d.end())); }
    ASSERT_EQ(comp->set_options(opts), 0) << comp->error_msg();

    pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
    ASSERT_EQ(comp->compress(&data, &compressed), 0) << comp->error_msg();
    pressio_data output = pressio_data::owning(pressio_float_dtype, data.dimensions());
    ASSERT_EQ(comp->decompress(&compressed, &output), 0) << comp->error_msg();

    const auto* orig  = static_cast<const float*>(data.data());
    const auto* recon = static_cast<const float*>(output.data());
    bool exact = true;
    for (size_t i = 0; i < data.num_elements() && exact; ++i)
        if (orig[i] != recon[i]) exact = false;
    EXPECT_TRUE(exact) << "roibin_split:float round-trip is not exact";
}

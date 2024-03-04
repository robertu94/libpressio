#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_manager.h>
#include <cuda_runtime.h>
#include <sstream>




struct tracking : public pressio_domain_manager_metrics_plugin {
    void view_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "view_begin " << dst->prefix() << '<' << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void view_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "view_end " << dst->prefix() << '<' << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void alloc_begin(std::shared_ptr<pressio_domain> const& dom) override {
        std::stringstream ss;
        ss << "alloc_begin " << dom->prefix();
        events.emplace_back(ss.str());
    }
    void alloc_end(std::shared_ptr<pressio_domain> const& dom) override {
        std::stringstream ss;
        ss << "alloc_end " << dom->prefix();
        events.emplace_back(ss.str());
    }
    void send_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "send_begin " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void make_readable_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_begin " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void make_readable_domain_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_domain_begin " << dst->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void copy_to_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "copy_to_begin " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void send_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "send_end " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void make_readable_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_end " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void make_readable_domain_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_end " << dst->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }
    void copy_to_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "copy_to_end " << dst.domain()->prefix() << "<" << src.domain()->prefix();
        events.emplace_back(ss.str());
    }

    domain_options get_metrics_results() override {
        domain_options opts;
        opts["tracking:events"] = events;
        return opts;
    }
    int set_options(domain_options const& opts) override {
        if(opts.find("tracking:reset") != opts.end()) {
            events.clear();
        }
        return 0;
    }
    virtual std::unique_ptr<pressio_domain_manager_metrics_plugin> clone() const override {
        return std::make_unique<tracking>();
    }

    std::vector<std::string> events;
};

TEST(Domains, TestDomains) {
    if(!domain_plugins().build("cudamalloc")) {
        GTEST_SKIP() << "this test requires cuda";
    }

    //force everything to go out of scope
    ASSERT_EQ(1,1); // have some instruction before scope ends
    {
        std::vector<size_t> dims{5,5};
        auto src = pressio_data::owning(pressio_float_dtype, dims, domain_plugins().build("cudamallochost"));
        float* src_ptr = static_cast<float*>(src.data());
        for (size_t i = 0; i < dims[1]; ++i) {
            for (size_t j = 0; j < dims[0]; ++j) {
                src_ptr[j+dims[0]*i] = static_cast<float>(i * j);
            }
        }

        pressio_domain_manager mgr;
        mgr.set_metrics(tracking{});
        auto test_readable = [&](
                auto&& dst,
                pressio_data const& src,
                const char* desc,
                std::vector<std::string> const& expected,
                std::vector<std::string> const& exclude = {}
        ) {
            auto target = mgr.make_readable(std::forward<decltype(dst)>(dst), src);
            auto results = mgr.get_metrics_results();
            auto events = std::get<std::vector<std::string>>(results.at("tracking:events"));
            mgr.set_options({{"tracking:reset", true}});
            for (auto const& e : expected) {
                EXPECT_THAT(events, ::testing::Contains(e)) << desc;
            }
            for (auto const& e : exclude) {
                EXPECT_THAT(events, ::testing::Not(::testing::Contains(e))) << desc;
            }
            return target;
        };

        auto malloc_tgt = test_readable(pressio_data::owning(pressio_float_dtype, dims, domain_plugins().build("malloc")), src, 
                      "cudamallochost->malloc: should copy",
                      {"readable_begin malloc<cudamallochost", "view_begin malloc<cudamallochost"},
                      {"alloc_begin malloc", "send_begin malloc<cudamallochost"}
                      );
        ASSERT_EQ(malloc_tgt, src);
        auto cuda_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, domain_plugins().build("cudamalloc")), src, 
                      "cudamallochost->cudamalloc: should send",
                      {"readable_begin cudamalloc<cudamallochost", "send_begin cudamalloc<cudamallochost"},
                      {"alloc_begin cudamalloc"}
                      );
        auto mallochost_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, domain_plugins().build("cudamallochost")), src, 
                      "cudamallochost->cudamallochost: should view",
                      {"readable_begin cudamallochost<cudamallochost","view_begin cudamallochost<cudamallochost" },
                      {"alloc_begin cudamalloc", "send_begin cudamalloc<cudamallochost"}
                      );
        ASSERT_EQ(mallochost_tgt, src);

        test_readable(pressio_data::owning(pressio_float_dtype, dims, domain_plugins().build("malloc")), cuda_tgt, 
                      "cudamalloc->malloc: should send",
                      {"readable_begin malloc<cudamalloc", "send_begin malloc<cudamalloc"},
                      {"alloc_begin malloc"});

        auto malloc_domtgt = test_readable(domain_plugins().build("malloc"), src, 
                      "cudamallochost->malloc: should view",
                      {"readable_domain_begin malloc<cudamallochost", "view_begin malloc<cudamallochost"},
                      {"send_begin malloc<cudamallochost"}
                      );
        test_readable(domain_plugins().build("malloc"), cuda_tgt, 
                      "cudamalloc->malloc: should send",
                      {"readable_domain_begin malloc<cudamalloc", "send_begin malloc<cudamalloc", "alloc_begin malloc"},
                      {});
    }
    ASSERT_EQ(1,1); // have some instruction after scope ends
}

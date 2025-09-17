#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <libpressio_ext/cpp/data.h>
#include <libpressio_ext/cpp/domain.h>
#include <libpressio_ext/cpp/domain_manager.h>
#include <libpressio_ext/cpp/printers.h>
#include <sstream>
#include <string_view>


using namespace ::libpressio::domains;

namespace libpressio { namespace domains_metrics {

struct tracking : public pressio_domain_manager_metrics_plugin {
    static std::string const& if_domain_prefix(pressio_data const& src) {
        static std::string moved{"{moved}"};
        if(src.domain()) {
            return src.domain()->prefix();
        } else {
            return moved;
        }
    }
    void view_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "view_begin " << dst->prefix() << '<' << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void view_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "view_end " << dst->prefix() << '<' << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void alloc_begin(std::shared_ptr<pressio_domain> const& dom, pressio_dtype, std::vector<size_t> const&) override {
        std::stringstream ss;
        ss << "alloc_begin " << dom->prefix();
        events.emplace_back(ss.str());
    }
    void alloc_end(std::shared_ptr<pressio_domain> const& dom, pressio_dtype, std::vector<size_t> const&) override {
        std::stringstream ss;
        ss << "alloc_end " << dom->prefix();
        events.emplace_back(ss.str());
    }
    void send_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "send_begin " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void make_readable_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_begin " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void make_readable_domain_begin(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_domain_begin " << dst->prefix() << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void copy_to_begin(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "copy_to_begin " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void send_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "send_end " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void make_readable_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_end " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void make_readable_domain_end(std::shared_ptr<pressio_domain> const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "readable_end " << dst->prefix() << "<" << if_domain_prefix(src);
        events.emplace_back(ss.str());
    }
    void copy_to_end(pressio_data const& dst, pressio_data const& src) override {
        std::stringstream ss;
        ss << "copy_to_end " << if_domain_prefix(dst) << "<" << if_domain_prefix(src);
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
}}

TEST(Domains, TestReadableDomains) {
    if(!libpressio::domain_plugins().build("cudamalloc")) {
        GTEST_SKIP() << "this test requires cuda";
    }

    //force everything to go out of scope
    ASSERT_EQ(1,1); // have some instruction before scope ends
    {
        std::vector<size_t> dims{50,50};
        auto src = pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamallochost"));
        float* src_ptr = static_cast<float*>(src.data());
        for (size_t i = 0; i < dims[1]; ++i) {
            for (size_t j = 0; j < dims[0]; ++j) {
                src_ptr[j+dims[0]*i] = static_cast<float>(i * j);
            }
        }

        pressio_domain_manager mgr;
        mgr.set_metrics(libpressio::domains_metrics::tracking{});
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

        //owning
        {
        auto malloc_tgt = test_readable(pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("malloc")), src, 
                      "cudamallochost->malloc: should copy",
                      {"readable_begin malloc<cudamallochost", "view_begin malloc<cudamallochost"},
                      {"alloc_begin malloc", "send_begin malloc<cudamallochost"}
                      );
        EXPECT_EQ(malloc_tgt, src);
        auto cuda_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamalloc")), src, 
                      "cudamallochost->cudamalloc: should send",
                      {"readable_begin cudamalloc<cudamallochost", "send_begin cudamalloc<cudamallochost"},
                      {"alloc_begin cudamalloc"}
                      );
        auto mallochost_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamallochost")), src, 
                      "cudamallochost->cudamallochost: should view",
                      {"readable_begin cudamallochost<cudamallochost","view_begin cudamallochost<cudamallochost" },
                      {"alloc_begin cudamalloc", "send_begin cudamalloc<cudamallochost"}
                      );
        EXPECT_EQ(mallochost_tgt, src);

        test_readable(pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("malloc")), cuda_tgt, 
                      "cudamalloc->malloc: should send",
                      {"readable_begin malloc<cudamalloc", "send_begin malloc<cudamalloc"},
                      {"alloc_begin malloc"});

        auto malloc_domtgt = test_readable(libpressio::domain_plugins().build("malloc"), src, 
                      "cudamallochost->malloc: should view",
                      {"readable_domain_begin malloc<cudamallochost", "view_begin malloc<cudamallochost"},
                      {"send_begin malloc<cudamallochost"}
                      );
        test_readable(libpressio::domain_plugins().build("malloc"), cuda_tgt, 
                      "cudamalloc->malloc: should send",
                      {"readable_domain_begin malloc<cudamalloc", "send_begin malloc<cudamalloc", "alloc_begin malloc"},
                      {});
        }

        //nonowning source
        {
            auto nonowning = pressio_data::nonowning(src);
            auto malloc_tgt = test_readable(pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("malloc")), nonowning, 
                    "nonowning cudamallochost->malloc: should copy",
                    {"readable_begin malloc<nonowning", "view_begin malloc<nonowning"},
                    {"alloc_begin malloc", "send_begin malloc<cudamallochost"}
                    );
            ASSERT_EQ(malloc_tgt, nonowning);
            auto cuda_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamalloc")), nonowning, 
                    "nonowning cudamallochost->cudamalloc: should send",
                    {"readable_begin cudamalloc<nonowning", "send_begin cudamalloc<nonowning"},
                    {"alloc_begin cudamalloc"}
                    );
            auto mallochost_tgt = test_readable( pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamallochost")), nonowning, 
                    "nonowning cudamallochost->cudamallochost: should view",
                    {"readable_begin cudamallochost<nonowning","view_begin cudamallochost<nonowning" },
                    {"alloc_begin cudamalloc", "send_begin cudamalloc<cudamallochost"}
                    );
            ASSERT_EQ(mallochost_tgt, nonowning);

            test_readable(pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("malloc")), cuda_tgt, 
                    "nonowning cudamalloc->malloc: should send",
                    {"readable_begin malloc<cudamalloc", "send_begin malloc<cudamalloc"},
                    {"alloc_begin malloc"});

            auto malloc_domtgt = test_readable(libpressio::domain_plugins().build("malloc"), nonowning, 
                    "nonowning cudamallochost->malloc: should view",
                    {"readable_domain_begin malloc<nonowning", "view_begin malloc<nonowning"},
                    {"send_begin malloc<nonowning"}
                    );
            test_readable(libpressio::domain_plugins().build("malloc"), cuda_tgt, 
                    "nonowning cudamalloc->malloc: should send",
                    {"readable_domain_begin malloc<cudamalloc", "send_begin malloc<cudamalloc", "alloc_begin malloc"},
                    {});
        }

        //nonowning target
        {
            auto malloc_owning_target = pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("malloc"));
            auto cudamalloc_owning_target = pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamalloc"));
            auto malloc_tgt = test_readable(pressio_data::nonowning(malloc_owning_target), src, 
                    "nonowningtarget cudamallochost->malloc: should copy",
                    {"readable_begin nonowning<cudamallochost", "view_begin nonowning<cudamallochost"},
                    {"alloc_begin malloc", "send_begin malloc<cudamallochost"}
                    );
            EXPECT_EQ(malloc_tgt, src);
            EXPECT_EQ(malloc_tgt.domain()->domain_id(), "cudamallochost");
            EXPECT_EQ(malloc_tgt.domain()->prefix(), "nonowning");
            auto cuda_tgt = test_readable(pressio_data::nonowning(cudamalloc_owning_target), src, 
                    "nonowningtarget cudamallochost->cudamalloc: should send",
                    {"readable_begin nonowning<cudamallochost", "send_begin nonowning<cudamallochost"},
                    {"alloc_begin cudamalloc"}
                    );
            EXPECT_EQ(cuda_tgt.domain()->domain_id(), "cudamalloc");
            EXPECT_EQ(cuda_tgt.domain()->prefix(), "nonowning");
            auto result = test_readable(pressio_data::nonowning(malloc_owning_target), cuda_tgt, 
                    "nonowningtarget cudamalloc->malloc: should send",
                    {"readable_begin nonowning<nonowning", "send_begin nonowning<nonowning"},
                    {"alloc_begin malloc"});
            EXPECT_EQ(result.domain()->domain_id(), "malloc");
            EXPECT_EQ(result.domain()->prefix(), "nonowning");

            //it doesn't make sense to to have the version here with a domain as the first argument
        }

        //test empty source -> domain; these should preserve the empty property as a nonowning buffer
        {
            auto empty_malloc = test_readable(
                    libpressio::domain_plugins().build("malloc"),
                    /*src*/pressio_data::empty(pressio_float_dtype, {50, 50}),
                    "empty malloc domain, should do metadata only",
                        {"view_begin malloc<malloc", "view_begin malloc<malloc"},
                        {}
                    );
            EXPECT_EQ(empty_malloc.has_data(), false);
            EXPECT_EQ(empty_malloc.dimensions(), (std::vector<size_t>{50, 50}));
            EXPECT_EQ(empty_malloc.dtype(), pressio_float_dtype);
            EXPECT_EQ(empty_malloc.domain()->domain_id(), "malloc");
            EXPECT_EQ(empty_malloc.domain()->prefix(), "nonowning");

            auto empty_cudamalloc = test_readable(
                    libpressio::domain_plugins().build("cudamalloc"),
                    /*src*/pressio_data::empty(pressio_float_dtype, {50, 50}),
                    "empty cudamalloc domain, should do metadata only",
                        {},
                        {}
                    );
            EXPECT_EQ(empty_cudamalloc.has_data(), false);
            EXPECT_EQ(empty_cudamalloc.dimensions(), (std::vector<size_t>{50, 50}));
            EXPECT_EQ(empty_cudamalloc.dtype(), pressio_float_dtype);
            EXPECT_EQ(empty_cudamalloc.domain()->domain_id(), "cudamalloc");
            EXPECT_EQ(empty_cudamalloc.domain()->prefix(), "cudamalloc");
        }

        //test empty source -> data; these should also preserve the property of the empty buffer
        //from the source
        {
            auto empty_malloc = test_readable(
                    pressio_data::owning(pressio_float_dtype, {50, 50}),
                    /*src*/pressio_data::empty(pressio_float_dtype, {50, 50}),
                    "empty malloc domain, should do metadata only",
                        {"view_begin malloc<malloc", "view_begin malloc<malloc"},
                        {}
                    );
            EXPECT_EQ(empty_malloc.has_data(), false);
            EXPECT_EQ(empty_malloc.dimensions(), (std::vector<size_t>{50, 50}));
            EXPECT_EQ(empty_malloc.dtype(), pressio_float_dtype);
            EXPECT_EQ(empty_malloc.domain()->domain_id(), "malloc");
            EXPECT_EQ(empty_malloc.domain()->prefix(), "nonowning");

            auto empty_cudamalloc = test_readable(
                    pressio_data::owning(pressio_double_dtype, {50,50}, libpressio::domain_plugins().build("cudamalloc")),
                    /*src*/pressio_data::empty(pressio_float_dtype, {50, 50}),
                    "empty cudamalloc domain, should do metadata only",
                        {},
                        {}
                    );
            EXPECT_EQ(empty_cudamalloc.has_data(), false);
            EXPECT_EQ(empty_cudamalloc.dimensions(), (std::vector<size_t>{50, 50}));
            EXPECT_EQ(empty_cudamalloc.dtype(), pressio_float_dtype);
            EXPECT_EQ(empty_cudamalloc.domain()->domain_id(), "cudamalloc");
            EXPECT_EQ(empty_cudamalloc.domain()->prefix(), "cudamalloc");
        }

    }
    ASSERT_EQ(1,1); // have some instruction after scope ends
}

TEST(Domains, TestWritableDomains) {
    if(!libpressio::domain_plugins().build("cudamalloc")) {
        GTEST_SKIP() << "this test requires cuda";
    }

    //force everything to go out of scope
    ASSERT_EQ(1,1); // have some instruction before scope ends
    {
        std::vector<size_t> dims{50,50};
        auto src = pressio_data::owning(pressio_float_dtype, dims, libpressio::domain_plugins().build("cudamallochost"));
        float* src_ptr = static_cast<float*>(src.data());
        for (size_t i = 0; i < dims[1]; ++i) {
            for (size_t j = 0; j < dims[0]; ++j) {
                src_ptr[j+dims[0]*i] = static_cast<float>(i * j);
            }
        }

        pressio_domain_manager mgr;
        mgr.set_metrics(libpressio::domains_metrics::tracking{});
        auto test_writeable = [&](
                std::shared_ptr<pressio_domain>&& dst,
                auto&& src,
                const char* desc,
                std::string expected_domain_id,
                std::vector<std::string> const& expected,
                std::vector<std::string> const& exclude = {}
        ) {
            auto target = mgr.make_writeable(std::forward<decltype(dst)>(dst), std::forward<decltype(src)>(src));
            auto results = mgr.get_metrics_results();
            auto events = std::get<std::vector<std::string>>(results.at("tracking:events"));
            mgr.set_options({{"tracking:reset", true}});
            for (auto const& e : expected) {
                EXPECT_THAT(events, ::testing::Contains(e)) << desc;
            }
            for (auto const& e : exclude) {
                EXPECT_THAT(events, ::testing::Not(::testing::Contains(e))) << desc;
            }
            EXPECT_EQ(target.domain()->domain_id(), expected_domain_id) << desc;
            return target;
        };

        //passes an pressio_data const&, where are domains equal
        test_writeable(libpressio::domain_plugins().build("cudamallochost"), src, 
                      "cudamallochost const&->cudamallocmalloc&&: should allocate",
                      "cudamallochost",
                      {"alloc_begin cudamallochost", "alloc_end cudamallochost"},
                      {}
                      );
        //passes an pressio_data&&, where domains are equal
        test_writeable(libpressio::domain_plugins().build("cudamallochost"), pressio_data::clone(src), 
                      "cudamallochost &&->cudamallochost&&: should view",
                      "cudamallochost",
                      {"view_begin cudamallochost<cudamallochost", "view_end cudamallochost<{moved}"},
                      {});
        //passes an pressio_data const&, where domains are accessible
        test_writeable(libpressio::domain_plugins().build("malloc"), src, 
                      "cudamallochost const&->malloc&&: should allocate",
                      "malloc",
                      {"alloc_begin malloc", "alloc_end malloc"},
                      {}
                      );
        //passes an pressio_data&&, where domains are accessible
        test_writeable(libpressio::domain_plugins().build("malloc"), pressio_data::clone(src), 
                      "cudamallochost &&->malloc&&: should view",
                      "cudamallochost",
                      {"view_begin malloc<cudamallochost", "view_end malloc<{moved}"},
                      {});
        //passes an pressio_data const&, where domains are not accessible
        test_writeable(libpressio::domain_plugins().build("cudamalloc"), src, 
                      "cudamallochost const&->cudamalloc&&: should allocate",
                      "cudamalloc",
                      {"alloc_begin cudamalloc", "alloc_end cudamalloc"},
                      {}
                      );
        //passes an pressio_data&&, where domains are not accessible
        test_writeable(libpressio::domain_plugins().build("cudamalloc"), pressio_data::clone(src), 
                      "cudamallochost &&->cudamalloc&&: should allocate",
                      "cudamalloc",
                      {"alloc_begin cudamalloc", "alloc_end cudamalloc"},
                      {});

        auto nonowning = pressio_data::nonowning(src);

        //test can clone a non-owning data involving domains
        auto cloned = pressio_data::clone(nonowning);
        EXPECT_EQ(cloned.dimensions(), src.dimensions());
        EXPECT_EQ(cloned.dtype(), src.dtype());
        EXPECT_EQ(cloned.domain()->prefix(), "cudamallochost") << "a clone of a non-owning domain should inherit its source";
        EXPECT_EQ(cloned.domain()->domain_id(), "cudamallochost") << "a clone of a non-owning domain should inherit its source";

        //passes an pressio_data const&, where are domains equal
        test_writeable(libpressio::domain_plugins().build("cudamallochost"), nonowning, 
                      "nonowning cudamallochost const&->cudamallocmalloc&&: should allocate",
                      "cudamallochost",
                      {"alloc_begin cudamallochost", "alloc_end cudamallochost"},
                      {}
                      );
        //passes an pressio_data&&, where domains are equal
        test_writeable(libpressio::domain_plugins().build("cudamallochost"), pressio_data::nonowning(nonowning), 
                      "nonowning cudamallochost &&->cudamallochost&&: should view",
                      "cudamallochost",
                      {"view_begin cudamallochost<nonowning", "view_end cudamallochost<{moved}"},
                      {});
        //passes an pressio_data const&, where domains are accessible
        test_writeable(libpressio::domain_plugins().build("malloc"), nonowning, 
                      "nonowning cudamallochost const&->malloc&&: should allocate",
                      "malloc",
                      {"alloc_begin malloc", "alloc_end malloc"},
                      {}
                      );
        //passes an pressio_data&&, where domains are accessible
        test_writeable(libpressio::domain_plugins().build("malloc"), pressio_data::nonowning(nonowning), 
                      "nonowning cudamallochost &&->malloc&&: should view",
                      "cudamallochost",
                      {"view_begin malloc<nonowning", "view_end malloc<{moved}"},
                      {});
        //passes an pressio_data const&, where domains are not accessible
        test_writeable(libpressio::domain_plugins().build("cudamalloc"), nonowning, 
                      "nonowning cudamallochost const&->cudamalloc&&: should allocate",
                      "cudamalloc",
                      {"alloc_begin cudamalloc", "alloc_end cudamalloc"},
                      {}
                      );
        //passes an pressio_data&&, where domains are not accessible
        test_writeable(libpressio::domain_plugins().build("cudamalloc"), pressio_data::nonowning(nonowning), 
                      "nonowning cudamallochost &&->cudamalloc&&: should allocate",
                      "cudamalloc",
                      {"alloc_begin cudamalloc", "alloc_end cudamalloc"},
                      {});
    }

    //it doesn't make sense to test non-owning targets

    ASSERT_EQ(1,1); // have some instruction after scope ends to make it easier to debug
}


TEST(Domains, TestUserDomainsAndMoves) {

    //first test the special case of malloc in the classic way
    {
        constexpr size_t N = 50;
        double* data = static_cast<double*>(malloc(sizeof(double)*N*N));
        pressio_data user_malloc(pressio_data::move(pressio_double_dtype, (void*)data, std::vector<size_t>{N,N}, pressio_data_libc_free_fn, nullptr));
        ASSERT_EQ(user_malloc.domain()->domain_id(), "malloc");
        ASSERT_EQ(user_malloc.domain()->prefix(), "malloc");
        ASSERT_EQ(user_malloc.data(), data);
        ASSERT_NE(nullptr, user_malloc.data());

        pressio_data nonowning_user_malloc(pressio_data::nonowning(user_malloc));
        ASSERT_EQ(user_malloc.data(), nonowning_user_malloc.data());
        ASSERT_NE(nullptr, nonowning_user_malloc.data());

        pressio_data owning_user_malloc(pressio_data::owning(user_malloc));
        ASSERT_NE(user_malloc.data(), owning_user_malloc.data());
        ASSERT_NE(nullptr, owning_user_malloc.data());

        auto cloned = pressio_data::clone(user_malloc);
        ASSERT_NE(user_malloc.data(), owning_user_malloc.data());
        ASSERT_NE(nullptr, cloned.data());
    }

    //now test the "new" way for malloc
    {
        constexpr size_t N = 50;
        double* data = static_cast<double*>(malloc(sizeof(double)*N*N));
        pressio_data user_malloc(pressio_data::move(pressio_double_dtype, data, {N,N}, libpressio::domain_plugins().build("malloc")));
        ASSERT_EQ(user_malloc.domain()->domain_id(), "malloc");
        ASSERT_EQ(user_malloc.domain()->prefix(), "malloc");
        ASSERT_EQ(user_malloc.data(), data);
        ASSERT_NE(nullptr, user_malloc.data());

        pressio_data nonowning_user_malloc(pressio_data::nonowning(user_malloc));
        ASSERT_EQ(user_malloc.data(), nonowning_user_malloc.data());
        ASSERT_NE(nullptr, nonowning_user_malloc.data());
        pressio_data owning_user_malloc(pressio_data::owning(user_malloc));
        ASSERT_NE(user_malloc.data(), owning_user_malloc.data());
        ASSERT_NE(nullptr, owning_user_malloc.data());
        auto cloned = pressio_data::clone(user_malloc);
        ASSERT_NE(user_malloc.data(), owning_user_malloc.data());
        ASSERT_NE(nullptr, cloned.data());
    }
}

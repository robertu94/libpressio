#include <algorithm>
#include <cmath>
#include <iterator>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "std_compat/algorithm.h"
#include "std_compat/functional.h"

/**
 * This module largely adapted from NUMPY. License appears below
 *  Copyright (c) 2005-2020, NumPy Developers.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *  met:
 *
 *      * Redistributions of source code must retain the above copyright
 *         notice, this list of conditions and the following disclaimer.
 *
 *      * Redistributions in binary form must reproduce the above
 *         copyright notice, this list of conditions and the following
 *         disclaimer in the documentation and/or other materials provided
 *         with the distribution.
 *
 *      * Neither the name of the NumPy Developers nor the names of any
 *         contributors may be used to endorse or promote products derived
 *         from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace libpressio { namespace ks_test {
template <class ForwardItItems, class ForwardItValues, class OutputIt>
void cdf(ForwardItItems items_begin, ForwardItItems items_end,
                  ForwardItValues values_begin, ForwardItValues values_end,
                  OutputIt out) {
  assert(std::is_sorted(items_begin, items_end));
  using const_reference = const typename std::iterator_traits<ForwardItValues>::reference;
  const double n = std::distance(items_begin, items_end);
  std::transform(
      values_begin, values_end, out,
      [n, items_begin, items_end](const_reference v) {
        return std::distance(items_begin, std::upper_bound(items_begin, items_end, v)) / n;
      });
}


template <class RandomIt1, class RandomIt2>
double ks_test_d(
      RandomIt1 data1_begin_p, RandomIt1 data1_end_p,
      RandomIt2 data2_begin_p, RandomIt2 data2_end_p) {

    std::vector<typename std::iterator_traits<RandomIt1>::value_type> data1(data1_begin_p, data1_end_p);
    std::vector<typename std::iterator_traits<RandomIt2>::value_type> data2(data2_begin_p, data2_end_p);

    auto data1_begin = data1.begin();
    auto data1_end = data1.end();
    auto data2_begin = data2.begin();
    auto data2_end = data2.end();

    //ensure precondition of sorted inputs
    std::sort(data1_begin, data1_end);
    std::sort(data2_begin, data2_end);
    const auto n1 = std::distance(data1_begin, data1_end);
    const auto n2 = std::distance(data2_begin, data2_end);
    
    //create all_data
    std::vector<double> all_data;
    all_data.reserve(n1 + n2);
    all_data.insert(all_data.end(), data1_begin, data1_end);
    all_data.insert(all_data.end(), data2_begin, data2_end);

    //compute cdfs 
    std::vector<double> cdf1, cdf2;
    cdf(data1_begin, data1_end, all_data.begin(), all_data.end(), std::back_inserter(cdf1));
    cdf(data2_begin, data2_end, all_data.begin(), all_data.end(), std::back_inserter(cdf2));
    
    std::vector<double> cdf_diff;
    cdf_diff.reserve(cdf1.size());
    std::transform(
        std::begin(cdf1), std::end(cdf1),
        std::begin(cdf2),
        std::begin(cdf_diff),
        compat::minus<>{}
        );
    auto min_max = std::minmax_element(std::begin(cdf_diff), std::end(cdf_diff));
    double maxS = *min_max.first;
    double minS = compat::clamp<double>(-*min_max.first, 0, 1);
    return std::max(minS, maxS);
}

struct kolmogorov_result {
  double sf, cdf, pdf;
};

kolmogorov_result
kolmogorov(double x)
{
    double P = 1.0;
    double D = 0;
    double sf, cdf, pdf;
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
    constexpr auto KOLMOG_CUTOVER = 0.82;
    constexpr auto MIN_EXPABLE = (-708-38); //smallest value at which std::exp(x) returns 0

    if (std::isnan(x)) {
        return {nan, nan, nan};
    }
    if (x <= 0) {
        return {1.0, 0.0, 0};
    }
    /* x <= 0.040611972203751713 */
    if (x <= M_PI/std::sqrt(-MIN_EXPABLE * 8)) {
        return {1.0, 0.0, 0};
    }

    P = 1.0;
    if (x <= KOLMOG_CUTOVER) {
        /*
         *  u = e^(-pi^2/(8x^2))
         *  w = sqrt(2pi)/x
         *  P = w*u * (1 + u^8 + u^24 + u^48 + ...)
         */
        const double w = std::sqrt(2 * M_PI)/x;
        const double logu8 = -M_PI * M_PI/(x * x); /* log(u^8) */
        const double u = std::exp(logu8/8);
        if (u == 0) {
            /*
             * P = w*u, but u < 1e-308, and w > 1,
             * so compute as logs, then exponentiate
             */
            double logP = logu8/8 + std::log(w);
            P = std::exp(logP);
        } else {
           /* Just unroll the loop, 3 iterations */
            double u8 = std::exp(logu8);
            double u8cub = std::pow(u8, 3);
            P = 1 + u8cub * P;
            D = 5*5 + u8cub * D;
            P = 1 + u8*u8 * P;
            D = 3*3 + u8*u8 * D;
            P = 1 + u8 * P;
            D = 1*1 + u8 * D;

            D = M_PI * M_PI/4/(x*x) * D - P;
            D *=  w * u/x;
            P = w * u * P;
        }
        cdf = P;
        sf = 1-P;
        pdf = D;
    }
    else {
        /*
         *  v = e^(-2x^2)
         *  P = 2 (v - v^4 + v^9 - v^16 + ...)
         *    = 2v(1 - v^3*(1 - v^5*(1 - v^7*(1 - ...)))
         */
        double logv = -2*x*x;
        double v = std::exp(logv);
        /*
         * Want q^((2k-1)^2)(1-q^(4k-1)) / q(1-q^3) < epsilon to break out of loop.
         * With KOLMOG_CUTOVER ~ 0.82, k <= 4.  Just unroll the loop, 4 iterations
         */
        double vsq = v*v;
        double v3 = std::pow(v, 3);
        double vpwr;

        vpwr = v3*v3*v;   /* v**7 */
        P = 1.0 - vpwr * P; /* P <- 1 - (1-v**(2k-1)) * P */
        D = 3*3 - vpwr * D;

        vpwr = v3*vsq;
        P = 1 - vpwr * P;
        D = 2*2 - vpwr * D;

        vpwr = v3;
        P = 1 - vpwr * P;
        D = 1*1 - vpwr * D;

        P = 2 * v * P;
        D = 8 * v * x * D;
        sf = P;
        cdf = 1 - sf;
        pdf = D;
    }
    pdf = std::max(0.0, pdf);
    cdf = compat::clamp(cdf, 0.0, 1.0);
    sf = compat::clamp(sf, 0.0, 1.0);
    return {sf, cdf, pdf};
}




  struct KSTestResult {
    double D;
    double prob;
  };

  struct ks_test {
    template <class RandomIt1, class RandomIt2>
    KSTestResult operator()(RandomIt1 const* input_begin, RandomIt1 const* input_end,
                         RandomIt2 const* output_begin, RandomIt2 const* output_end)
    {
    const auto n1 = std::distance(input_begin, input_end);
    const auto n2 = std::distance(output_begin, output_end);
    const auto en = std::sqrt((n1*n2)/static_cast<double>(n1+n2));

    KSTestResult result;
    result.D = ks_test_d(input_begin, input_end, output_begin, output_end);
    result.prob = kolmogorov((en + 0.12 + 0.11 / en ) * result.D).sf;
    return result;

    }
  };


class ks_test_plugin : public libpressio_metrics_plugin {

public:
  int begin_compress_impl(const struct pressio_data* input,
                      struct pressio_data const*) override
  {
    input_data = pressio_data::clone(*input);
    return 0;
  }
  int end_decompress_impl(struct pressio_data const*,
                      struct pressio_data const* output, int) override
  {
      auto result = pressio_data_for_each<KSTestResult>(input_data, *output, ks_test{});
      pvalue = result.prob;
      d = result.D;
      return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "Kolmogorov–Smirnov test for difference in distributions");
    set(opt, "ks_test:pvalue", "the p-value of the test statistic");
    set(opt, "ks_test:d", "the test statistic");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override
  {
    pressio_options opt;
    set(opt, "ks_test:pvalue", pvalue);
    set(opt, "ks_test:d", d);
    return opt;
  }

  int set_options(struct pressio_options const&) override
  {
    return 0;
  }

  pressio_options get_options() const override
  {
    return pressio_options{};
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<ks_test_plugin>(*this);
  }

  const char* prefix() const override {
    return "ks_test";
  }

private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<double> pvalue;
  compat::optional<double> d;
};

static pressio_register metrics_ks_test_plugin(metrics_plugins(), "ks_test",
                          []() { return compat::make_unique<ks_test_plugin>(); });
} }

#include <vector>
#include <numeric>
#include <memory>
#include <array>
#include <cstdlib>
#include <string>
#include "pressio_data.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/domain.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "libpressio_ext/cpp/domain_send.h"
#include "libpressio_ext/cpp/printers.h"
#include "multi_dimensional_iterator.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class PressioDataTests: public ::testing::Test {
  protected:
    void SetUp() {
      data = std::vector<int>(6);
      std::iota(std::begin(data), std::end(data), 0);
    }

    void TearDown() {
    }

    /**
     * libpressio is column major
     * size:    2, 3 
     * strides: 1, 2
     *
     * so the data layout is this:
     * [[ 0, 2, 4],
     *  [ 1, 3, 5]]
     */
    const size_t dims[2]  = {2,3};
    std::vector<int> data;
};

namespace {
class tracking_domain : public libpressio::domains::pressio_domain {
public:
  void* alloc(size_t n) override {
    ++outstanding_allocations;
    return std::malloc(n);
  }

  void free(void* ptr, size_t) override {
    if (ptr) {
      --outstanding_allocations;
      std::free(ptr);
    }
  }

  void memcpy(void* dst, void* src, size_t n) override {
    std::memcpy(dst, src, n);
  }

  std::string const& prefix() const override {
    static const std::string value = "test_tracking_domain";
    return value;
  }

  std::shared_ptr<libpressio::domains::pressio_domain> clone() override {
    return std::make_shared<tracking_domain>();
  }

  static int outstanding_allocations;

protected:
  bool equal(libpressio::domains::pressio_domain const& rhs) const noexcept override {
    return dynamic_cast<tracking_domain const*>(&rhs) != nullptr;
  }
};

int tracking_domain::outstanding_allocations = 0;

libpressio::pressio_register tracking_domain_registration(
    libpressio::domain_plugins(), "test_tracking_domain", [] {
      return std::make_shared<tracking_domain>();
    });

std::shared_ptr<libpressio::domains::pressio_domain> tracking_domain_instance() {
  return libpressio::domain_plugins().build("test_tracking_domain");
}

class alternate_tracking_domain final : public tracking_domain {
public:
  std::string const& prefix() const override {
    static const std::string value = "test_alternate_tracking_domain";
    return value;
  }

  std::shared_ptr<libpressio::domains::pressio_domain> clone() override {
    return std::make_shared<alternate_tracking_domain>();
  }

protected:
  bool equal(libpressio::domains::pressio_domain const& rhs) const noexcept override {
    return dynamic_cast<alternate_tracking_domain const*>(&rhs) != nullptr;
  }
};

libpressio::pressio_register alternate_tracking_domain_registration(
    libpressio::domain_plugins(), "test_alternate_tracking_domain", [] {
      return std::make_shared<alternate_tracking_domain>();
    });

std::shared_ptr<libpressio::domains::pressio_domain> alternate_tracking_domain_instance() {
  return libpressio::domain_plugins().build("test_alternate_tracking_domain");
}

class tracking_domain_send final : public libpressio::domains::pressio_domain_send {
public:
  void send(pressio_data& destination, pressio_data const& source) const override {
    destination.domain()->memcpy(destination.data(), source.data(), source.size_in_bytes());
  }
};

libpressio::pressio_register tracking_to_alternate_send_registration(
    libpressio::domain_send_plugins(),
    "test_tracking_domain>test_alternate_tracking_domain", [] {
      return std::make_unique<tracking_domain_send>();
    });
libpressio::pressio_register alternate_to_tracking_send_registration(
    libpressio::domain_send_plugins(),
    "test_alternate_tracking_domain>test_tracking_domain", [] {
      return std::make_unique<tracking_domain_send>();
    });
} // namespace

TEST(PressioDataDomainLifetime, MoveAssigningAliasedViewPreservesOwner) {
  ASSERT_EQ(tracking_domain::outstanding_allocations, 0);
  {
    auto data = pressio_data::owning(
        pressio_int32_dtype, {16}, tracking_domain_instance());
    auto* original = static_cast<int32_t*>(data.data());

    data = domain_manager().make_readable(data.domain(), std::move(data));

    ASSERT_EQ(data.data(), original);
    ASSERT_EQ(tracking_domain::outstanding_allocations, 1);
    static_cast<int32_t*>(data.data())[0] = 42;
    EXPECT_EQ(static_cast<int32_t*>(data.data())[0], 42);
  }
  EXPECT_EQ(tracking_domain::outstanding_allocations, 0);
}

TEST(PressioDataDomainLifetime, RepeatedCopiesMovesViewsAndConversionsReleaseEveryAllocation) {
  ASSERT_EQ(tracking_domain::outstanding_allocations, 0);

  constexpr size_t iterations = 1024;
  {
    // Keep one live owning object between iterations.  Each iteration replaces
    // it after creating a complete graph of copy, move, non-owning, readable,
    // and writable representations.  A missing release in any replacement
    // accumulates and is detected both immediately and at scope exit.
    auto current = pressio_data::owning(
        pressio_int32_dtype, {16}, tracking_domain_instance());
    ASSERT_EQ(tracking_domain::outstanding_allocations, 1);

    for (size_t i = 0; i < iterations; ++i) {
      // Copy allocates on a cloned tracking domain.  Moving that copy must not
      // allocate or free its storage.
      auto copied = current;
      ASSERT_EQ(tracking_domain::outstanding_allocations, 2) << "iteration " << i;
      auto moved = std::move(copied);
      ASSERT_EQ(tracking_domain::outstanding_allocations, 2) << "iteration " << i;

      // Create views from both an owning object and an existing view.  Neither
      // may affect the allocation count or take ownership of moved.data().
      auto view = pressio_data::nonowning(moved);
      auto view_of_view = pressio_data::nonowning(view);
      ASSERT_EQ(tracking_domain::outstanding_allocations, 2) << "iteration " << i;

      // Every representation created above is independently made readable and
      // writable.  make_readable starts with allocated reusable storage, then
      // replaces it with a view; make_writeable(const&) must instead allocate.
      auto verify_const_conversions = [&](pressio_data const& source) {
        const int allocations_before = tracking_domain::outstanding_allocations;
        {
          auto reusable_readable = pressio_data::owning(
              pressio_int32_dtype, {16}, tracking_domain_instance());
          ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before + 1)
              << "iteration " << i;
          auto readable = domain_manager().make_readable(
              std::move(reusable_readable), source);
          ASSERT_EQ(readable.data(), source.data()) << "iteration " << i;
          ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before)
              << "iteration " << i;
          auto writable = domain_manager().make_writeable(
              tracking_domain_instance(), readable);
          ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before + 1)
              << "iteration " << i;
        }
        ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before)
            << "iteration " << i;

        // Exercise both domain-destination make_readable overloads.  Since
        // these domains are accessible, each result is a nonowning view.
        auto destination_domain = tracking_domain_instance();
        auto readable_on_lvalue_domain = domain_manager().make_readable(
            destination_domain, source);
        auto readable_on_rvalue_domain = domain_manager().make_readable(
            tracking_domain_instance(), source);
        ASSERT_EQ(readable_on_lvalue_domain.data(), source.data()) << "iteration " << i;
        ASSERT_EQ(readable_on_rvalue_domain.data(), source.data()) << "iteration " << i;
        ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before)
            << "iteration " << i;

        // make_writeable accepts an rvalue destination domain.  Exercise its
        // const-source allocation path and movable-view reuse path.
        {
          auto writable_on_domain = domain_manager().make_writeable(
              tracking_domain_instance(), readable_on_rvalue_domain);
          ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before + 1)
              << "iteration " << i;
        }
        ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before)
            << "iteration " << i;
        auto movable_view = pressio_data::nonowning(readable_on_lvalue_domain);
        auto reused_writable = domain_manager().make_writeable(
            tracking_domain_instance(), std::move(movable_view));
        ASSERT_EQ(reused_writable.data(), source.data()) << "iteration " << i;
        ASSERT_EQ(tracking_domain::outstanding_allocations, allocations_before)
            << "iteration " << i;
      };
      verify_const_conversions(current);
      verify_const_conversions(moved);
      verify_const_conversions(view);
      verify_const_conversions(view_of_view);

      // Make an allocated destination readable from a view once more, then
      // retain its writable copy across the iteration boundary.
      auto reusable_readable = pressio_data::owning(
          pressio_int32_dtype, {16}, tracking_domain_instance());
      auto readable = domain_manager().make_readable(
          std::move(reusable_readable), view_of_view);
      ASSERT_EQ(readable.data(), moved.data()) << "iteration " << i;
      auto writable_from_const = domain_manager().make_writeable(
          tracking_domain_instance(), readable);
      ASSERT_EQ(tracking_domain::outstanding_allocations, 3) << "iteration " << i;

      // The owning moved representation takes the movable/reuse writable
      // branch.  Its allocation must remain owned exactly once.
      auto writable_from_move = domain_manager().make_writeable(
          tracking_domain_instance(), std::move(moved));
      ASSERT_EQ(tracking_domain::outstanding_allocations, 3) << "iteration " << i;

      // An inaccessible nonowning input cannot be reused and therefore takes
      // the allocation branch of make_writeable.
      int external[16] = {};
      auto foreign_view = pressio_data::nonowning(
          pressio_int32_dtype, external, {16}, "malloc");
      auto writable_from_foreign_view = domain_manager().make_writeable(
          tracking_domain_instance(), std::move(foreign_view));
      ASSERT_EQ(tracking_domain::outstanding_allocations, 4) << "iteration " << i;

      // Replacing the previous persistent allocation is another move-assignment
      // path.  The temporary objects leave one allocation for current.
      current = std::move(writable_from_const);
    }

    ASSERT_EQ(tracking_domain::outstanding_allocations, 1);
  }
  EXPECT_EQ(tracking_domain::outstanding_allocations, 0);
}

TEST(PressioDataDomainLifetime, ExhaustiveDomainManagerBranchMatrixDoesNotLeak) {
  ASSERT_EQ(tracking_domain::outstanding_allocations, 0);

  // The two test domains use host malloc/free but compare unequal.  Together
  // with the registered send plugins this exercises every generic
  // pressio_domain_manager branch: accessible view/reuse, inaccessible send,
  // destination allocation, existing-destination reuse, and empty metadata.
  for (size_t iteration = 0; iteration < 64; ++iteration) {
    for (bool source_is_alternate : {false, true}) {
      for (bool destination_is_alternate : {false, true}) {
        auto source_domain = source_is_alternate
            ? alternate_tracking_domain_instance() : tracking_domain_instance();
        auto destination_domain = destination_is_alternate
            ? alternate_tracking_domain_instance() : tracking_domain_instance();
        const bool accessible = source_is_alternate == destination_is_alternate;

        for (bool source_is_view : {false, true}) {
          for (bool source_is_empty : {false, true}) {
            auto source_owner = source_is_empty
                ? pressio_data::empty(pressio_int32_dtype, {16}, source_domain)
                : pressio_data::owning(pressio_int32_dtype, {16}, source_domain);
            auto source = source_is_view
                ? pressio_data::nonowning(source_owner) : std::move(source_owner);
            const int source_allocations = source_is_empty ? 0 : 1;
            ASSERT_EQ(tracking_domain::outstanding_allocations, source_allocations)
                << "iteration " << iteration;

            // make_readable(data&&, const data&) with an allocated and empty
            // destination covers all reusable-buffer branches.
            for (bool destination_is_empty : {false, true}) {
              {
                auto destination = destination_is_empty
                    ? pressio_data::empty(pressio_int32_dtype, {16}, destination_domain)
                    : pressio_data::owning(pressio_int32_dtype, {16}, destination_domain);
                const int before = tracking_domain::outstanding_allocations;
                auto readable = domain_manager().make_readable(std::move(destination), source);
                EXPECT_EQ(readable.has_data(), source.has_data());
                const int allocation_delta =
                    (accessible || !source.has_data())
                        ? (destination_is_empty ? 0 : -1)
                        : (destination_is_empty ? 1 : 0);
                EXPECT_EQ(tracking_domain::outstanding_allocations,
                    before + allocation_delta)
                    << "iteration " << iteration;
              }
              ASSERT_EQ(tracking_domain::outstanding_allocations, source_allocations)
                  << "iteration " << iteration;
            }

            // Both domain-destination make_readable overloads must view when
            // accessible and allocate/send when inaccessible.
            {
              auto destination = destination_domain;
              auto readable = domain_manager().make_readable(destination, source);
              EXPECT_EQ(readable.has_data(), source.has_data());
              EXPECT_EQ(tracking_domain::outstanding_allocations,
                  source_allocations + ((!accessible && source.has_data()) ? 1 : 0));
            }
            ASSERT_EQ(tracking_domain::outstanding_allocations, source_allocations);
            {
              auto readable = domain_manager().make_readable(destination_domain, source);
              EXPECT_EQ(readable.has_data(), source.has_data());
              EXPECT_EQ(tracking_domain::outstanding_allocations,
                  source_allocations + ((!accessible && source.has_data()) ? 1 : 0));
            }
            ASSERT_EQ(tracking_domain::outstanding_allocations, source_allocations);
          }
        }

        // make_writeable only accepts an rvalue destination domain.  Cover
        // const and movable owning/view/empty sources for both accessibility
        // outcomes.  Recreate each rvalue source so every value category is
        // independently tested.
        for (bool source_is_view : {false, true}) {
          for (bool source_is_empty : {false, true}) {
            {
              auto owner = source_is_empty
                  ? pressio_data::empty(pressio_int32_dtype, {16}, source_domain)
                  : pressio_data::owning(pressio_int32_dtype, {16}, source_domain);
              auto source = source_is_view
                  ? pressio_data::nonowning(owner) : std::move(owner);
              const int before = tracking_domain::outstanding_allocations;
              auto writable = domain_manager().make_writeable(destination_domain->clone(), source);
              EXPECT_TRUE(writable.has_data());
              EXPECT_EQ(tracking_domain::outstanding_allocations, before + 1);
            }
            ASSERT_EQ(tracking_domain::outstanding_allocations, 0);
            {
              auto owner = source_is_empty
                  ? pressio_data::empty(pressio_int32_dtype, {16}, source_domain)
                  : pressio_data::owning(pressio_int32_dtype, {16}, source_domain);
              auto source = source_is_view
                  ? pressio_data::nonowning(owner) : std::move(owner);
              const int before = tracking_domain::outstanding_allocations;
              auto writable = domain_manager().make_writeable(destination_domain->clone(), std::move(source));
              EXPECT_TRUE(writable.has_data());
              EXPECT_EQ(tracking_domain::outstanding_allocations,
                  before + ((accessible && !source_is_empty) ? 0 : 1));
            }
            ASSERT_EQ(tracking_domain::outstanding_allocations, 0);
          }
        }
      }
    }
  }
  EXPECT_EQ(tracking_domain::outstanding_allocations, 0);
}

TEST_F(PressioDataTests, NormalizeDims) {
  {
  auto orig = pressio_data::nonowning(pressio_int32_dtype, data.data(), {dims[0], dims[1]});
  EXPECT_EQ(orig.normalized_dims(1), std::vector<size_t>{6});
  EXPECT_EQ(orig.normalized_dims(2), (std::vector<size_t>{2,3}));
  EXPECT_EQ(orig.normalized_dims(3), (std::vector<size_t>{2,3,0}));
  EXPECT_EQ(orig.normalized_dims(3, 1), (std::vector<size_t>{2,3,1}));
  EXPECT_EQ(orig.normalized_dims(3, 0), (std::vector<size_t>{2,3,0}));
  }

  {
  auto orig2 = pressio_data::empty(pressio_int32_dtype, {2, 3, 4});
  EXPECT_EQ(orig2.normalized_dims(1), std::vector<size_t>{24});
  EXPECT_EQ(orig2.normalized_dims(2), (std::vector<size_t>{6,4}));
  EXPECT_EQ(orig2.normalized_dims(3), (std::vector<size_t>{2,3,4}));
  EXPECT_EQ(orig2.normalized_dims(3, 1), (std::vector<size_t>{2,3,4}));
  EXPECT_EQ(orig2.normalized_dims(3, 0), (std::vector<size_t>{2,3,4}));
  EXPECT_EQ(orig2.normalized_dims(4), (std::vector<size_t>{2,3,4,0}));
  EXPECT_EQ(orig2.normalized_dims(4, 0), (std::vector<size_t>{2,3,4,0}));
  EXPECT_EQ(orig2.normalized_dims(4, 1), (std::vector<size_t>{2,3,4,1}));
  }

}
TEST_F(PressioDataTests, Transpose) {
  auto original = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  auto transposed = pressio_data_transpose(original, nullptr);

  EXPECT_EQ(pressio_data_num_dimensions(transposed), 2);
  EXPECT_EQ(pressio_data_get_dimension(transposed, 0), 3);
  EXPECT_EQ(pressio_data_get_dimension(transposed, 1), 2);

  int* begin = (int*)pressio_data_ptr(transposed, nullptr);
  std::vector<int> actual(begin, begin+6);

  /**
   * the transposed layout is this
   *
   * [[ 0 1
   *  [ 2 3
   *  [ 4 5]]
   * which linearizes to [0 2 4 1 3 5]
   *
   */
  std::vector<int> expected{0, 2, 4, 1, 3, 5};
  EXPECT_EQ(expected, actual);
  pressio_data_free(original);
  pressio_data_free(transposed);
}

TEST_F(PressioDataTests, MakePressioData) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  EXPECT_EQ(pressio_data_dtype(d), pressio_int32_dtype);
  EXPECT_EQ(pressio_data_num_dimensions(d), 2);
  EXPECT_EQ(pressio_data_get_dimension(d, 0), 2);
  EXPECT_EQ(pressio_data_get_dimension(d, 1), 3);
  EXPECT_EQ(pressio_data_ptr(d, nullptr), data.data());
  size_t size = 0;
  EXPECT_EQ(pressio_data_ptr(d, &size), data.data());
  EXPECT_EQ(size, 6*sizeof(int));

  pressio_data_free(d);
}

TEST_F(PressioDataTests, Printers) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  std::stringstream ss;
  ss << *d;
  EXPECT_THAT(ss.str(), testing::StrEq("data{ type=int32_t dims={2, 3, } domain=nonowning has_data=[0, 1, 2, 3, 4, 5, ]}"));
  pressio_data_free(d);
}

TEST_F(PressioDataTests, InitalizerLists) {
  pressio_data d_int{1,2,3};
  pressio_data d_double{1.0,2.0,3.0};

  EXPECT_EQ(d_int.dtype(), pressio_int32_dtype);
  EXPECT_EQ(d_int.num_elements(), 3);
  EXPECT_EQ(d_int.has_data(), true);
  std::vector<int> expected_int{1,2,3};
  std::vector<int> actual_int(
      static_cast<int*>(d_int.data()),
      static_cast<int*>(d_int.data())+ d_int.num_elements()
    );
  EXPECT_THAT(actual_int, testing::ElementsAreArray(expected_int));

  EXPECT_EQ(d_double.dtype(), pressio_double_dtype);
  EXPECT_EQ(d_double.num_elements(), 3);
  EXPECT_EQ(d_double.has_data(), true);
  std::vector<double> expected_double{1.0,2.0,3.0};
  std::vector<double> actual_double(
      static_cast<double*>(d_double.data()),
      static_cast<double*>(d_double.data())+ d_double.num_elements()
    );
  EXPECT_THAT(actual_double, testing::ElementsAreArray(expected_double));

}

namespace {
struct accumulator {
  template <class T>
    double operator()(T begin, T end) {
        return static_cast<double>(std::accumulate(begin, end, 0));
    }
};
}

TEST_F(PressioDataTests, ForEach) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  auto result = pressio_data_for_each<double>(*d, accumulator{});

  EXPECT_EQ(result, static_cast<double>(std::accumulate(std::begin(data), std::end(data), 0)));
  pressio_data_free(d);
}


TEST_F(PressioDataTests, Cast) {
  pressio_data* orig = pressio_data_new_copy(pressio_int32_dtype, data.data(), 2, dims);
  pressio_data* casted = pressio_data_cast(orig, pressio_double_dtype);
  std::vector<double> expected(data.size());
  std::copy(std::begin(data), std::end(data), std::begin(expected));
  auto actual_p  = static_cast<double*>(pressio_data_ptr(casted, nullptr));
  std::vector<double> actual(actual_p, actual_p+data.size());

  EXPECT_THAT(actual, testing::ElementsAreArray(expected));

  pressio_data_free(orig);
  pressio_data_free(casted);
}

TEST_F(PressioDataTests, MakeCopy) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  EXPECT_NE(d, nullptr);
  size_t size = 0;
  void* copy = pressio_data_copy(d, &size);
  void* copy2 = pressio_data_copy(d, &size);
  EXPECT_NE(copy, data.data());
  EXPECT_NE(copy, copy2);

  pressio_data_free(d);
  free(copy);
  free(copy2);
}
TEST_F(PressioDataTests, Clone) {
  pressio_data* d = pressio_data_new_nonowning(pressio_int32_dtype, data.data(), 2, dims);
  pressio_data* cloned = pressio_data_new_clone(d);
  EXPECT_NE(d, nullptr);
  size_t bytes, bytes2;
  unsigned char * orig = (unsigned char*)pressio_data_ptr(d, &bytes);
  unsigned char * clone = (unsigned char*)pressio_data_ptr(cloned, &bytes2);
  ASSERT_EQ(bytes, bytes2);
  ASSERT_NE(orig, clone);
  EXPECT_TRUE(memcpy(orig, clone, bytes));

  pressio_data_free(d);
  pressio_data_free(cloned);
}

TEST_F(PressioDataTests, Select) {
  size_t dim[] = {9ul, 10ul};
  auto* data = pressio_data_new_owning(pressio_int32_dtype, 2ul, dim);
  auto* ptr = static_cast<int*>(pressio_data_ptr(data, nullptr));
  std::iota(ptr, ptr+90, 0);


  std::array<size_t,2> start{1,0};
  std::array<size_t,2> stride{5,7};
  std::array<size_t,2> count{2,2};
  std::array<size_t,2> block{2,3};
  std::array<int,24> expected = {
     1,  2,    6,  7,
    10, 11,   15, 16,
    19, 20,   24, 25,

    64, 65,   69, 70,
    73, 74,   78, 79,
    82, 83,   87, 88,
  };

  auto* slab = pressio_data_select(data,
      start.data(),
      stride.data(),
      count.data(),
      block.data());

  ASSERT_EQ(pressio_data_num_elements(slab), 24);
  ASSERT_EQ(pressio_data_dtype(slab), pressio_int32_dtype);

  //check the actual values
  std::array<int,24> results;
  std::copy_n((int*)pressio_data_ptr(slab, nullptr), 24, begin(results));
  EXPECT_EQ(results, expected);

  pressio_data_free(data);
  pressio_data_free(slab);
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> stride = {2, 3};
  std::vector<size_t> count = {2, 2};
  std::vector<size_t> start = {0, 0};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count),
      std::begin(stride),
      std::begin(start)
      );
  std::vector<int> results;
  int expected[] = {0, 2, 9, 11};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_default_start) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> stride = {2, 3};
  std::vector<size_t> count = {2, 2};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count),
      std::begin(stride)
      );
  std::vector<int> results;
  int expected[] = {0, 2, 9, 11};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}


TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_default_stride) {
  std::array<int, 12> values;
  std::iota(begin(values), end(values), 0);
  std::vector<size_t> global_dims = {3, 4};
  std::vector<size_t> count = {2, 2};
  auto range = std::make_shared<multi_dimensional_range<int>>(
      values.data(),
      std::begin(global_dims),
      std::end(global_dims),
      std::begin(count)
      );
  std::vector<int> results;
  int expected[] = {0, 1, 3, 4};
  std::copy(std::begin(*range), std::end(*range), std::back_inserter(results));
  EXPECT_THAT(results, ::testing::ElementsAreArray(expected));
}

TEST(test_mulit_dimensional_array, test_mulit_dimensional_array_operator) {
  size_t dim[] = {9ul, 10ul};
  auto* data = pressio_data_new_owning(pressio_int32_dtype, 2ul, dim);
  auto* ptr = static_cast<int*>(pressio_data_ptr(data, nullptr));
  std::iota(ptr, ptr+90, 0);
  
  std::array<size_t,2> start{1,0};
  std::array<size_t,2> stride{5,7};
  std::array<size_t,2> count{2,2};
  auto range = std::make_shared<multi_dimensional_range<int>>(ptr,
      dim,
      dim+2,
      std::begin(count),
      std::begin(stride),
      std::begin(start)
      );
  EXPECT_EQ((*range)(0,0), 1);
  EXPECT_EQ((*range)(1,0), 6);
  EXPECT_EQ((*range)(0,1), 64);
  EXPECT_EQ((*range)(1,1), 69);
  ASSERT_EQ(range->num_dims(), 2);
  EXPECT_EQ(range->get_global_dims(0), 9);
  EXPECT_EQ(range->get_global_dims(1), 10);
  EXPECT_EQ(range->get_local_dims(0), 2);
  EXPECT_EQ(range->get_local_dims(1), 2);

  pressio_data_free(data);
}


TEST(test_to_from_tuple, test_to_from_tuple) {
    std::tuple<int32_t, float, double, uint8_t> t{
        3, 4.2f, 3.7, 'x'
    };
    auto tuple_data = pressio_data(t);
    EXPECT_EQ(tuple_data.size_in_bytes(), 4+4+8+1);
    auto result = tuple_data.to_tuple<decltype(t)>();
    EXPECT_EQ(t, result);
}
TEST(test_join_split, test_join_split_headerlen) {
    pressio_data a{1,2,3};
    pressio_data b{1.2,2.3,3.4};
    pressio_data c{1.2f,2.3f};
    pressio_data joined = pressio_data::join({a,b,c}, pressio_data_header_len);
    std::vector<pressio_data> s{
        pressio_data::empty(pressio_int32_dtype, {}),
        pressio_data::empty(pressio_double_dtype, {}),
        pressio_data::empty(pressio_float_dtype, {}),
    };
    pressio_data::split(joined, pressio_data_header_len, s);
    EXPECT_EQ(s[0], a);
    EXPECT_EQ(s[1], b);
    EXPECT_EQ(s[2], c);
}
TEST(test_join_split, test_join_split_dimstype) {
    pressio_data a{1,2,3};
    pressio_data b{1.2,2.3,3.4};
    pressio_data c{1.2f,2.3f};
    pressio_data joined = pressio_data::join({a,b,c}, pressio_data_header_dimstype);
    std::vector<pressio_data> s(3);
    pressio_data::split(joined, pressio_data_header_dimstype, s);
    EXPECT_EQ(s[0], a);
    EXPECT_EQ(s[1], b);
    EXPECT_EQ(s[2], c);
}

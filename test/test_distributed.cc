#include <gtest/gtest-param-test.h>
#include <tuple>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <ostream>
#include <libpressio_ext/cpp/distributed_manager.h>
using namespace libpressio::distributed;

using ret_type = compat::optional<std::vector<size_t>>;

static size_t master = 0;
static size_t num_in_group(std::vector<size_t> const& w, size_t group) {
  return std::count(std::begin(w), std::end(w), group);
}
static size_t num_worker_groups(std::vector<size_t> const& w) {
  std::set<size_t> s;
  return std::count_if(std::begin(w), std::end(w), [&s](size_t i){
      if(i == master) return false;
      return s.insert(i).second;
  });
}
std::set<size_t> worker_groups(std::vector<size_t> const& w){ 
  std::set<size_t> s;
  std::copy_if(std::begin(w), std::end(w), std::inserter(s, std::end(s)),
      [](size_t i) {
        return i != master;
      });
  return s;
}

template<class Container, class Value>
bool element_of(Value&& v, Container && c) {
  return std::find(std::begin(c), std::end(c), std::forward<Value>(v)) != std::end(c);
}

namespace boost {
void PrintTo(::compat::optional<std::vector<size_t>> const& result, std::ostream* out) {
  if(result) {
    *out << '{';
    for (auto const& i : *result) {
      *out << i << ", ";
    }
    *out << '}';
  } else {
    *out << "empty";
  }
}
}

TEST(DistributedParams, InvalidRoot) {
  const unsigned int size = 4;
  const unsigned int root = 4;
  const unsigned int n_workers = 1;
  const unsigned int n_masters = 1;

  EXPECT_EQ(distributed_build_groups(size, n_workers, n_masters, root), ret_type());
}

TEST(DistributedParams, InvalidTotal) {
  const unsigned int root = 0;
  const unsigned int size = 4;
  const unsigned int n_workers = 8;
  const unsigned int n_masters = 1;

  EXPECT_EQ(distributed_build_groups(size, n_workers, n_masters, root), ret_type());
}

TEST(DistributedParams, InvalidAllAsMasters) {
  const unsigned int root = 0;
  const unsigned int size = 4;
  const unsigned int n_workers = 0;
  const unsigned int n_masters = 4;

  EXPECT_EQ(distributed_build_groups(size, n_workers, n_masters, root), ret_type());
}

TEST(DistributedParams, InvalidAllAsWorkers) {
  const unsigned int root = 0;
  const unsigned int size = 4;
  const unsigned int n_workers = 4;
  const unsigned int n_masters = 0;

  EXPECT_EQ(distributed_build_groups(size, n_workers, n_masters, root), ret_type());
}


TEST(DistributedParams, SizeOne) {
  const unsigned int root = 0;
  const unsigned int size = 1;
  const unsigned int n_workers = 0;
  const unsigned int n_masters = 0;

  EXPECT_EQ(distributed_build_groups(size, n_workers, n_masters, root), (ret_type{std::vector<size_t>{0ull}}));
}

TEST(DistributedParams, FullAuto) {
  const unsigned int root = 0;
  const unsigned int size = 8;
  const unsigned int n_workers = 0;
  const unsigned int n_masters = 0;

  auto result = distributed_build_groups(size, n_workers, n_masters, root);

  ASSERT_TRUE(result);
  EXPECT_EQ(num_in_group(*result, master), 1) << "there should be one master";
  EXPECT_EQ(num_worker_groups(*result), 7) << "there should be seven worker groups";
  for (auto i : worker_groups(*result)) {
    EXPECT_EQ(num_in_group(*result, i), 1) << "each worker group should have one member";
  }
  EXPECT_EQ((*result)[root], master);
}


TEST(DistributedParams, WorkersAuto) {
  const unsigned int root = 3;
  const unsigned int size = 8;
  const unsigned int n_workers = 0;
  const unsigned int n_masters = 4;

  auto result = distributed_build_groups(size, n_workers, n_masters, root);

  ASSERT_TRUE(result);
  EXPECT_EQ(num_in_group(*result, master), 4) << "there should be one master";
  EXPECT_EQ(num_worker_groups(*result), 4) << "there should be seven worker groups";
  for (auto i : worker_groups(*result)) {
    EXPECT_EQ(num_in_group(*result, i), 1) << "each worker group should have one member";
  }
  EXPECT_EQ((*result)[root], master);
}


TEST(DistributedParams, MastersAuto) {
  const unsigned int root = 3;
  const unsigned int size = 8;
  const unsigned int n_workers = 4;
  const unsigned int n_masters = 0;

  auto result = distributed_build_groups(size, n_workers, n_masters, root);

  ASSERT_TRUE(result);
  EXPECT_EQ(num_in_group(*result, master), 4) << "there should be one master";
  EXPECT_EQ(num_worker_groups(*result), 4) << "there should be seven worker groups";
  for (auto i : worker_groups(*result)) {
    EXPECT_EQ(num_in_group(*result, i), 1) << "each worker group should have one member";
  }
  EXPECT_EQ((*result)[root], master);
}

TEST(DistributedParams, SpecificWorkersAndMasters) {
  const unsigned int root = 3;
  const unsigned int size = 8;
  const unsigned int n_workers = 3;
  const unsigned int n_masters = 2;

  auto result = distributed_build_groups(size, n_workers, n_masters, root);

  ASSERT_TRUE(result);
  EXPECT_EQ(num_in_group(*result, master), 2) << "there should be one master";
  EXPECT_EQ(num_worker_groups(*result), 3) << "there should be seven worker groups";

  for (auto i : worker_groups(*result)) {
    EXPECT_EQ(num_in_group(*result, i), 2) << "each worker group should have 1 or 2 members";
  }
  EXPECT_EQ((*result)[root], master);
}


TEST(DistributedParams, SpecificWorkersAndMasters2) {
  const unsigned int root = 3;
  const unsigned int size = 8;
  const unsigned int n_workers = 3;
  const unsigned int n_masters = 1;

  auto result = distributed_build_groups(size, n_workers, n_masters, root);

  ASSERT_TRUE(result);
  EXPECT_EQ(num_in_group(*result, master), 1) << "there should be one master";
  EXPECT_EQ(num_worker_groups(*result), 3) << "there should be seven worker groups";

  for (auto i : worker_groups(*result)) {
    std::vector<size_t> expected{2,3};
    EXPECT_TRUE(element_of(num_in_group(*result, i), expected)) << "each worker group should have 2 or 3 members";
  }
  EXPECT_EQ((*result)[root], master);
}

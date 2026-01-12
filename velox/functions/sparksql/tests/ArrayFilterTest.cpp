/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {

class SparkArrayFilterTest : public SparkFunctionBaseTest {};

TEST_F(SparkArrayFilterTest, filterSingleParameterLambda) {
  auto rowVector = makeRowVector({makeArrayVector<int64_t>({{1, 2, 3}})});
  // Note: Using equalto() and pmod() instead of == and % because DuckDB parser
  // (used in tests) needs explicit function names. In actual Spark SQL, users
  // write: SELECT filter(array(1, 2, 3), x -> x % 2 == 1)
  auto result = evaluate<ArrayVector>(
      "filter(c0, x -> equalto(pmod(x, 2), 1))", rowVector);
  auto expected = makeArrayVector<int64_t>({{1, 3}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterIndexedLambda) {
  auto rowVector = makeRowVector({makeArrayVector<int64_t>({{0, 2, 3}})});
  // Note: Using greaterthan() instead of > because DuckDB parser (used in
  // tests) needs explicit function names. In actual Spark SQL, users write:
  // SELECT filter(array(0, 2, 3), (x, i) -> x > i)
  auto result = evaluate<ArrayVector>(
      "filter(c0, (x, i) -> greaterthan(x, i))", rowVector);
  auto expected = makeArrayVector<int64_t>({{2, 3}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterIndexedLambdaMultipleConditions) {
  auto rowVector = makeRowVector({makeArrayVector<int64_t>({{10, 20, 30}})});
  // Note: Using lessthan() instead of < because DuckDB parser needs explicit
  // function names. In actual Spark SQL, users write: SELECT filter(array(...),
  // (x, i) -> i < 2)
  auto result =
      evaluate<ArrayVector>("filter(c0, (x, i) -> lessthan(i, 2))", rowVector);
  auto expected = makeArrayVector<int64_t>({{10, 20}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterIndexedLambdaExactIndex) {
  auto rowVector = makeRowVector({makeArrayVector<int64_t>({{10, 20, 30}})});
  // Note: Using equalto() instead of == because DuckDB parser needs explicit
  // function names. In actual Spark SQL, users write: SELECT filter(array(...),
  // (x, i) -> i == 2)
  auto result =
      evaluate<ArrayVector>("filter(c0, (x, i) -> equalto(i, 2))", rowVector);
  auto expected = makeArrayVector<int64_t>({{30}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterEmptyArray) {
  auto rowVector = makeRowVector({makeArrayVector<int64_t>({{}})});
  auto result = evaluate<ArrayVector>("filter(c0, (x, i) -> true)", rowVector);
  auto expected = makeArrayVector<int64_t>({{}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterNull) {
  auto rowVector =
      makeRowVector({makeNullableArrayVector<int64_t>({std::nullopt})});
  auto result =
      evaluate<ArrayVector>("filter(c0, x -> greaterthan(x, 1))", rowVector);
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(SparkArrayFilterTest, filterWithCaptures) {
  auto rowVector = makeRowVector(
      {makeArrayVector<int64_t>({{1, 2, 3}, {4, 5, 6}}),
       makeFlatVector<int64_t>({1, 4})});
  // Note: Using greaterthan() instead of > because DuckDB parser needs explicit
  // function names. In actual Spark SQL, users write:
  // SELECT filter(c0, x -> x > c1)
  auto result =
      evaluate<ArrayVector>("filter(c0, x -> greaterthan(x, c1))", rowVector);
  auto expected = makeArrayVector<int64_t>({{2, 3}, {5, 6}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterMultipleRows) {
  auto rowVector = makeRowVector(
      {makeArrayVector<int64_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})});
  // Note: Using equalto() and pmod() instead of == and % because DuckDB parser
  // needs explicit function names. In actual Spark SQL, users write:
  // SELECT filter(c0, x -> x % 2 == 0)
  auto result = evaluate<ArrayVector>(
      "filter(c0, x -> equalto(pmod(x, 2), 0))", rowVector);
  auto expected = makeArrayVector<int64_t>({{2}, {4, 6}, {8}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterIndexedLambdaMultipleRows) {
  auto rowVector =
      makeRowVector({makeArrayVector<int64_t>({{0, 2, 3}, {1, 2, 3}})});
  // Note: Using greaterthan() instead of > because DuckDB parser needs explicit
  // function names. In actual Spark SQL, users write:
  // SELECT filter(c0, (x, i) -> x > i)
  auto result = evaluate<ArrayVector>(
      "filter(c0, (x, i) -> greaterthan(x, i))", rowVector);
  auto expected = makeArrayVector<int64_t>({{2, 3}, {1, 2, 3}});
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterWithNullElements) {
  // Test: filter(array(NULL, 2, 3), (x, i) -> x > i)
  // Expected: [2, 3] (NULL is filtered out in the lambda comparison)
  std::vector<std::optional<std::vector<std::optional<int64_t>>>> data = {
      {{std::nullopt, 2, 3}}};
  auto rowVector = makeRowVector({makeNullableArrayVector<int64_t>(data)});
  auto result = evaluate<ArrayVector>(
      "filter(c0, (x, i) -> greaterthan(x, i))", rowVector);
  std::vector<std::optional<std::vector<std::optional<int64_t>>>> expectedData =
      {{{2, 3}}};
  auto expected = makeNullableArrayVector<int64_t>(expectedData);
  assertEqualVectors(expected, result);
}

TEST_F(SparkArrayFilterTest, filterWithNullElementsLater) {
  // Test: filter(array(1, 2, NULL), (x, i) -> i > 1)
  // Expected: [NULL] (at i=2, the NULL element passes the i > 1 condition)
  std::vector<std::optional<std::vector<std::optional<int64_t>>>> data = {
      {{1, 2, std::nullopt}}};
  auto rowVector = makeRowVector({makeNullableArrayVector<int64_t>(data)});
  auto result = evaluate<ArrayVector>(
      "filter(c0, (x, i) -> greaterthan(i, 1))", rowVector);
  // Verify that the first row has one element (the NULL value)
  ASSERT_EQ(1, result->size());
  ASSERT_FALSE(result->isNullAt(0));
  auto arrayResult = result->as<ArrayVector>();
  ASSERT_EQ(1, arrayResult->sizeAt(0)); // Array has 1 element
  // The element is NULL
  ASSERT_TRUE(arrayResult->elements()->isNullAt(arrayResult->offsetAt(0)));
}

} // namespace facebook::velox::functions::sparksql::test

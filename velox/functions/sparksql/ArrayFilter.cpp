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
#include "velox/expression/Expr.h"
#include "velox/expression/LambdaExpr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Spark SQL array filter function with indexed lambda support.
// Supports two signatures:
// 1. array(T), function(T, boolean) -> array(T)
// 2. array(T), function(T, bigint, boolean) -> array(T)
class SparkArrayFilterFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2);

    // Flatten input array.
    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto& decodedArray = *arrayDecoder.get();
    auto flatArray = flattenArray(rows, args[0], decodedArray);

    VectorPtr elements = flatArray->elements();
    const auto numElements = elements->size();

    // Detect lambda signature by checking the FunctionVector's type.
    // The lambda signature is determined by the number of input types in
    // FunctionType.
    auto* funcVector = args[1]->asUnchecked<FunctionVector>();
    VELOX_CHECK_NOT_NULL(funcVector);

    auto funcType =
        std::dynamic_pointer_cast<const FunctionType>(funcVector->type());
    VELOX_CHECK_NOT_NULL(funcType);
    const bool hasIndexParam = (funcType->children().size() - 1) == 2;

    // Prepare lambda arguments.
    std::vector<VectorPtr> lambdaArgs = {elements};
    VectorPtr indices;

    if (hasIndexParam) {
      // Create 0-based indices for each element in the array.
      indices =
          createIndexVector(flatArray.get(), rows, numElements, context.pool());
      lambdaArgs.push_back(indices);
    }

    // Apply filter lambda.
    BufferPtr resultSizes;
    BufferPtr resultOffsets;
    BufferPtr selectedIndices;
    auto numSelected = doApply(
        rows,
        flatArray,
        args[1],
        lambdaArgs,
        context,
        resultOffsets,
        resultSizes,
        selectedIndices);

    // Wrap filtered elements.
    auto wrappedElements = numSelected ? BaseVector::wrapInDictionary(
                                             BufferPtr(nullptr),
                                             std::move(selectedIndices),
                                             numSelected,
                                             std::move(elements),
                                             true /*flattenIfRedundant*/)
                                       : nullptr;

    // Set nulls for rows not present in 'rows'.
    BufferPtr newNulls = addNullsForUnselectedRows(flatArray, rows);
    auto localResult = std::make_shared<ArrayVector>(
        flatArray->pool(),
        flatArray->type(),
        std::move(newNulls),
        rows.end(),
        std::move(resultOffsets),
        std::move(resultSizes),
        wrappedElements);
    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // array(T), function(T, boolean) -> array(T)
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .returnType("array(T)")
            .argumentType("array(T)")
            .argumentType("function(T,boolean)")
            .build(),
        // array(T), function(T, bigint, boolean) -> array(T)
        exec::FunctionSignatureBuilder()
            .typeVariable("T")
            .returnType("array(T)")
            .argumentType("array(T)")
            .argumentType("function(T,bigint,boolean)")
            .build()};
  }

 private:
  // Creates a vector of 0-based indices for array elements.
  // For array [a, b, c], creates [0, 1, 2].
  // For two arrays [a, b] and [c, d, e], creates [0, 1, 0, 1, 2].
  static VectorPtr createIndexVector(
      const ArrayVector* arrays,
      const SelectivityVector& rows,
      vector_size_t numElements,
      memory::MemoryPool* pool) {
    auto indices = std::make_shared<FlatVector<int64_t>>(
        pool,
        BIGINT(),
        BufferPtr(nullptr),
        numElements,
        AlignedBuffer::allocate<int64_t>(numElements, pool),
        std::vector<BufferPtr>());

    auto* rawIndices = indices->mutableRawValues();
    const auto* rawOffsets = arrays->rawOffsets();
    const auto* rawSizes = arrays->rawSizes();

    rows.applyToSelected([&](vector_size_t row) {
      if (arrays->isNullAt(row)) {
        return;
      }
      auto offset = rawOffsets[row];
      auto size = rawSizes[row];
      // Fill with 0-based indices: 0, 1, 2, ..., size-1
      std::iota(rawIndices + offset, rawIndices + offset + size, 0);
    });

    return indices;
  }

  // Applies filter lambda and returns the number of selected elements.
  static vector_size_t doApply(
      const SelectivityVector& rows,
      const std::shared_ptr<ArrayVector>& input,
      const VectorPtr& lambda,
      const std::vector<VectorPtr>& lambdaArgs,
      exec::EvalCtx& context,
      BufferPtr& resultOffsets,
      BufferPtr& resultSizes,
      BufferPtr& selectedIndices) {
    const auto* inputOffsets = input->rawOffsets();
    const auto* inputSizes = input->rawSizes();

    auto* pool = context.pool();
    resultSizes = allocateSizes(rows.end(), pool);
    resultOffsets = allocateOffsets(rows.end(), pool);
    auto* rawResultSizes = resultSizes->asMutable<vector_size_t>();
    auto* rawResultOffsets = resultOffsets->asMutable<vector_size_t>();

    const auto numElements = lambdaArgs[0]->size();
    selectedIndices = allocateIndices(numElements, pool);
    auto* rawSelectedIndices = selectedIndices->asMutable<vector_size_t>();

    vector_size_t numSelected = 0;

    auto elementToTopLevelRows =
        getElementToTopLevelRows(numElements, rows, input.get(), pool);

    exec::LocalDecodedVector bitsDecoder(context);
    auto iter = lambda->asUnchecked<FunctionVector>()->iterator(&rows);
    while (auto entry = iter.next()) {
      auto elementRows =
          toElementRows<ArrayVector>(numElements, *entry.rows, input.get());
      auto wrapCapture = toWrapCapture<ArrayVector>(
          numElements, entry.callable, *entry.rows, input);

      VectorPtr bits;
      entry.callable->apply(
          elementRows,
          nullptr,
          wrapCapture,
          &context,
          lambdaArgs,
          elementToTopLevelRows,
          &bits);
      bitsDecoder.get()->decode(*bits, elementRows);
      entry.rows->applyToSelected([&](vector_size_t row) {
        if (input->isNullAt(row)) {
          return;
        }
        auto size = inputSizes[row];
        auto offset = inputOffsets[row];
        rawResultOffsets[row] = numSelected;
        for (auto i = 0; i < size; ++i) {
          if (!bitsDecoder.get()->isNullAt(offset + i) &&
              bitsDecoder.get()->valueAt<bool>(offset + i)) {
            ++rawResultSizes[row];
            rawSelectedIndices[numSelected] = offset + i;
            ++numSelected;
          }
        }
      });
    }

    selectedIndices->setSize(numSelected * sizeof(vector_size_t));
    return numSelected;
  }
};

} // namespace

/// Spark SQL filter is null preserving for the array. But since an
/// expr tree with a lambda depends on all named fields, including
/// captures, a null in a capture does not automatically make a
/// null result.

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_spark_array_filter,
    SparkArrayFilterFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<SparkArrayFilterFunction>());

} // namespace facebook::velox::functions::sparksql

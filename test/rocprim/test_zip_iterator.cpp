// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

// required rocprim headers
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/iterator/zip_iterator.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

TEST(RocprimZipIteratorTests, Traits)
{
    ASSERT_TRUE((
        std::is_same<
            rocprim::zip_iterator<
                rocprim::tuple<int*, double*, char*>
            >::reference,
            rocprim::tuple<int&, double&, char&>
        >::value
    ));
    ASSERT_TRUE((
        std::is_same<
            rocprim::zip_iterator<
                rocprim::tuple<const int*, double*, const char*>
            >::reference,
            rocprim::tuple<const int&, double&, const char&>
        >::value
    ));
    auto to_double = [](const int& x) -> double { return double(x); };
    ASSERT_TRUE((
        std::is_same<
            rocprim::zip_iterator<
                rocprim::tuple<
                    rocprim::counting_iterator<int>,
                    rocprim::transform_iterator<int*, decltype(to_double)>
                >
            >::reference,
            rocprim::tuple<
                rocprim::counting_iterator<int>::reference,
                rocprim::transform_iterator<int*, decltype(to_double)>::reference
            >
        >::value
    ));
}

TEST(RocprimZipIteratorTests, Basics)
{
    int a[] = { 1, 2, 3, 4, 5};
    int b[] = { 6, 7, 8, 9, 10};
    double c[] = { 1., 2., 3., 4., 5.};
    auto iterator_tuple = rocprim::make_tuple(a, b, c);

    // Constructor
    rocprim::zip_iterator<decltype(iterator_tuple)> zit(iterator_tuple);

    // dereferencing
    ASSERT_EQ(*zit, rocprim::make_tuple(1, 6, 1.0));
    *zit = rocprim::make_tuple(1, 8, 15);
    ASSERT_EQ(*zit, rocprim::make_tuple(1, 8, 15.0));
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(b[0], 8);
    ASSERT_EQ(c[0], 15.0);
    auto ref = *zit;
    ref = rocprim::make_tuple(1, 6, 1.0);
    ASSERT_EQ(*zit, rocprim::make_tuple(1, 6, 1.0));
    ASSERT_EQ(a[0], 1);
    ASSERT_EQ(b[0], 6);
    ASSERT_EQ(c[0], 1.0);

    // inc, dec, advance
    ++zit;
    ASSERT_EQ(*zit, rocprim::make_tuple(2, 7, 2.0));
    zit++;
    ASSERT_EQ(*zit, rocprim::make_tuple(3, 8, 3.0));
    --zit;
    ASSERT_EQ(*zit, rocprim::make_tuple(2, 7, 2.0));
    zit--;
    ASSERT_EQ(*zit, rocprim::make_tuple(1, 6, 1.0));
    zit += 3;
    ASSERT_EQ(*zit, rocprim::make_tuple(4, 9, 4.0));
    zit -= 2;
    ASSERT_EQ(*zit, rocprim::make_tuple(2, 7, 2.0));

    // <, >, <=, >=, ==, !=
    auto zit2 = rocprim::zip_iterator<decltype(iterator_tuple)>(iterator_tuple);
    ASSERT_TRUE(zit2 < zit);
    ASSERT_TRUE(zit > zit2);

    ASSERT_TRUE(zit2 <= zit);
    ASSERT_TRUE(zit <= zit);
    ASSERT_TRUE(zit2 <= zit2);
    ASSERT_TRUE(zit >= zit2);
    ASSERT_TRUE(zit >= zit);
    ASSERT_TRUE(zit2 >= zit2);

    ASSERT_TRUE(zit2 != zit);
    ASSERT_TRUE(zit != zit2);
    ASSERT_TRUE(zit2 != ++rocprim::zip_iterator<decltype(iterator_tuple)>(iterator_tuple));

    ASSERT_TRUE(zit2 == zit2);
    ASSERT_TRUE(zit == zit);
    ASSERT_TRUE(zit2 == rocprim::zip_iterator<decltype(iterator_tuple)>(iterator_tuple));

    // distance
    ASSERT_EQ(zit - zit2, 1);
    ASSERT_EQ(zit2 - zit, -1);
    ASSERT_EQ(zit - zit, 0);

    // []
    ASSERT_EQ((zit2[0]), rocprim::make_tuple(1, 6, 1.0));
    ASSERT_EQ((zit2[2]), rocprim::make_tuple(3, 8, 3.0));
    // +
    ASSERT_EQ(*(zit2+3), rocprim::make_tuple(4, 9, 4.0));
}

template<class T1, class T2, class T3>
struct tuple3_transform_op
{
    __device__ __host__
    T1 operator()(const rocprim::tuple<T1, T2, T3>& t) const
    {
        return T1(rocprim::get<0>(t) + rocprim::get<1>(t) + rocprim::get<2>(t));
    }
};

TEST(RocprimZipIteratorTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T1 = int;
    using T2 = double;
    using T3 = unsigned char;
    using U = T1;
    const bool debug_synchronous = false;
    const size_t size = 1024 * 16;

    // using default stream
    hipStream_t stream = 0;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T1> input1 = test_utils::get_random_data_wrapped<T1>(size, 1, 100, seed_value);
        std::vector<T2> input2 = test_utils::get_random_data_wrapped<T2>(size, 1, 100, seed_value);
        std::vector<T3> input3 = test_utils::get_random_data_wrapped<T3>(size, 1, 100, seed_value);
        std::vector<U> output(input1.size());

        common::device_ptr<T1> d_input1(input1);
        common::device_ptr<T2> d_input2(input2);
        common::device_ptr<T3> d_input3(input3);
        common::device_ptr<U>  d_output(output.size());

        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<U> expected(input1.size());
        std::transform(
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input1.begin(), input2.begin(), input3.begin())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input1.end(), input2.end(), input3.end())
            ),
            expected.begin(),
            tuple3_transform_op<T1, T2, T3>()
        );

        // Run
        HIP_CHECK(rocprim::transform(
            rocprim::make_zip_iterator(
                rocprim::make_tuple(d_input1.get(), d_input2.get(), d_input3.get())),
            d_output.get(),
            input1.size(),
            tuple3_transform_op<T1, T2, T3>(),
            stream,
            debug_synchronous));
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output = d_output.load();

        // Check if output values are as expected
        // precision of tuple3_transform_op<T1, T2, T3> is precision<T1> * 2
        // presision of T1 -> U is precision<U>
        test_utils::assert_near(output,
                                expected,
                                std::max(test_utils::precision<U>, test_utils::precision<T1> * 2));
    }

}

template<class T1, class T2, class T3>
struct tuple3to2_transform_op
{
    __device__ __host__ inline
    rocprim::tuple<T1, T2> operator()(const rocprim::tuple<T1, T2, T3>& t) const
    {
        return rocprim::make_tuple(
            rocprim::get<0>(t), T2(rocprim::get<1>(t) + rocprim::get<2>(t))
        );
    }
};

template<class T1, class T2>
struct tuple2_reduce_op
{
    __device__ __host__ inline
    rocprim::tuple<T1, T2> operator()(const rocprim::tuple<T1, T2>& t1,
                                      const rocprim::tuple<T1, T2>& t2) const
    {
        return rocprim::make_tuple(
            rocprim::get<0>(t1) + rocprim::get<0>(t2),
            rocprim::get<1>(t1) + rocprim::get<1>(t2)
        );
    };
};

TEST(RocprimZipIteratorTests, TransformReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T1 = int;
    using T2 = unsigned int;
    using T3 = unsigned char;
    using U1 = T1;
    using U2 = T2;
    const bool debug_synchronous = false;
    const size_t size = 1024 * 16;

    // using default stream
    hipStream_t stream = 0;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T1> input1 = test_utils::get_random_data_wrapped<T1>(size, 1, 100, seed_value);
        std::vector<T2> input2 = test_utils::get_random_data_wrapped<T2>(size, 1, 50, seed_value);
        std::vector<T3> input3 = test_utils::get_random_data_wrapped<T3>(size, 1, 10, seed_value);
        std::vector<U1> output1(1, 0);
        std::vector<U2> output2(1, 0);

        common::device_ptr<T1> d_input1(input1);
        common::device_ptr<T2> d_input2(input2);
        common::device_ptr<T3> d_input3(input3);
        common::device_ptr<U1> d_output1(output1.size());
        common::device_ptr<U2> d_output2(output2.size());

        // Calculate expected results on host
        U1 expected1 = std::accumulate(input1.begin(), input1.end(), T1(0));
        U2 expected2 = std::accumulate(input2.begin(), input2.end(), T2(0))
            + std::accumulate(input3.begin(), input3.end(), T2(0));

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::reduce(
            nullptr,
            temp_storage_size_bytes,
            rocprim::make_transform_iterator(
                rocprim::make_zip_iterator(
                    rocprim::make_tuple(d_input1.get(), d_input2.get(), d_input3.get())),
                tuple3to2_transform_op<T1, T2, T3>()),
            rocprim::make_zip_iterator(rocprim::make_tuple(d_output1.get(), d_output2.get())),
            input1.size(),
            tuple2_reduce_op<T1, T2>(),
            stream,
            debug_synchronous));
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);
        ASSERT_NE(d_temp_storage.get(), nullptr);

        // Run
        HIP_CHECK(rocprim::reduce(
            d_temp_storage.get(),
            temp_storage_size_bytes,
            rocprim::make_transform_iterator(
                rocprim::make_zip_iterator(
                    rocprim::make_tuple(d_input1.get(), d_input2.get(), d_input3.get())),
                tuple3to2_transform_op<T1, T2, T3>()),
            rocprim::make_zip_iterator(rocprim::make_tuple(d_output1.get(), d_output2.get())),
            input1.size(),
            tuple2_reduce_op<T1, T2>(),
            stream,
            debug_synchronous));
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output1 = d_output1.load();
        output2 = d_output2.load();

        // Check if output values are as expected
        // precision of tuple3to2_transform_op<T1, T2, T3> is (0, precision<T2>)
        // precision of subsequent reduction by tuple2_reduce_op<T1, T2> is
        // (max(precision<T1>, precision<U1>), max(precision<T2>, precision<U2>)) * size)
        test_utils::assert_near(output1[0],
                                expected1,
                                std::max(test_utils::precision<T1>, test_utils::precision<U1>)
                                    * size);

        test_utils::assert_near(output2[0],
                                expected2,
                                (std::max(test_utils::precision<T2>, test_utils::precision<U2>)
                                 + test_utils::precision<T2>)*size);
    }

}

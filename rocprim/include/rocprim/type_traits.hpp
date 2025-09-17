// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_TYPE_TRAITS_HPP_
#define ROCPRIM_TYPE_TRAITS_HPP_

#include "config.hpp"
#include "functional.hpp"

#include "type_traits_interface.hpp"

#include "types/tuple.hpp"

#include <functional>
#include <utility>

/// \addtogroup utilsmodule_typetraits
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Extension of `std::make_unsigned`, which includes support for 128-bit integers.
template<class T>
struct make_unsigned : std::make_unsigned<T>
{};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<>
struct make_unsigned<::rocprim::int128_t>
{
    using type = ::rocprim::uint128_t;
};

template<>
struct make_unsigned<::rocprim::uint128_t>
{
    using type = ::rocprim::uint128_t;
};
#endif

static_assert(std::is_same<make_unsigned<::rocprim::int128_t>::type, ::rocprim::uint128_t>::value,
              "'rocprim::int128_t' needs to implement 'make_unsigned' trait.");

/// \brief Extension of `std::numeric_limits`, which includes support for 128-bit integers.
template<class T>
struct numeric_limits : std::numeric_limits<T>
{};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<>
struct numeric_limits<rocprim::uint128_t> : std::numeric_limits<unsigned int>
{
    static constexpr int digits   = 128;
    static constexpr int digits10 = 38;

    static constexpr rocprim::uint128_t max()
    {
        return rocprim::int128_t{-1};
    }

    static constexpr rocprim::uint128_t min()
    {
        return rocprim::uint128_t{0};
    }

    static constexpr rocprim::uint128_t lowest()
    {
        return min();
    }
};

template<>
struct numeric_limits<rocprim::int128_t> : std::numeric_limits<int>
{
    static constexpr int digits   = 127;
    static constexpr int digits10 = 38;

    static constexpr rocprim::int128_t max()
    {
        return numeric_limits<rocprim::uint128_t>::max() >> 1;
    }

    static constexpr rocprim::int128_t min()
    {
        return -numeric_limits<rocprim::int128_t>::max() - 1;
    }

    static constexpr rocprim::int128_t lowest()
    {
        return min();
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Used to retrieve a type that can be treated as unsigned version of the template parameter.
/// \tparam T The signed type to find an unsigned equivalent for.
/// \tparam size the desired size (in bytes) of the unsigned type
template<typename T, int size = 0>
struct get_unsigned_bits_type
{
    using unsigned_type = typename get_unsigned_bits_type<T, sizeof(T)>::
        unsigned_type; ///< Typedefed to the unsigned type.
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<typename T>
struct get_unsigned_bits_type<T, 1>
{
    using unsigned_type = uint8_t;
};

template<typename T>
struct get_unsigned_bits_type<T, 2>
{
    using unsigned_type = uint16_t;
};

template<typename T>
struct get_unsigned_bits_type<T, 4>
{
    using unsigned_type = uint32_t;
};

template<typename T>
struct get_unsigned_bits_type<T, 8>
{
    using unsigned_type = uint64_t;
};

template<typename T>
struct get_unsigned_bits_type<T, 16>
{
    using unsigned_type = ::rocprim::uint128_t;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
  static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
  return key ^ mask;
}

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key ;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleIn is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleIn(UnsignedBits key) ->
    typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
}

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key;
};

template<typename T, typename UnsignedBits>
[[deprecated("TwiddleOut is deprecated."
             "Use radix_key_codec instead.")]] static ROCPRIM_DEVICE ROCPRIM_INLINE auto
    TwiddleOut(UnsignedBits key) ->
    typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace detail
{

// invoke_result is based on std::invoke_result.
// The main difference is using ROCPRIM_HOST_DEVICE, this allows to
// use invoke_result with device-only lambdas/functors in host-only functions
// on HIP-clang.

template<class T>
struct is_reference_wrapper : std::false_type
{};
template<class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type
{};

template<class T>
struct invoke_impl
{
    template<class F, class... Args>
    ROCPRIM_HOST_DEVICE static auto call(F&& f, Args&&... args)
        -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
};

template<class B, class MT>
struct invoke_impl<MT B::*>
{
    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<std::is_base_of<B, Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> T&&;

    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<is_reference_wrapper<Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> decltype(t.get());

    template<class T,
             class Td = typename std::decay<T>::type,
             class    = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
             class    = typename std::enable_if<!is_reference_wrapper<Td>::value>::type>
    ROCPRIM_HOST_DEVICE static auto get(T&& t) -> decltype(*std::forward<T>(t));

    template<class T,
             class... Args,
             class MT1,
             class = typename std::enable_if<std::is_function<MT1>::value>::type>
    ROCPRIM_HOST_DEVICE static auto call(MT1 B::*pmf, T&& t, Args&&... args)
        -> decltype((invoke_impl::get(std::forward<T>(t)).*pmf)(std::forward<Args>(args)...));

    template<class T>
    ROCPRIM_HOST_DEVICE static auto call(MT B::*pmd, T&& t)
        -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
};

template<class F, class... Args, class Fd = typename std::decay<F>::type>
ROCPRIM_HOST_DEVICE auto INVOKE(F&& f, Args&&... args)
    -> decltype(invoke_impl<Fd>::call(std::forward<F>(f), std::forward<Args>(args)...));

// Conforming C++14 implementation (is also a valid C++11 implementation):
template<typename AlwaysVoid, typename, typename...>
struct invoke_result_impl
{};
template<typename F, typename... Args>
struct invoke_result_impl<decltype(void(INVOKE(std::declval<F>(), std::declval<Args>()...))),
                          F,
                          Args...>
{
    using type = decltype(INVOKE(std::declval<F>(), std::declval<Args>()...));
};

template<class T>
struct is_tuple
{
public:
    static constexpr bool value = false;
};

template<class... Args>
struct is_tuple<::rocprim::tuple<Args...>>
{
private:
    template<size_t Index>
    ROCPRIM_HOST_DEVICE
    static constexpr bool is_tuple_impl()
    {
        return is_tuple_impl<Index + 1>();
    }

    template<>
    ROCPRIM_HOST_DEVICE
    static constexpr bool is_tuple_impl<sizeof...(Args)>()
    {
        return true;
    }

public:
    static constexpr bool value = is_tuple_impl<0>();
};

template<class T>
struct is_tuple_of_references
{
    static_assert(sizeof(T) == 0, "is_tuple_of_references is only implemented for rocprim::tuple");
};

template<class... Args>
struct is_tuple_of_references<::rocprim::tuple<Args...>>
{
private:
    template<size_t Index>
    ROCPRIM_HOST_DEVICE static constexpr bool is_tuple_of_references_impl()
    {
        using tuple_t   = ::rocprim::tuple<Args...>;
        using element_t = ::rocprim::tuple_element_t<Index, tuple_t>;
        return std::is_reference<element_t>::value && is_tuple_of_references_impl<Index + 1>();
    }

    template<>
    ROCPRIM_HOST_DEVICE static constexpr bool is_tuple_of_references_impl<sizeof...(Args)>()
    {
        return true;
    }

public:
    static constexpr bool value = is_tuple_of_references_impl<0>();
};

template<typename Iterator>
using value_type_t = typename std::iterator_traits<Iterator>::value_type;

template<typename EqualityOp, int Ret = 0>
struct guarded_inequality_wrapper
{
    /// Wrapped equality operator
    EqualityOp op;

    /// Out-of-bounds limit
    size_t guard;

    /// Constructor
    ROCPRIM_HOST_DEVICE inline guarded_inequality_wrapper(EqualityOp op, size_t guard)
        : op(op), guard(guard)
    {}

    /// \brief Guarded boolean inequality operator.
    ///
    /// \tparam T Type of the operands compared by the equality operator
    /// \param a Left hand-side operand
    /// \param b Right hand-side operand
    /// \param idx Index of the thread calling to this operator. This is used to determine which
    /// operations are out-of-bounds
    /// \returns <tt>!op(a, b)</tt> for a certain equality operator \p op when in-bounds.
    template<typename T>
    ROCPRIM_HOST_DEVICE
    inline bool
        operator()(const T& a, const T& b, size_t idx) const
    {
        // In-bounds return operation result, out-of-bounds return ret.
        return (idx < guard) ? !op(a, b) : Ret;
    }
};

} // end namespace detail

/// \brief Behaves like ``std::invoke_result``, but allows the use of invoke_result
/// with device-only lambdas/functors in host-only functions on HIP-clang.
///
/// \tparam F Type of the function.
/// \tparam Args Input type(s) to the function ``F``.
template<class F, class... Args>
struct invoke_result : detail::invoke_result_impl<void, F, Args...>
{
#ifdef DOXYGEN_DOCUMENTATION_BUILD
    /// \brief The return type of the Callable type F if invoked with the arguments Args.
    /// \hideinitializer
    using type = detail::invoke_result_impl<void, F, Args...>::type;
#endif // DOXYGEN_DOCUMENTATION_BUILD
};

/// \brief Helper type. It is an alias for ``invoke_result::type``.
///
/// \tparam F Type of the function.
/// \tparam Args Input type(s) to the function ``F``.
template<class F, class... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

/// \brief Utility wrapper around ``invoke_result`` for binary operators.
///
/// \tparam T Input type to the binary operator.
/// \tparam F Type of the binary operator.
template<class T, class F>
struct invoke_result_binary_op
{
    /// \brief The result type of the binary operator.
    using type = typename invoke_result<F, T, T>::type;
};

/// \brief Helper type. It is an alias for ``invoke_result_binary_op::type``.
///
/// \tparam T Input type to the binary operator.
/// \tparam F Type of the binary operator.
template<class T, class F>
using invoke_result_binary_op_t = typename invoke_result_binary_op<T, F>::type;

namespace detail
{

/// \brief If `T` is a rocPRIM binary functional type, provides the member constant `value` equal `true`.
///   For any other type, `value` is `false`.
template<typename T>
struct is_binary_functional
{
    static constexpr bool value = false;
};

template<typename T>
struct is_binary_functional<less<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<less_equal<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<greater<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<greater_equal<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<equal_to<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<not_equal_to<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<plus<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<minus<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<multiplies<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<maximum<T>>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_binary_functional<minimum<T>>
{
    static constexpr bool value = true;
};

} // namespace detail

/// \brief Helper struct it has the Type and the number of aligned bytes.
///
/// \tparam T is the Type used to get the number of aligned bytes.
template<typename T>
struct align_bytes
{
    /// Number of aligned bytes for type T
    static constexpr unsigned value = alignof(T);
    /// Type defined by T
    using Type = T;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip specialized versions
template<typename T>
struct align_bytes<volatile T> : align_bytes<T>
{};
template<typename T>
struct align_bytes<const T> : align_bytes<T>
{};
template<typename T>
struct align_bytes<const volatile T> : align_bytes<T>
{};
#endif

namespace detail
{

template<typename T>
struct word_type
{
    static constexpr auto align_bytes_value = align_bytes<T>::value;

    template<typename Unit>
    struct IsMultiple
    {
        static constexpr auto unit_align_bytes = align_bytes<Unit>::value;
        static constexpr bool is_multiple
            = (sizeof(T) % sizeof(Unit) == 0)
              && (int(align_bytes_value) % int(unit_align_bytes) == 0);
    };

    using type = typename std::conditional<IsMultiple<int>::is_multiple,
                                           unsigned int,
                                           typename std::conditional<IsMultiple<short>::is_multiple,
                                                                     unsigned short,
                                                                     unsigned char>::type>::type;
};

template<typename T>
struct word_type<volatile T> : word_type<T>
{};
template<typename T>
struct word_type<const T> : word_type<T>
{};
template<typename T>
struct word_type<const volatile T> : word_type<T>
{};

} // namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule_typetraits

#endif // ROCPRIM_TYPE_TRAITS_HPP_

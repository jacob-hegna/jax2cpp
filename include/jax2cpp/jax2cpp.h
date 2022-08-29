#pragma once

#ifdef J2C_USE_EIGEN
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#endif

namespace j2c {

enum class Backend {
  kEigen,
  kBlaze,
  kXTensor,
};

template <Backend backend> struct Lax {};

template <int... Is> struct Shape {};

#ifdef J2C_USE_EIGEN
#include <Eigen/Eigen>

template <> struct Lax<Backend::kEigen> {
  template <class T> struct convert_from_shape {};
  template <int... Is> struct convert_from_shape<Shape<Is...>> {
    using type = Eigen::Sizes<Is...>;

    static auto to_array() { return Eigen::array<int, sizeof...(Is)>({Is...}); }
  };

  template <class FloatT, class Shape>
  using Tensor =
      Eigen::TensorFixedSize<FloatT, typename convert_from_shape<Shape>::type>;

  template <class FloatT, class Shape, class T>
  Tensor<FloatT, Shape> static make_tensor(const T &t) {
    Tensor<FloatT, Shape> tensor;
    tensor.setValues(t);
    return tensor;
  }

  template <class ShapeT, class T> static auto internal_reshape(const T &mat) {
    return mat.reshape(convert_from_shape<ShapeT>::to_array());
  }

  template <class PermutationT, class T> static auto transpose(const T &mat) {
    return mat.shuffle(convert_from_shape<PermutationT>::to_array());
  }

  template <class T> static auto abs(const T &mat) { return mat.abs(); }
  template <class T, class S> static auto add(const T &lhs, const S &rhs) {
    return lhs + rhs;
  }
  template <class T, class S> static auto sub(const T &lhs, const S &rhs) {
    return lhs - rhs;
  }
  template <class T, class S> static auto mul(const T &lhs, const S &rhs) {
    return lhs * rhs;
  }
  template <class T, class S> static auto div(const T &lhs, const S &rhs) {
    return lhs / rhs;
  }
  template <class T> static auto neg(const T &mat) { return -mat; }
  template <int I, class T> static auto integer_pow_fast(const T &lhs) {
    if constexpr (I == 1) {
      return lhs;
    } else if constexpr (I == 2) {
      return lhs.square();
    } else if constexpr (I == 3) {
      return lhs.cube();
    } else {
      return integer_pow(lhs, I);
    }
  }
  template <class T> static auto integer_pow(const T &lhs, int rhs) {
    return lhs.pow(static_cast<double>(rhs));
  }
  template <class T> static auto exp(const T &mat) { return mat.exp(); }
  template <class T> static auto tanh(const T &lhs) { return lhs.tanh(); }
  template <class T, class S>
  static auto dot_general(const T &lhs, const S &rhs) {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};

    return lhs.contract(rhs, product_dims);
  }
  template <class T, class S> static auto pow(const T &mat, S exp) {
    return mat.pow(exp);
  }
  template <class ReshapeT, class BroadcastT, class ShuffleT, class T>
  static auto broadcast_in_dim(const T &mat) {
    return internal_reshape<ReshapeT>(mat)
        .broadcast(convert_from_shape<BroadcastT>::to_array())
        .shuffle(convert_from_shape<ShuffleT>::to_array());
  }
  template <class DimsT, class T> static auto reduce_sum(const T &mat) {
    return mat.sum(convert_from_shape<DimsT>::to_array());
  }
  template <class T> static auto convert_element_type(const T &mat) {
    return mat;
  }
};
#endif

#ifdef J2C_USE_BLAZE
#include <blaze/blaze.h>

template <> struct Lax<Backend::kBlaze> {
  template <class T> T abs(const T &mat) { return blaze::abs(mat); }
};
#endif

} // namespace j2c

//
// Created by hans on 08.07.25.
//

#ifndef QUATERNION_CUH
#define QUATERNION_CUH

#include "Vec3.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

struct Quaternion {
  float w, x, y, z;
  __host__ __device__ Quaternion() = default;

  __host__ __device__ Quaternion(const float w, const float x, const float y,
                                 const float z)
      : w(w), x(x), y(y), z(z) {}

  explicit __host__ __device__ Quaternion(const Vec3 &vec)
      : w(1.0f), x(vec.x), y(vec.y), z(vec.z) {}

  __host__ __device__ __forceinline__ Quaternion
  operator+=(const Quaternion &q) {
    this->w += q.w;
    this->x += q.x;
    this->y += q.y;
    this->z += q.z;
    return *this;
  }

  __host__ __device__ __forceinline__ Quaternion
  operator-=(const Quaternion &q) {
    this->w -= q.w;
    this->x -= q.x;
    this->y -= q.y;
    this->z -= q.z;
    return *this;
  }

  __host__ __device__ __forceinline__ Quaternion
  operator*(const Quaternion &q) const {
    return {w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w};
  }

  __host__ __device__ __forceinline__ Quaternion
  operator*(const float s) const {
    return {w * s, x * s, y * s, z * s};
  }

  __host__ __device__ __forceinline__ friend Quaternion
  operator*(const float lhs, const Quaternion rhs) {
    return rhs * lhs;
  }

  __host__ __device__ __forceinline__ float3 toEulerAngles() const {
    const float sinr = 2.0f * (w * x + y * z);
    const float cosr = 1.0f - 2.0f * (x * x + y * y);
    const float roll = atan2f(sinr, cosr);

    float sinp = 2.0f * (w * y - z * x);
    sinp = fminf(1.0f, fmaxf(-1.0f, sinp));
    const float pitch = asinf(sinp);

    const float siny = 2.0f * (w * z + x * y);
    const float cosy = 1.0f - 2.0f * (y * y + z * z);
    const float yaw = atan2f(siny, cosy);

    constexpr float RAD2DEG = 180.0f / CUDART_PI_F;
    return make_float3(roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG);
  }

  __host__ __device__ __forceinline__ Quaternion normalize() const {
    const auto length = this->length();
    if (length <= 0.0f)
      return {1, 0, 0, 0};
    const auto inv = 1.0f / length;
    return {w * inv, x * inv, y * inv, z * inv};
  }

  __host__ __device__ __forceinline__ float length() const {
    return sqrtf(w * w + x * x + y * y + z * z);
  }

  __host__ __device__ __forceinline__ Vec3 rotate(const Vec3 &v) const {
    Vec3 u{x, y, z};
    Vec3 uv = cross(u, v);
    Vec3 uuv = cross(u, uv);
    return v + uv * (2.0f * w) + uuv * 2.0f;
  }
};

__host__ __forceinline__ Quaternion
QuaternionFromJson(const nlohmann::json &a) {
  return Quaternion{a[0].get<float>(), a[1].get<float>(), a[2].get<float>(),
                    a[3].get<float>()};
}

__host__ __device__ __forceinline__ Quaternion conjugate(const Quaternion &q) {
  return {q.w, -q.x, -q.y, -q.z};
}
#endif // QUATERNION_CUH

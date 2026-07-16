//
// Created by hans on 07.07.25.
//

#ifndef VEC3_CUH
#define VEC3_CUH
#include <math.h>

struct Vec3 {
  float x, y, z;
  Vec3() = default;
  __host__ __device__ Vec3(const float x, const float y, const float z)
      : x(x), y(y), z(z) {}
  __host__ __device__ __forceinline__ bool
  operator==(const Vec3 &v) const noexcept {
    return this->x == v.x && this->y == v.y && this->z == v.z;
  }
  __host__ __device__ __forceinline__ Vec3 &
  operator+=(const Vec3 &rhs) noexcept {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }

  __host__ __device__ __forceinline__ Vec3 &
  operator-=(const Vec3 &rhs) noexcept {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
  }

  __host__ __device__ __forceinline__ Vec3 operator+(const Vec3 &rhs) const {
    return {this->x + rhs.x, this->y + rhs.y, this->z + rhs.z};
  }

  __host__ __device__ __forceinline__ Vec3 operator-(const Vec3 &rhs) const {
    return {this->x - rhs.x, this->y - rhs.y, this->z - rhs.z};
  }
  __host__ __device__ __forceinline__ friend Vec3 operator-(const Vec3 &v) {
    return {-v.x, -v.y, -v.z};
  }

  __host__ __device__ __forceinline__ Vec3 operator*(const float rhs) const {
    return {this->x * rhs, this->y * rhs, this->z * rhs};
  }

  __host__ __device__ __forceinline__ Vec3 operator*(const Vec3 v) const {
    return {this->x * v.x, this->y * v.y, this->z * v.z};
  }

  __host__ __device__ __forceinline__ friend Vec3
  operator*(const float lhs, const Vec3 &rhs) noexcept {
    return rhs * lhs;
  }

  __host__ __device__ __forceinline__ Vec3 operator/(const float rhs) const {
    return {this->x / rhs, this->y / rhs, this->z / rhs};
  }
};
__host__ __device__ __forceinline__ int3 toInt3(const Vec3 v) {
  return {static_cast<int>(v.x), static_cast<int>(v.y), static_cast<int>(v.z)};
}
__host__ __device__ __forceinline__ Vec3 ceil(const Vec3 v) {
  return {ceilf(v.x), ceilf(v.y), ceilf(v.z)};
}
__host__ __device__ __forceinline__ int3 toInt3ceil(Vec3 v) {
  return {toInt3(ceil(v))};
}
__host__ __device__ __forceinline__ float dot(const Vec3 &lhs,
                                              const Vec3 &rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ __forceinline__ Vec3 cross(const Vec3 &lhs,
                                               const Vec3 &rhs) {
  return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}

__host__ __device__ __forceinline__ float length(const Vec3 &v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ __forceinline__ Vec3 normalize(const Vec3 &v) {
  const float len = length(v);
  return len == 0 ? v : v / len;
}
#endif // VEC3_CUH

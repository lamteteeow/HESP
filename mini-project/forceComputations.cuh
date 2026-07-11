//
// Created by hans on 16.07.25.
//

#ifndef FORCECOMPUTATIONS_CUH
#define FORCECOMPUTATIONS_CUH
#include "PlaneDevice.cuh"
#include "SphereDevice.cuh"

__host__ __device__ __forceinline__ void sphereOnSphere(const size_t i,
                                                        const int j,
                                                        const SphereDevice sd,
                                                        Vec3 &fi, Vec3 &ti) {
  const Vec3 pi = sd.d_positions[i];
  const Vec3 vi = sd.d_velocities[i];
  const float ri = sd.d_radii[i];
  const Vec3 pj = sd.d_positions[j];
  const Vec3 vj = sd.d_velocities[j];
  const Vec3 x_delta = pi - pj;
  const float distance = length(x_delta);
  const float overlap = (ri + sd.d_radii[j]) - distance;
  if (overlap <= 0.0f)
    return;
  const Vec3 v_delta = vi - vj;
  const Vec3 x_hat = x_delta / distance;
  const float v_rel_normal = dot(v_delta, x_hat);
  const float f_n_mag = sd.d_kn[i] * overlap - sd.d_gamma_n[i] * v_rel_normal;
  const Vec3 f_n = f_n_mag * x_hat;

  const float coulomb = sd.d_mu[i] * fabs(f_n_mag);

  const Vec3 v_normal = x_hat * v_rel_normal;
  const Vec3 rotational_speed = ri * sd.d_angularVelocities[i] +
                                sd.d_radii[j] * sd.d_angularVelocities[j];
  const Vec3 v_tangent = cross(x_hat, rotational_speed);
  const Vec3 v_surface = v_delta - v_normal + v_tangent;
  const float v_surface_length = length(v_surface);
  Vec3 f_t{0.0f, 0.0f, 0.0f};
  if (v_surface_length > 1e-8f) {
    const float f_t_mag = fminf(sd.d_gamma_t[i] * v_surface_length, coulomb);
    const Vec3 v_hat_surface = -(v_surface / v_surface_length);
    f_t = v_hat_surface * f_t_mag;
  }

  fi += f_n + f_t;
  ti += cross(-ri * x_hat, f_t);
}
__host__ __device__ __forceinline__ void
sphereOnPlane(const size_t i, const int j, const SphereDevice sd,
              const PlaneDevice pd, Vec3 &fi, Vec3 &ti) {
  // TODO: implement sphere-plane contact force
}

#endif // FORCECOMPUTATIONS_CUH

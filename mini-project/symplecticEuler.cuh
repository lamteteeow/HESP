//
// Created by hans on 22.06.25.
//

#ifndef COMPUTESPRINGDASHPOTFORCES_CUH
#define COMPUTESPRINGDASHPOTFORCES_CUH
#include <cuda_runtime.h>

#include "contact.cuh"
#include "Vec3.cuh"
#include "Quaternion.cuh"
#include "PlaneDevice.cuh"
#include "SphereDevice.cuh"
#include "MeshDevice.cuh"

__device__ __forceinline__ void force(size_t i, size_t j, MeshDevice md, CollisionResult result, Vec3& fi, Vec3& ti)
{
    const size_t A_meshIdx = md.d_mesh_ids[i];
            const size_t A_start = md.d_meshes_offset[A_meshIdx];
            const size_t A_end = md.d_meshes_offset[A_meshIdx + 1];
            const size_t B_meshIdx = md.d_mesh_ids[j];
            const size_t B_start = md.d_meshes_offset[B_meshIdx];
            const size_t B_end = md.d_meshes_offset[B_meshIdx + 1];
            Quaternion oi = md.d_orientations[i];
            Quaternion oj = md.d_orientations[j];
            Vec3 pi = md.d_positions[i];
            Vec3 pj = md.d_positions[j];
            Vec3 vi = md.d_velocities[i];
            Vec3 vj = md.d_velocities[j];
            Vec3 avi = md.d_angularVelocities[i];
            Vec3 avj = md.d_angularVelocities[j];
            Vec3 outNormal = result.collisionNormal;
            Vec3 dir = pi - pj;
            if (dot(outNormal, dir) < 0.0f)
                outNormal = -outNormal;
            float outDepth = result.penetrationDepth;
            //fullEPA(i,j,md,simplex.pts, outNormal, outDepth);
            //simpleEPA(simplex.pts, outNormal, outDepth,i);

            Vec3 contactPointA = result.contactPointA;
            Vec3 contactPointB = result.contactPointB;
            Vec3 contactPoint = 0.5f * (contactPointA + contactPointB);
            //printf("[%lu]cpA:%f/%f/%f cpB:%f/%f/%f cp:%f/%f/%f\n", i,contactPointA.x,contactPointA.y,contactPointA.z,contactPointB.x,contactPointB.y,contactPointB.z, contactPoint.x, contactPoint.y, contactPoint.z);
            Vec3 rA = contactPointA - pi;
            Vec3 rB = contactPointB - pj;
            Vec3 velocityA_p = vi + cross(md.d_angularVelocities[i], rA);
            Vec3 velocityB_p = vj + cross(md.d_angularVelocities[j], rB);
            float v_rel_normal = dot(velocityA_p - velocityB_p, outNormal);
            //printf("[%lu]rA:%f/%f/%f rB:%f/%f/%f velA:%f/%f/%f velB:%f/%f/%f vrel:%f\n", i, rA.x, rA.y, rA.z,rB.x,rB.y,rB.z,velocityA_p.x,velocityA_p.y,velocityA_p.z, velocityB_p.x, velocityB_p.y, velocityB_p.z, v_rel_normal);
            //printf("[%lu]n:%f/%f/%f d:%f vrel:%f pos:%f/%f/%f\n", i, outNormal.x, outNormal.y, outNormal.z, outDepth,v_rel_normal, md.d_positions[i].x, md.d_positions[i].y, md.d_positions[i].z);
            float f_n_mag = md.d_kn[i] * outDepth - md.d_gamma_n[i] * v_rel_normal;
            Vec3 f_n = outNormal * f_n_mag;
            fi+= f_n;

            const float coulomb = md.d_mu[i] * fabs(f_n_mag);

            const Vec3 v_rel = velocityB_p - velocityA_p;
            const Vec3 v_normal = outNormal * dot(v_rel, outNormal);
            const Vec3 v_tangent = v_rel - v_normal + cross(avj, rB) - cross(avi, rA);
            const Vec3 v_surface = v_rel - v_normal + v_tangent;
            const float v_surface_length = length(v_surface);
            Vec3 f_t{0.0f, 0.0f, 0.0f};
            if (v_surface_length > 1e-8f)
            {
                const float f_t_mag = fminf(md.d_gamma_t[i] * v_surface_length, coulomb);
                Vec3 v_hat_surface = -(v_surface / v_surface_length);
                f_t = v_hat_surface * f_t_mag;
                fi += f_t;
            }
            //printf("[%lu]fn:%f/%f/%f fnmag:%f ft:%f/%f/%f\n", i, f_n.x, f_n.y, f_n.z, f_n_mag, f_t.x, f_t.y, f_t.z);

            ti += cross(rA, f_t);
}

__global__ inline void cfMeshOnMesh(const Vec3 grav,
                                    const MeshDevice md,
                                    int* d_neighborsOfCell)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= md.n) return;
    Vec3 fi = {0.0f, 0.0f, 0.0f};
    Vec3 ti = {0.0f, 0.0f, 0.0f};
    Vec3 out_fi = {0.0f, 0.0f, 0.0f};
    Vec3 out_ti = {0.0f, 0.0f, 0.0f};
    int base = md.d_cellIndexes[i] * 27;
    for (int nbr = 0; nbr < 27; ++nbr)
    {
        int cell = d_neighborsOfCell[base + nbr];
        if (cell < 0) continue;
        for (int j = md.d_cellHeads[cell]; j != -1; j = md.d_cellTails[j])
        {
            if (j == i) continue;

            /*Simplex simplex;
            if (!GJK(i, j, md, simplex))*/
            CollisionResult result = sat_collision_check(i,j,md);
            if (!result.collided)
            {
                //printf("[%lu]Daneben\n", i);
                continue;
            }
            //printf("[%lu]Treffer\n", i);
            /*printf("[%lu]A:%f/%f/%f B:%f/%f/%f C:%f/%f/%f D:%f/%f/%f\n", i,
                   simplex.pts[0].x, simplex.pts[0].y, simplex.pts[0].z,
                   simplex.pts[1].x, simplex.pts[1].y, simplex.pts[1].z,
                   simplex.pts[2].x, simplex.pts[2].y, simplex.pts[2].z,
                   simplex.pts[3].x, simplex.pts[3].y, simplex.pts[3].z);*/
            /*const size_t A_meshIdx = md.d_mesh_ids[i];
            const size_t A_start = md.d_meshes_offset[A_meshIdx];
            const size_t A_end = md.d_meshes_offset[A_meshIdx + 1];
            const size_t B_meshIdx = md.d_mesh_ids[j];
            const size_t B_start = md.d_meshes_offset[B_meshIdx];
            const size_t B_end = md.d_meshes_offset[B_meshIdx + 1];
            Quaternion oi = md.d_orientations[i];
            Quaternion oj = md.d_orientations[j];
            Vec3 pi = md.d_positions[i];
            Vec3 pj = md.d_positions[j];
            Vec3 vi = md.d_velocities[i];
            Vec3 vj = md.d_velocities[j];
            Vec3 avi = md.d_angularVelocities[i];
            Vec3 avj = md.d_angularVelocities[j];
            Vec3 outNormal = result.collisionNormal;
            Vec3 dir = pi - pj;
            if (dot(outNormal, dir) < 0.0f)
                outNormal = -outNormal;
            float outDepth = result.penetrationDepth;
            //fullEPA(i,j,md,simplex.pts, outNormal, outDepth);
            //simpleEPA(simplex.pts, outNormal, outDepth,i);
            //Vec3 contactPointA = pi + 0.5f * outDepth * outNormal;
            //Vec3 contactPointB = pj + 0.5f * outDepth * outNormal;
            Vec3 contactPointA = result.contactPointA;
            Vec3 contactPointB = result.contactPointB;
            Vec3 contactPoint = 0.5f * (contactPointA + contactPointB);
            //printf("[%lu]cpA:%f/%f/%f cpB:%f/%f/%f cp:%f/%f/%f\n", i,contactPointA.x,contactPointA.y,contactPointA.z,contactPointB.x,contactPointB.y,contactPointB.z, contactPoint.x, contactPoint.y, contactPoint.z);
            Vec3 rA = contactPointA - pi;
            Vec3 rB = contactPointB - pj;
            Vec3 velocityA_p = vi + cross(md.d_angularVelocities[i], rA);
            Vec3 velocityB_p = vj + cross(md.d_angularVelocities[j], rB);
            float v_rel_normal = dot(velocityA_p - velocityB_p, outNormal);
            //printf("[%lu]rA:%f/%f/%f rB:%f/%f/%f velA:%f/%f/%f velB:%f/%f/%f vrel:%f\n", i, rA.x, rA.y, rA.z,rB.x,rB.y,rB.z,velocityA_p.x,velocityA_p.y,velocityA_p.z, velocityB_p.x, velocityB_p.y, velocityB_p.z, v_rel_normal);
            //printf("[%lu]n:%f/%f/%f d:%f vrel:%f pos:%f/%f/%f\n", i, outNormal.x, outNormal.y, outNormal.z, outDepth,v_rel_normal, md.d_positions[i].x, md.d_positions[i].y, md.d_positions[i].z);
            float f_n_mag = md.d_kn[i] * outDepth - md.d_gamma_n[i] * v_rel_normal;
            Vec3 f_n = outNormal * f_n_mag;
            fi+= f_n;

            const float coulomb = md.d_mu[i] * fabs(f_n_mag);

            const Vec3 v_rel = velocityB_p - velocityA_p;
            const Vec3 v_normal = outNormal * dot(v_rel, outNormal);
            const Vec3 v_tangent = v_rel - v_normal + cross(avj, rB) - cross(avi, rA);
            const Vec3 v_surface = v_rel - v_normal + v_tangent;
            const float v_surface_length = length(v_surface);
            Vec3 f_t{0.0f, 0.0f, 0.0f};
            if (v_surface_length > 1e-8f)
            {
                const float f_t_mag = fminf(md.d_gamma_t[i] * v_surface_length, coulomb);
                Vec3 v_hat_surface = -(v_surface / v_surface_length);
                f_t = v_hat_surface * f_t_mag;
                fi += f_t;
            }
            //printf("[%lu]fn:%f/%f/%f fnmag:%f ft:%f/%f/%f\n", i, f_n.x, f_n.y, f_n.z, f_n_mag, f_t.x, f_t.y, f_t.z);

            ti += cross(rA, f_t);*/

            force(i,j,md,result, out_fi, out_ti);
            //printf("[%lu]fi:%f/%f/%f out_fi:%f/%f/%f\n", i, fi.x, fi.y, fi.z, out_fi.x, out_fi.y, out_fi.z);
            //printf("[%lu]ti:%f/%f/%f out_ti:%f/%f/%f\n", i, ti.x, ti.y, ti.z, out_ti.x, out_ti.y, out_ti.z);

        }
    }
    out_fi += grav * md.d_masses[i];

    md.d_forces[i] = out_fi;
    md.d_torques[i] = out_ti;
    //printf("[%lu]f:%f/%f/%f t:%f/%f/%f\n", i, md.d_forces[i].x, md.d_forces[i].y, md.d_forces[i].z, md.d_torques[i].x, md.d_torques[i].y, md.d_torques[i].z);
}

__global__ inline void cfMeshOnPlane(const MeshDevice md,
                                     const PlaneDevice pd)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= md.n) return;
    const Vec3 pi = md.d_positions[i];
    const Vec3 vi = md.d_velocities[i];

    Vec3 fi = {0.0f, 0.0f, 0.0f};
    Vec3 ti = {0.0f, 0.0f, 0.0f};

    for (int p = 0; p < pd.n; ++p)
    {
        const Vec3 planeNormal = pd.d_normals[p];
        const float planeD = pd.d_distances[p];
        // fetch body state
        Quaternion q = md.d_orientations[i];
        Vec3 pos = md.d_positions[i];
        Vec3 vel = md.d_velocities[i];
        Vec3 ang = md.d_angularVelocities[i];

        // find the deepest‐penetrating vertex
        float minDist = 1e30f;
        Vec3 deepest = {0, 0, 0};
        int start = md.d_meshes_offset[md.d_mesh_ids[i]];
        int end = md.d_meshes_offset[md.d_mesh_ids[i] + 1];
        for (int v = start; v < end; ++v)
        {
            // world‐space vertex
            Vec3 wv = q.rotate(md.d_vertices[v]) + pos;
            //printf("[%lu]v:%d wv: %f/%f/%f\n",i,v, wv.x,wv.y,wv.z);
            float dist = dot(planeNormal, wv) - planeD;
            //printf("[%lu]v:%d dist: %f\n",i,v, dist);
            if (dist < minDist)
            {
                minDist = dist;
                deepest = wv;
            }
        }

        if (minDist >= 0.0f) continue;
        printf("[%lu]minDist:%f\n deep:%f/%f/%f\n",i,minDist, deepest.x, deepest.y, deepest.z);
        // contact info
        float penetration = -minDist;
        Vec3 contactPoint = deepest - minDist * planeNormal;
        Vec3 rA = contactPoint - pos;

        // compute normal spring/dashpot
        Vec3 crossAngRa = cross(ang, rA);
        printf("[%lu]crossAngRa:%f/%f/%f ang:%f/%f/%f rA%f/%f/%f\n",i, crossAngRa.x, crossAngRa.y, crossAngRa.z, ang.x, ang.y, ang.z, rA.x,rA.y,rA.z);
        Vec3 vA_p = vel + cross(ang, rA);
        // plane is static, so v_plane = 0
        float v_rel_n = dot(/*vB_p*/Vec3{0, 0, 0} - vA_p, planeNormal);
        float fn_mag;
        if (penetration > 1e-4f)
            fn_mag = fmaxf(md.d_kn[i] * penetration - md.d_gamma_n[i] * v_rel_n, 0.0f);
        else
            fn_mag = fmaxf(md.d_kn[i] * penetration, 0.0f);  // keine Dämpfung bei sehr kleiner d
        fn_mag = fminf(fn_mag, 1e2);

        Vec3 f_n = planeNormal * fn_mag;
        Vec3 torque = cross(rA, f_n);
        if (length(torque) > 1e2)
            torque = torque * 1e2 / length(torque);
        ti += torque;
        // accumulate normal
        fi += f_n;
        ti += cross(rA, f_n);
        printf("[%lu]fn:%f/%f/%f fnmag:%f ft:%f/%f/%f\n", i, f_n.x, f_n.y, f_n.z, fn_mag, torque.x, torque.y, torque.z);

        // (optional) friction
        Vec3 v_rel = /*vB_p*/Vec3{0, 0, 0} - vA_p;
        Vec3 v_n_vec = planeNormal * dot(v_rel, planeNormal);
        Vec3 v_tan = v_rel - v_n_vec; // no spin on plane
        float v_tlen = length(v_tan);
        if (v_tlen > 1e-8f)
        {
            float f_t_mag = fminf(md.d_gamma_t[i] * v_tlen,
                                  md.d_mu[i] * fn_mag);
            Vec3 t_hat = v_tan / v_tlen;
            Vec3 f_t = -t_hat * f_t_mag;
            fi += f_t;
            ti += cross(rA, f_t);
        }
    }
    md.d_forces[i] += fi;
    md.d_torques[i] += ti;
}

__global__ inline void cfSphereOnSphere(const Vec3 grav,
                                        const SphereDevice sd,
                                        int* d_neighborsOfCell)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sd.n) return;
    const Vec3 pi = sd.d_positions[i];
    const Vec3 vi = sd.d_velocities[i];
    const float ri = sd.d_radii[i];

    Vec3 fi = {0.0f, 0.0f, 0.0f};
    Vec3 ti = {0.0f, 0.0f, 0.0f};

    const int cellIdx = sd.d_cellIndexes[i];
    const int base = cellIdx * 27;
    for (int k = 0; k < 27; ++k)
    {
        const int cellOrNeighbor = d_neighborsOfCell[base + k];
        if (cellOrNeighbor == -1) continue;
        for (int j = sd.d_cellHeads[cellOrNeighbor]; j != -1; j = sd.d_cellTails[j])
        {
            if (i == j) continue;
            const Vec3 pj = sd.d_positions[j];
            const Vec3 vj = sd.d_velocities[j];
            const Vec3 x_delta = pi - pj;
            const float distance = length(x_delta);
            const float overlap = (ri + sd.d_radii[j]) - distance;
            if (overlap <= 0.0f) continue;
            const Vec3 v_delta = vi - vj;
            const Vec3 x_hat = x_delta / distance;
            const float v_rel_normal = dot(v_delta, x_hat);
            const float f_n_mag = sd.d_kn[i] * overlap - sd.d_gamma_n[i] * v_rel_normal;
            Vec3 f_n = f_n_mag * x_hat;

            const float coulomb = sd.d_mu[i] * fabs(f_n_mag);

            const Vec3 v_normal = x_hat * v_rel_normal;
            const Vec3 rotational_speed = ri * sd.d_angularVelocities[i] + sd.d_radii[j] * sd.d_angularVelocities[j];
            const Vec3 v_tangent = cross(x_hat, rotational_speed);
            const Vec3 v_surface = v_delta - v_normal + v_tangent;
            const float v_surface_length = length(v_surface);
            Vec3 f_t{0.0f, 0.0f, 0.0f};
            if (v_surface_length > 1e-8f)
            {
                const float f_t_mag = fminf(sd.d_gamma_t[i] * v_surface_length, coulomb);
                Vec3 v_hat_surface = -(v_surface / v_surface_length);
                f_t = v_hat_surface * f_t_mag;
            }

            fi += f_n + f_t;
            ti += cross(-ri * x_hat, f_t);
        }
    }

    fi += grav * sd.d_masses[i];

    sd.d_forces[i] = fi;
    sd.d_torques[i] = ti;
}

__global__ inline void cfSphereOnPlane(const SphereDevice sd,
                                       const PlaneDevice pd)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sd.n) return;
    const Vec3 pi = sd.d_positions[i];
    const Vec3 vi = sd.d_velocities[i];
    const float ri = sd.d_radii[i];

    Vec3 f_i = {0.0f, 0.0f, 0.0f};
    Vec3 t_i = {0.0f, 0.0f, 0.0f};

    for (int p = 0; p < pd.n; ++p)
    {
        Vec3 x_hat = pd.d_normals[p];
        const float d = pd.d_distances[p];
        const float distance = dot(x_hat, pi) - d;
        const float overlap = ri - distance;
        if (overlap <= 0.0f) continue;
        Vec3 v_delta = vi;
        // Ab hier gleich?
        const float v_rel_normal = dot(v_delta, x_hat);
        const float f_n_mag = sd.d_kn[i] * overlap - sd.d_gamma_n[i] * v_rel_normal;
        Vec3 f_n = f_n_mag * x_hat;

        const float coulomb = sd.d_mu[i] * fabs(f_n_mag);

        const Vec3 v_normal = x_hat * v_rel_normal;
        const Vec3 rotational_speed = ri * sd.d_angularVelocities[i];
        const Vec3 v_tangent = cross(x_hat, rotational_speed);
        const Vec3 v_surface = v_delta - v_normal + v_tangent;
        const float v_surface_length = length(v_surface);

        Vec3 f_t{0.0f, 0.0f, 0.0f};
        if (v_surface_length > 1e-8f)
        {
            const float f_t_mag = fminf(sd.d_gamma_t[i] * v_surface_length, coulomb);
            Vec3 v_hat_surface = -(v_surface / v_surface_length);
            f_t = v_hat_surface * f_t_mag;
        }
        f_i += f_n + f_t;
        t_i += cross(-ri * x_hat, f_t);

        //protection against spheres sinking into the plane (Baumgarte bias)
        /*const float slop = 0.01f * ri;
        const float beta = 0.2f;
        float corr = fmaxf(overlap - slop, 0.0f) * beta;
        sd.d_positions[i] += x_hat * corr;*/
    }
    sd.d_forces[i] += f_i;
    sd.d_torques[i] += t_i;
}

// rotate vector v by unit quaternion q
__device__ inline Vec3 rotate(const Quaternion& q, const Vec3& v)
{
    // assumes q = (w, x, y, z), v treated as pure‐imaginary quaternion
    // optimized form: v' = 2*(u·v)*u + (w*w - u·u)*v + 2*w*(u×v)
    Vec3 u = {q.x, q.y, q.z};
    float s = q.w;
    float uu = dot(u, u);
    float uv = dot(u, v);
    Vec3 term1 = u * (2.0f * uv);
    Vec3 term2 = v * (s * s - uu);
    Vec3 term3 = cross(u, v) * (2.0f * s);
    return term1 + term2 + term3;
}


__global__ inline void IntegrateVelAndPos(const float dt,
                                          const SphereDevice sd)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sd.n) return;
    sd.d_velocities[i] += dt * sd.d_forces[i] / sd.d_masses[i];

    sd.d_angularVelocities[i] += dt * sd.d_torques[i] / sd.d_inertia[i];

    sd.d_positions[i] += dt * sd.d_velocities[i];

    Quaternion q = sd.d_orientations[i];
    q += 0.5f * Quaternion(sd.d_angularVelocities[i]) * q * dt;
    sd.d_orientations[i] = q.normalize();
}

__global__ inline void IntegrateVelAndPosMesh(const float dt, MeshDevice md)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= md.n) return;

    md.d_velocities[i] += dt * md.d_forces[i] * (1.0f / md.d_masses[i]);

    Quaternion q = md.d_orientations[i];
    const Vec3 torque_local = conjugate(q).rotate(md.d_torques[i]);
    const Vec3 angVel_local = torque_local * md.d_inertia_inv[i];
    const Vec3 angVel_world = q.rotate(angVel_local);
    md.d_angularVelocities[i] += dt * angVel_world;

    md.d_positions[i] += dt * md.d_velocities[i];
    const Quaternion qDot = Quaternion(md.d_angularVelocities[i]) * q * 0.5f;
    q += qDot * dt;
    md.d_orientations[i] = q.normalize();
}

__device__ inline void boundaryCollision(const int i,
                                         Vec3* velocities,
                                         Vec3* positions,
                                         const Vec3 MIN,
                                         const Vec3 MAX,
                                         const float* radii)
{
    const float radius = radii[i];
    if (positions[i].x - radius < MIN.x)
    {
        positions[i].x = MIN.x + radius;
        velocities[i].x = -velocities[i].x;
    }
    else if (positions[i].x + radius > MAX.x)
    {
        positions[i].x = MAX.x - radius;
        velocities[i].x = -velocities[i].x;
    }
    if (positions[i].y - radius < MIN.y)
    {
        positions[i].y = MIN.y + radius;
        velocities[i].y = -velocities[i].y;
    }
    else if (positions[i].y + radius > MAX.y)
    {
        positions[i].y = MAX.y - radius;
        velocities[i].y = -velocities[i].y;
    }
    if (positions[i].z - radius < MIN.z)
    {
        positions[i].z = MIN.z + radius;
        velocities[i].z = -velocities[i].z;
    }
    else if (positions[i].z + radius > MAX.z)
    {
        positions[i].z = MAX.z - radius;
        velocities[i].z = -velocities[i].z;
    }
}

#endif //COMPUTESPRINGDASHPOTFORCES_CUH

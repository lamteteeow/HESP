//
// Created by hans on 18.07.25.
//

#ifndef CONTACT_CUH
#define CONTACT_CUH
#include "MeshDevice.cuh"
#include "simplex.cuh"
#include "Vec3.cuh"


__device__ __forceinline__ Vec3 supportFaceCenter(
    const Quaternion& q, const Vec3& pos,
    const Vec3& d, const Vec3* verts, int start, int end
)
{
    Vec3 dloc = conjugate(q).rotate(d);
    float best = -1e30f;
    // find max dot
    for (int i = start; i < end; ++i)
    {
        best = fmaxf(best, dot(verts[i], dloc));
    }
    // average those that tie for best
    Vec3 avg{0, 0, 0};
    int cnt = 0;
    const float eps = 1e-3f;
    for (int i = start; i < end; ++i)
    {
        if (fabsf(dot(verts[i], dloc) - best) < eps)
        {
            avg += verts[i];
            ++cnt;
        }
    }
    avg = avg * (1.0f / cnt);
    return q.rotate(avg) + pos;
}

__device__ __forceinline__ Vec3 support(const Vec3 d,
                                        const Vec3* vertices,
                                        const size_t start,
                                        const size_t end)
{
    Vec3 best = vertices[start];
    float maxDot = dot(best, d);
    for (size_t i = start + 1; i < end; ++i)
    {
        Vec3 current = vertices[i];
        const float dotProd = dot(current, d);
        if (dotProd > maxDot)
        {
            //printf("max:%f dot:%f best:%f/%f/%f curr:%f/%f/%f\n",maxDot, dotProd, best.x, best.y, best.z, current.x, current.y, current.z);
            maxDot = dotProd;
            best = current;
        }
    }
    return best;
}

__device__ __forceinline__ Vec3 support_WL(const Quaternion q,
                                           const Vec3 position,
                                           const Vec3 d,
                                           const Vec3* vertices,
                                           const size_t start,
                                           const size_t end)
{
    const Vec3 d_local = conjugate(q).rotate(d);
    const Vec3 max_local = support(d_local, vertices, start, end);
    return q.rotate(max_local) + position;
}

__device__ __forceinline__ Vec3 minkowski_difference_WL(const size_t A_idx,
                                                        const size_t B_idx,
                                                        const MeshDevice md,
                                                        const Vec3 d,
                                                        const size_t A_start,
                                                        const size_t A_end,
                                                        const size_t B_start,
                                                        const size_t B_end)
{
    const Vec3 A = support_WL(md.d_orientations[A_idx], md.d_positions[A_idx], d, md.d_vertices, A_start, A_end);
    const Vec3 B = support_WL(md.d_orientations[B_idx], md.d_positions[B_idx], -d, md.d_vertices, B_start, B_end);
    return A - B;
}

__device__ __forceinline__
Vec3 supportMinkowski(const size_t A,
                      const size_t B,
                      MeshDevice md,
                      const Vec3& d)
{
    // lookup the per-mesh vertex ranges
    const size_t A_meshIdx = md.d_mesh_ids[A];
    const size_t A_start = md.d_meshes_offset[A_meshIdx];
    const size_t A_end = md.d_meshes_offset[A_meshIdx + 1];
    const size_t B_meshIdx = md.d_mesh_ids[B];
    const size_t B_start = md.d_meshes_offset[B_meshIdx];
    const size_t B_end = md.d_meshes_offset[B_meshIdx + 1];

    return minkowski_difference_WL(
        A, B, md,
        d,
        A_start, A_end,
        B_start, B_end
    );
}

__device__ __forceinline__ bool handleLine(Simplex& simplex, Vec3& direction, const size_t idx)
{
    const Vec3 A = simplex[0];
    const Vec3 B = simplex[1];
    const Vec3 AO = -A;
    const Vec3 AB = B - A;
    const bool dotABAO = dot(AB, AO) > 0;
    //printf("[%lu] dotABAO fs1: %d\n", idx, dotABAO);
    if (dotABAO)
        direction = cross(cross(AB, AO), AB);
    else
    {
        simplex.size = 1;
        direction = AO;
    }
    return false;
}

__device__ __forceinline__ bool handleTriangle(Simplex& simplex, Vec3& direction, const size_t idx)
{
    const Vec3 A = simplex.pts[0];
    const Vec3 B = simplex.pts[1];
    const Vec3 C = simplex.pts[2];

    const Vec3 AB = B - A;
    const Vec3 AC = C - A;
    const Vec3 AO = -A;

    // face normal (not normalized)
    const Vec3 ABC = cross(AB, AC);

    // Region outside AB?
    // normal to AB toward C-side
    const Vec3 abPerp = cross(AB, ABC);
    const bool dotABperpAO = dot(abPerp, AO) > 0;
    //printf("[%lu] dotABperpAO ts2: %d\n", idx, dotABperpAO);
    if (dotABperpAO)
    {
        // keep AB
        simplex.size = 2;
        // new dir = orthogonal to AB toward origin
        direction = cross(cross(AB, AO), AB);
        return false;
    }
    // Region outside AC?
    // normal to AC toward B-side
    const Vec3 acPerp = cross(ABC, AC);
    const bool dotACperpAO = dot(acPerp, AO) > 0;
    //printf("[%lu] dotACperpAO ts2: %d\n", idx, dotACperpAO);
    if (dotACperpAO)
    {
        // keep AC
        simplex.pts[1] = C;
        simplex.size = 2;
        direction = cross(cross(AC, AO), AC);
        return false;
    }
    // Origin is within “wedge” above triangle face
    // Keep ABC.  But ensure normal points toward origin.
    if (dot(ABC, AO) > 0)
    {
        // normal is already pointing out of the face toward origin
        direction = ABC;
    }
    else
    {
        // flip winding: swap B/C so that normal = cross(AC,AB)
        simplex.pts[1] = C;
        simplex.pts[2] = B;
        direction = -ABC;
    }
    // keep all three points unchanged
    return false;
}

__device__ __forceinline__ bool handleTetrahedron(Simplex& simplex, Vec3 direction, const size_t idx)
{
    // s.size == 4, points A=pts[0], B=pts[1], C=pts[2], D=pts[3]
    const Vec3 A = simplex.pts[0];
    const Vec3 B = simplex.pts[1];
    const Vec3 C = simplex.pts[2];
    const Vec3 D = simplex.pts[3];
    const Vec3 AO = -A;

    // helper to test a face and reduce to triangle case if origin is outside
    auto testFace = [&](Vec3 U, Vec3 V, Vec3 W) -> bool
    {
        // U,V,W are the triangle’s corners in local simplex order: [A,U,V]
        Vec3 UV = V - U;
        Vec3 UW = W - U;
        Vec3 normal = cross(UV, UW);
        // make sure normal points out of the tetra (away from the opposite point)
        Vec3 againstD = (U == A ? D - A : A - U);
        if (dot(normal, againstD) > 0) normal = -normal;

        // if origin is outside this face:
        bool dotNORMALAO = dot(normal, AO) > 0;
        //printf("[%lu] dotNORMALAO ts3: %d\n", idx, dotNORMALAO);
        if (dotNORMALAO)
        {
            // reduce simplex to that face:
            simplex.size = 3;
            simplex.pts[0] = A;
            simplex.pts[1] = U;
            simplex.pts[2] = V;
            // call triangle handler to set direction & possibly reorder
            return handleTriangle(simplex, direction, idx);
        }
        return false;
    };

    // test each face that includes A: ABC, ACD, ADB
    bool test = testFace(B, C, D);
    //printf("[%lu] testBCD rf: %d\n", idx, test);
    if (test) return false; // origin outside ABC
    test = testFace(C, D, B);
    //printf("[%lu] testCDB rf: %d\n", idx, test);
    if (test) return false; // origin outside ACD
    test = testFace(D, B, C);
    //printf("[%lu] testDBC rf: %d\n", idx, test);
    if (test) return false; // origin outside ADB

    // If we get here, origin is inside all three faces → inside tetrahedron
    return true;
}

__device__ __forceinline__ bool containsOrigin(Simplex& simplex, Vec3& direction, const size_t idx)
{
    switch (simplex.size)
    {
    case 2: return handleLine(simplex, direction, idx);
    case 3: return handleTriangle(simplex, direction, idx);
    case 4: return handleTetrahedron(simplex, direction, idx);
    default: // should never happen
        direction = Vec3{0, 0, 0};
        return false;
    }
}

__device__ __forceinline__ bool GJK(const size_t A_idx, const size_t B_idx, MeshDevice md, Simplex& simplex)
{
    simplex.init();
    const size_t A_meshIdx = md.d_mesh_ids[A_idx];
    const size_t A_start = md.d_meshes_offset[A_meshIdx];
    const size_t A_end = md.d_meshes_offset[A_meshIdx + 1];
    const size_t B_meshIdx = md.d_mesh_ids[B_idx];
    const size_t B_start = md.d_meshes_offset[B_meshIdx];
    const size_t B_end = md.d_meshes_offset[B_meshIdx + 1];
    Vec3 direction = {1.0f,2.0f,3.0f};
    //Vec3 direction = md.d_positions[A_idx] - md.d_positions[B_idx];
    //printf("[%lu] set dir: %f/%f/%f\n", A_idx, direction.x, direction.y, direction.z);
    if (direction == Vec3{0.0f, 0.0f, 0.0f}) direction = Vec3{1.0f,2.0f,3.0f};  ;
    //printf("[%lu] reset dir: %f/%f/%f\n", A_idx, direction.x, direction.y, direction.z);
    const Vec3 s_0 = supportMinkowski(A_idx, B_idx, md, direction);
    //const Vec3 s_0 = minkowski_difference_WL(A_idx, B_idx, md, direction, A_start, A_end, B_start, B_end);
    //printf("[%lu]: support: %f/%f/%f\n", A_idx, s_0.x, s_0.y, s_0.z);
    simplex.push(s_0);
    direction = -s_0;
    for (size_t i = 0; i < 50; i++)
    {
        //printf("[%lu] iter:%lu, size:%d\n", A_idx, i, simplex.size);
        const Vec3 s = supportMinkowski(A_idx, B_idx, md, direction);
        //const Vec3 s = minkowski_difference_WL(A_idx, B_idx, md, direction, A_start, A_end, B_start, B_end);
        //printf("[%lu]: support0: %f/%f/%f\n", A_idx, s.x, s.y, s.z);
        const float dotPD = dot(s, direction);
        //printf("[%lu] dotPD rf: %d\n", A_idx, dotPD <= 0.0f);
        if (dotPD <= 0.0f) return false;
        simplex.push(s);
        const bool contains = containsOrigin(simplex, direction, A_idx);
        //printf("[%lu] contains rt: %d\n", A_idx, contains);
        if (contains)
            return true;
        //printf("[%lu] dir: %f/%f/%f\n", A_idx, direction.x, direction.y, direction.z);
    }
    //printf("[%lu] end rf\n", A_idx);
    return false;
}

__device__ __forceinline__ size_t getDuplicateIndex(const Vec3 simplexPts[4])
{
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = i + 1; j < 4; j++)
        {
            if (i == j)
                continue;
            if (simplexPts[i] == simplexPts[j])
                return i;
        }
    }
    return -1;
}

__device__ __forceinline__ void simpleEPA(const Vec3 simplexPts[4],
                                          Vec3& outNormal,
                                          float& outDepth,
                                          size_t threadIdx)
{
    Vec3 unique[4];
    int  uniqN = 0;
    constexpr float DUP_EPS = 1e-6f;
    for (int i = 0; i < 4; ++i) {
        bool isDup = false;
        for (int j = 0; j < uniqN; ++j) {
            if (dot((unique[j] - simplexPts[i]),(unique[j] - simplexPts[i])) < DUP_EPS*DUP_EPS) {
                isDup = true; break;
            }
        }
        if (!isDup) unique[uniqN++] = simplexPts[i];
    }

    // 2) SPEZIALFALL: genau 3 verschieden Punkte → Dreieck
    if (uniqN == 3) {
        Vec3 p0 = unique[0],
             p1 = unique[1],
             p2 = unique[2];
        // rohe Normalen-Berechnung
        Vec3 n = cross(p1 - p0, p2 - p0);
        // Normieren (und sicherstellen, dass wir keinen Null-Vektor haben)
        n = normalize(n);

        // Ausrichtung: raus aus dem Dreieck, also weg vom Ursprung
        // (Zentroid ist hier nicht zwingend nötig, du kannst auch dot(n,p0) verwenden):
        if (dot(n, p0) < 0.0f)
            n = -n;

        // Tiefe = Abstand der Ebene zum Ursprung
        float depth = dot(n, p0);

        outNormal = n;
        outDepth  = depth;
        return;  // fertig, wir brauchen keine testFace-Schleife mehr
    }

    const Vec3 A = simplexPts[0];
    const Vec3 B = simplexPts[1];
    const Vec3 C = simplexPts[2];
    const Vec3 D = simplexPts[3];
    printf("[%lu]A:%f/%f/%f B:%f/%f/%f C:%f/%f/%f D:%f/%f/%f\n", threadIdx,
                   simplexPts[0].x, simplexPts[0].y, simplexPts[0].z,
                   simplexPts[1].x, simplexPts[1].y, simplexPts[1].z,
                   simplexPts[2].x, simplexPts[2].y, simplexPts[2].z,
                   simplexPts[3].x, simplexPts[3].y, simplexPts[3].z);
    float bestDist = INFINITY;
    Vec3 bestNormal = {0.0f, 0.0f, 0.0f};
    constexpr float EPS = 1e-6f;
    auto testFace = [&](const Vec3 a, const Vec3 b, const Vec3 c)
    {
        printf("[%lu] EPA a:%f/%f/%f b:%f/%f/%f c:%f/%f/%f\n", threadIdx, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
        // Compute outward unit-normal
        Vec3 n = cross(b - a, c - a);

        // avoid zero-normals from tiny faces with aligned or identical points
        bool dotZero = dot(n, n) < EPS * EPS;
        printf("[%lu] EPA dotZero tr: %d\n", threadIdx, dotZero);
        if (dotZero) return;
        n = normalize(n);
        // Ensure it points outward
        bool reverse = dot(n, a) < 0.0f;
        printf("[%lu] EPA reverse: %d\n", threadIdx, reverse);
        if (reverse) n = -n;
        printf("[%lu] EPA n:%f/%f/%f\n", threadIdx, n.x, n.y, n.z);
        // Distance from origin to the plane of the face
        const float d = dot(n, a);
        printf("[%lu]d:%f n:+%f/%f/%f\n", threadIdx, d, n.x, n.y, n.z);

        // set if minimal positive distance
        bool newBest = d < bestDist;
        printf("[%lu] EPA newBest: %d\n", threadIdx, newBest);
        if (newBest)
        {
            bestDist = d;
            bestNormal = n;
        }
    };

    // Test all four faces
    testFace(A, C, B);
    testFace(A, B, D);
    testFace(A, D, C);
    testFace(B, C, D);

    outNormal = bestNormal;
    outDepth = bestDist;
}

struct CollisionResult {
    bool collided;
    float penetrationDepth;
    Vec3 collisionNormal;
    Vec3 mtv;
    Vec3 contactPointA;  // heuristisch
    Vec3 contactPointB;  // heuristisch
};

__device__ __forceinline__ Vec3 transform_vertex(const Vec3& v_local, const Vec3& scale, const Quaternion& q, const Vec3& pos) {
    Vec3 scaled = {v_local.x * scale.x, v_local.y * scale.y, v_local.z * scale.z};
    Vec3 rotated = q.rotate(scaled);  // du brauchst eine Quaternion::rotate(Vec3) Methode
    return rotated + pos;
}


__device__ __forceinline__ CollisionResult sat_collision_check(int A_idx, int B_idx, const MeshDevice& md)
{
    CollisionResult result;
    result.collided = false;
    result.penetrationDepth = INFINITY;

    // Daten für A
    int A_meshIdx = md.d_mesh_ids[A_idx];
    int A_start = md.d_meshes_offset[A_meshIdx];
    int A_end = md.d_meshes_offset[A_meshIdx + 1];
    Quaternion qA = md.d_orientations[A_idx];
    Vec3 pA = md.d_positions[A_idx];
    Vec3 sA = md.d_scales[A_idx];

    // Daten für B
    int B_meshIdx = md.d_mesh_ids[B_idx];
    int B_start = md.d_meshes_offset[B_meshIdx];
    int B_end = md.d_meshes_offset[B_meshIdx + 1];
    Quaternion qB = md.d_orientations[B_idx];
    Vec3 pB = md.d_positions[B_idx];
    Vec3 sB = md.d_scales[B_idx];

    // Durchlaufe alle Flächen von A und B, erzeugen Trennachsen
    for (int phase = 0; phase < 2; ++phase) {
        int meshIdx = (phase == 0) ? A_meshIdx : B_meshIdx;
        int start = (phase == 0) ? A_start : B_start;
        int end = (phase == 0) ? A_end : B_end;
        Quaternion q = (phase == 0) ? qA : qB;
        Vec3 pos = (phase == 0) ? pA : pB;
        Vec3 scale = (phase == 0) ? sA : sB;

        for (int i = md.d_faces_offset[meshIdx]; i < md.d_faces_offset[meshIdx + 1]; ++i) {
            int3 f = md.d_faces[i];
            Vec3 v0 = transform_vertex(md.d_vertices[start + f.x], scale, q, pos);
            Vec3 v1 = transform_vertex(md.d_vertices[start + f.y], scale, q, pos);
            Vec3 v2 = transform_vertex(md.d_vertices[start + f.z], scale, q, pos);

            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 axis = normalize(cross(edge1, edge2));  // Flächennormale

            // Projiziere beide Objekte auf die Achse
            float minA = INFINITY, maxA = -INFINITY;
            for (int j = A_start; j < A_end; ++j) {
                Vec3 pt = transform_vertex(md.d_vertices[j], sA, qA, pA);
                float proj = dot(pt, axis);
                minA = fminf(minA, proj);
                maxA = fmaxf(maxA, proj);
            }

            float minB = INFINITY, maxB = -INFINITY;
            for (int j = B_start; j < B_end; ++j) {
                Vec3 pt = transform_vertex(md.d_vertices[j], sB, qB, pB);
                float proj = dot(pt, axis);
                minB = fminf(minB, proj);
                maxB = fmaxf(maxB, proj);
            }

            float overlap = fminf(maxA, maxB) - fmaxf(minA, minB);
            if (overlap <= 0.0f)
                return {false};  // Trennung gefunden

            if (overlap < result.penetrationDepth) {
                result.penetrationDepth = overlap;
                result.collisionNormal = axis;
            }
        }
    }

    // Richtung prüfen
    Vec3 centerA = md.d_positions[A_idx];
    Vec3 centerB = md.d_positions[B_idx];
    Vec3 dir = centerB - centerA;
    if (dot(dir, result.collisionNormal) < 0.0f)
        result.collisionNormal = result.collisionNormal * -1.0f;

    result.mtv = result.collisionNormal * result.penetrationDepth;
    result.collided = true;

    // Heuristischer Kontaktpunkt: Mittelpunkt der minimal überlappenden Achse
    result.contactPointA = centerA + 0.5f * result.mtv;
    result.contactPointB = centerB - 0.5f * result.mtv;

    return result;
}


// at top, shrink it slightly
constexpr int EPA_MAX_FACES      = 4 + 3*10;   // only need ~10 iters for cubes
constexpr int EPA_MAX_EDGES      = EPA_MAX_FACES * 3;
constexpr int EPA_MAX_ITERS      = 10;         // fewer iters for simple polyhedra
constexpr float EPA_EPS          = 1e-4f;

struct EPAFace {
    Vec3  a,b,c;
    Vec3  normal;
    float dist;
    bool  obsolete;
};

__device__ __forceinline__ EPAFace makeFace(
    const Vec3 &a, const Vec3 &b, const Vec3 &c
){
    EPAFace f;
    f.a = a; f.b = b; f.c = c;
    Vec3 n = normalize(cross(b-a, c-a));
    if (dot(n,a) < 0.0f) n = -n;
    f.normal   = n;
    f.dist     = dot(n, a);
    f.obsolete = false;
    return f;
}

// now pass md by const& so no giant struct copy:
__device__ __forceinline__ Vec3
mink_diff(const size_t A, const size_t B,
          const MeshDevice md, const Vec3 &d)
{
    size_t A_mid = md.d_mesh_ids[A],
           B_mid = md.d_mesh_ids[B];
    size_t A0 = md.d_meshes_offset[A_mid],
           A1 = md.d_meshes_offset[A_mid+1];
    size_t B0 = md.d_meshes_offset[B_mid],
           B1 = md.d_meshes_offset[B_mid+1];
    Vec3 Aw = support_WL(md.d_orientations[A],
                        md.d_positions[A],
                        d, md.d_vertices, A0, A1);
    Vec3 Bw = support_WL(md.d_orientations[B],
                        md.d_positions[B],
                        -d, md.d_vertices, B0, B1);
    return Aw - Bw;
}

__device__ __forceinline__
void fullEPA(
    const size_t A,
    const size_t B,
    const MeshDevice &md,
    const Vec3 simplexPts[4],
    Vec3&      outNormal,
    float&     outDepth
){
    EPAFace faces[EPA_MAX_FACES];
    int     nFaces = 0;

    // init tetra
    faces[nFaces++] = makeFace(simplexPts[0], simplexPts[2], simplexPts[1]);
    faces[nFaces++] = makeFace(simplexPts[0], simplexPts[1], simplexPts[3]);
    faces[nFaces++] = makeFace(simplexPts[0], simplexPts[3], simplexPts[2]);
    faces[nFaces++] = makeFace(simplexPts[1], simplexPts[2], simplexPts[3]);

    // temp horizon
    struct Edge { Vec3 u,v; };
    Edge horizon[EPA_MAX_EDGES];

    for(int iter=0; iter<EPA_MAX_ITERS; ++iter){
        // find closest live face
        int   bestI = -1;
        float bestD = 1e30f;
        for(int i=0;i<nFaces;++i){
            if(faces[i].obsolete) continue;
            if(faces[i].dist < bestD){
                bestD = faces[i].dist;
                bestI = i;
            }
        }
        auto &f = faces[bestI];

        // new support
        Vec3 p = mink_diff(A,B,md,f.normal);
        float d = dot(f.normal, p);
        if (d - f.dist < EPA_EPS){
            outNormal = f.normal;
            outDepth  = f.dist;
            return;
        }

        // mark visible
        for(int i=0;i<nFaces;++i){
            if(!faces[i].obsolete &&
               dot(faces[i].normal, p - faces[i].a) > 0.0f)
                faces[i].obsolete = true;
        }
        // build horizon edges
        int hCount = 0;
        auto isShared = [&](const Edge &e){
            for(int j=0;j<nFaces;++j) if(faces[j].obsolete){
                Vec3 arr[3] = { faces[j].a, faces[j].b, faces[j].c };
                for(int k=0;k<3;++k){
                    if(e.u == arr[(k+1)%3] && e.v == arr[k])
                        return true;
                }
            }
            return false;
        };
        for(int i=0;i<nFaces;++i) if(faces[i].obsolete){
            Vec3 arr[3] = { faces[i].a, faces[i].b, faces[i].c };
            for(int e=0;e<3;++e){
                Edge E{ arr[e], arr[(e+1)%3] };
                if(!isShared(E) && hCount < EPA_MAX_EDGES)
                    horizon[hCount++] = E;
            }
        }

        // compact out dead faces
        int write=0;
        for(int i=0;i<nFaces;++i){
            if(!faces[i].obsolete) faces[write++] = faces[i];
        }
        nFaces = write;

        // **cap** before adding
        if(nFaces + hCount > EPA_MAX_FACES) break;

        // add new faces
        for(int e=0;e<hCount;++e){
            faces[nFaces++] = makeFace(horizon[e].u,
                                       horizon[e].v,
                                       p);
        }
    }

    // fallback
    int bestI = 0;
    for(int i=1;i<nFaces;++i){
        if(faces[i].dist < faces[bestI].dist)
            bestI = i;
    }
    outNormal = faces[bestI].normal;
    outDepth  = faces[bestI].dist;
}


#endif //CONTACT_CUH

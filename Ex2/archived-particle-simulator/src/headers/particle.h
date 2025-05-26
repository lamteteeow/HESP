class Particle {
public:
    // Properties
    float3 position;  // 3D position of the particle
    float3 velocity;  // 3D velocity of the particle
    float mass;       // Mass of the particle
    float3 acceleration; // 3D acceleration of the particle

    // Constructor
    Particle() 
        : position(make_float3(0.0f, 0.0f, 0.0f)), velocity(make_float3(0.0f, 0.0f, 0.0f)), 
          mass(1.0f), acceleration(make_float3(0.0f, 0.0f, 0.0f)) {}
    
    Particle(float3 pos, float3 vel, float m) 
        : position(pos), velocity(vel), mass(m), acceleration(make_float3(0.0f, 0.0f, 0.0f)) {}

    // Method to update the particle's position and velocity
    void update(float dt) {
        // Update position using current velocity
        position = position + make_float3(velocity.x * dt, velocity.y * dt, velocity.z * dt);
        // Update velocity using current acceleration
        velocity = velocity + make_float3(acceleration.x * dt, acceleration.y * dt, acceleration.z * dt);
    }

    // Method to reset acceleration
    void resetAcceleration() {
        acceleration = make_float3(0.0f, 0.0f, 0.0f);
    }

    // Method to apply force to the particle
    void applyForce(float3 force) {
        acceleration = acceleration + make_float3(force.x * mass, force.y * mass, force.z * mass); // F = ma => a = F/m
    }

};

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
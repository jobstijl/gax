//! The world on the CPU: ships flying on constant-rate motors, and bursts of particles handed
//! to the GPU. A few particles are mirrored on the CPU with the same traced kernel, so the two
//! can be compared (`Scene::check`).

use crate::gfx::{Gfx, Instance, MAX_PARTICLES, Particle};
use gax::pga2d::{Motor, Point};

/// A small xorshift generator (no dependencies).
pub struct Rng(u64);

impl Rng {
    /// Seeded.
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

/// A rate `B` with `exp(dt B)` moving forward at `speed` and turning at `spin` (radians per
/// second): `B = (speed e20 + spin e12) / 2` in PGA2D (see `Motor::translation`).
pub fn rate(speed: f32, spin: f32) -> Point<(), f32> {
    Point::from_coeffs([speed * 0.5, 0.0, spin * 0.5])
}

struct Ship {
    motor: Motor<(), f32>,
    rate: Point<(), f32>,
    color: [f32; 4],
    hue: f32,
}

/// A particle followed on the CPU too.
struct Probe {
    index: usize,
    cpu: Motor<(), f32>,
    rate: Point<(), f32>,
    age: f32,
    lifetime: f32,
}

/// The world.
pub struct Scene {
    ships: Vec<Ship>,
    rng: Rng,
    until_burst: f32,
    head: usize,
    spawned: usize,
    probes: Vec<Probe>,
    /// Particles per burst.
    pub burst: usize,
    /// The largest difference between a probe's GPU and CPU motor seen so far.
    pub worst: f32,
    /// Probes compared so far.
    pub compared: usize,
}

/// Half the width of the view, in world units.
pub const HALF_WIDTH: f32 = 12.0;

impl Scene {
    /// `ships` ships, and bursts of `burst` particles.
    pub fn new(ships: usize, burst: usize) -> Scene {
        let mut rng = Rng::new(0x5eed);
        let ships = (0..ships)
            .map(|k| {
                let hue = k as f32 / ships as f32;
                let start = Motor::translation(rng.range(-8.0, 8.0), rng.range(-5.0, 5.0))
                    .into_inner()
                    * Motor::rotation(Point::xy(0.0, 0.0), rng.range(0.0, std::f32::consts::TAU)).into_inner();
                let c = hsv(hue);
                Ship {
                    motor: start,
                    rate: rate(rng.range(1.5, 4.0), rng.range(-1.2, 1.2)),
                    color: [c[0], c[1], c[2], 1.0],
                    hue,
                }
            })
            .collect();
        Scene {
            ships,
            rng,
            until_burst: 0.0,
            head: 0,
            spawned: 0,
            probes: Vec::new(),
            burst,
            worst: 0.0,
            compared: 0,
        }
    }

    /// Particles to draw.
    pub fn particles(&self) -> usize {
        self.spawned.min(MAX_PARTICLES)
    }

    /// Advance by `dt`: ships on the CPU (the traced kernel's Rust form), bursts written to the
    /// GPU pool, the GPU step recorded into `enc`, and the probes stepped on the CPU.
    pub fn update(&mut self, gfx: &mut Gfx, enc: &mut wgpu::CommandEncoder, dt: f32) {
        for s in &mut self.ships {
            s.motor = crate::particle_step(s.motor, s.rate, dt);
            // Keep ships near the view: wrap by moving the motor's translation.
            let p = (gax::Unit::new_unchecked(s.motor) >> Point::xy(0.0, 0.0)).to_euclidean();
            let (w, h) = (HALF_WIDTH + 1.0, HALF_WIDTH * 0.6 + 1.0);
            let (dx, dy) = (wrap(p[0], w) - p[0], wrap(p[1], h) - p[1]);
            if dx != 0.0 || dy != 0.0 {
                s.motor = Motor::translation(dx, dy).into_inner() * s.motor;
            }
        }
        let instances: Vec<Instance> = self
            .ships
            .iter()
            .map(|s| Instance {
                motor: s.motor.into(),
                color: s.color,
            })
            .collect();
        gfx.set_instances(&instances);

        self.until_burst -= dt;
        if self.until_burst <= 0.0 && !self.ships.is_empty() {
            self.until_burst = 0.35;
            let k =
                (self.rng.range(0.0, 1.0) * self.ships.len() as f32) as usize % self.ships.len();
            self.spawn(gfx, k);
        }
        gfx.step_particles(enc, dt, self.particles());
        for p in &mut self.probes {
            p.cpu = crate::particle_step(p.cpu, p.rate, dt);
            p.age += dt;
        }
    }

    fn spawn(&mut self, gfx: &Gfx, ship: usize) {
        let (origin, hue) = (self.ships[ship].motor, self.ships[ship].hue);
        let ps: Vec<Particle> = (0..self.burst)
            .map(|_| {
                let turn =
                    Motor::rotation(Point::xy(0.0, 0.0), self.rng.range(0.0, std::f32::consts::TAU)).into_inner();
                Particle {
                    motor: (origin * turn).into(),
                    rate: rate(self.rng.range(0.5, 6.0), self.rng.range(-6.0, 6.0)).into(),
                    life: [
                        0.0,
                        self.rng.range(0.6, 2.2),
                        hue + self.rng.range(-0.08, 0.08),
                        0.0,
                    ],
                }
            })
            .collect();
        // Into the ring, in at most two pieces.
        let first = (MAX_PARTICLES - self.head).min(ps.len());
        gfx.write_particles(self.head, &ps[..first]);
        gfx.write_particles(0, &ps[first..]);
        // Follow the burst's first particle on the CPU; drop probes the ring overwrote.
        let (start, end) = (self.head, self.head + ps.len());
        self.probes.retain(|p| {
            let i = p.index + if p.index < start { MAX_PARTICLES } else { 0 };
            !(start..end).contains(&i) && p.age < p.lifetime
        });
        self.probes.push(Probe {
            index: self.head,
            cpu: ps[0].motor.into(),
            rate: ps[0].rate.into(),
            age: 0.0,
            lifetime: ps[0].life[1],
        });
        self.head = (self.head + ps.len()) % MAX_PARTICLES;
        self.spawned += ps.len();
    }

    /// Compare the probes' GPU motors with their CPU twins (blocking readback); returns the
    /// largest coefficient difference.
    pub fn check(&mut self, gfx: &Gfx) -> f32 {
        let mut worst = 0.0f32;
        for p in &self.probes {
            let gpu: Motor<(), f32> = gfx.read_particles(p.index, 1)[0].motor.into();
            for (a, b) in gpu.c.iter().zip(&p.cpu.c) {
                worst = worst.max((a - b).abs());
            }
            self.compared += 1;
        }
        self.worst = self.worst.max(worst);
        worst
    }
}

fn wrap(x: f32, half: f32) -> f32 {
    if x > half {
        x - 2.0 * half
    } else if x < -half {
        x + 2.0 * half
    } else {
        x
    }
}

/// A saturated colour for a hue in `[0, 1)`.
pub fn hsv(h: f32) -> [f32; 3] {
    let f = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        1.0 - k.min(4.0 - k).clamp(0.0, 1.0)
    };
    [f(5.0), f(3.0), f(1.0)]
}

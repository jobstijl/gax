# numga's examples in gax

The examples of [numga](https://github.com/EelcoHoogendoorn/numga) (Eelco Hoogendoorn's Python
geometric algebra library), ported to gax: 52 binaries, each a small animation, each with numga's
tests and the numeric checks of its scenarios. The math of each example is a module that does not
touch the drawing code (numga checks the same of its own).

```sh
cargo run --release --bin mechanics_spinning_top          # a window; space pauses, r restarts
cargo run --release --bin mechanics_spinning_top -- --gif top.gif
cargo run --release --bin mechanics_spinning_top -- --png top.png --at 2.5
cargo test --release                                       # every example's tests
```

The shared drawing code (`src/`) is a software canvas in linear light: antialiased lines, disks,
polygons and text, per-pixel shading for the ray-traced examples, plot axes with contours and
colormaps, a depth-sorted 3D scene, and cameras posed by gax motors.

| area | examples |
|---|---|
| mechanics | `crystal_waves`, `inertia` (with `simplex`), `manipulability`, `modes`, `riccati`, `robot_arm`, `spinning_top`, `symmetry`, `tennis_racket`, `xpbd` (the Lie-group steppers are `src/shared/mechanics_lie.rs`) |
| optics | `lens_camera`, `thin_lens` |
| geometry | `camera_fit`, `cyclides`, `epipolar`, `fitting`, `kalman`, `multiview`, `odometry`, `pose_diffusion`, `projection`, `qem`, `registration`, `scenegraph`, `skinning`, `surface_curvature` |
| quadrics | `cayley_klein`, `cga_spherical_quadrics`, `conformal_elliptical`, `elliptic_physics`, `gaussian`, `quadric_collision`, `quadrics`, `s3_raytracer`, `spherical_quadrics` |
| math | `hopf`, `invariant_decomposition`, `klein_quadric`, `pascal`, `poncelet`, `spin_groups` |
| relativity | `curvature`, `dirac`, `gravitational_lensing`, `impulse` |
| quantum | `graphene`, `magnetic_resonance`, `process_tomography`, `two_spins` |
| electromagnetism | `constitutive`, `maxwell`, `second_harmonic` |

Binaries are named `<area>_<example>`.

## Differences from numga

* **Conventions.** gax's dual and regressive product are metric-free (ADR-009): numga's PGA3D
  points and joins come out with the opposite sign, which flips a few signs (joint torques, the
  orientation of a joined axis); the ports note each. Where numga uses the metric (Hodge) dual,
  in STA and R(4,1), the ports write it out.
* **Random streams.** numpy's streams cannot be reproduced; the ports use a small xorshift, and
  where a check depends on the draw (epipolar, three S³ scenes of elliptic physics) the seed is
  chosen and the reason given in a comment.
* **Missing tools in gax** are written locally: a non-symmetric eigensolver (spin groups),
  complex coefficients (Klein quadric), polynomial roots (cyclides), quadrature (lensing,
  curvature). Generalized eigenproblems against a singular metric are solved exactly with the
  forms' roles swapped in `eigh_with`.
* **Found on the way.** numga's Kalman prediction adds the motion noise inside the conjugation
  (operator precedence), which breaks the covariance's symmetry; the port adds it after. numga's
  multiview comment of "~95% progress at step 9-10" for the alternating solver does not
  reproduce (about 53%); the Schur variant converges as expected.
* **Scale.** Large batches are scaled to stay interactive (crystal waves' headings, the
  dispersion scans), with the tests at numga's sizes where it matters.

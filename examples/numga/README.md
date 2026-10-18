# numga's examples in gax

The examples of [numga](https://github.com/EelcoHoogendoorn/numga) (Eelco Hoogendoorn's Python
geometric algebra library), ported to gax: all of them, 105 binaries, each a small animation, each with
numga's tests and the numeric checks of its scenarios. The math of each example is a module that does not
touch the drawing code (numga checks the same of its own).

```sh
cargo run --release --bin mechanics_spinning_top          # a window; space pauses, r restarts
cargo run --release --bin mechanics_spinning_top -- --gif top.gif
cargo run --release --bin mechanics_spinning_top -- --png top.png --at 2.5
cargo test --release                                       # every example's tests
```

The shared drawing code (`src/`) draws with light, as `examples/warp` does (the shared
`gax-colour` crate): every colour is a light, a PGA3D point whose weight is its intensity; strokes
add their light with a glow, fills and images cover by their opacity, and the canvas is shown
through AgX. It is gax down to the pixels: pixels and rectangles are PGA2D points, axes and views
are point maps, a camera projects by meeting its rays with the image plane, and distances are the
norms of joins. On top: plot axes with contours and OkLab colormaps, a depth-sorted 3D scene, and
cameras posed by gax motors.

| area | examples |
|---|---|
| mechanics | `area_transport`, `bending`, `composite_strip`, `crystal_waves`, `earthquake`, `elastica`, `foucault`, `hanging_chain`, `inertia` (with `simplex`), `jeffery`, `kepler`, `manipulability`, `modal_xpbd`, `modes`, `riccati`, `robot_arm`, `rolling_ball`, `slip_systems`, `spinning_top`, `stress`, `symmetry`, `tennis_racket`, `tides`, `tricycle`, `vortices`, `wing`, `xpbd` (the Lie-group steppers are `src/shared/mechanics_lie.rs`) |
| optics | `coating`, `crystal_diffraction`, `lens_camera`, `photoelasticity`, `thin_lens` |
| geometry | `bezier`, `camera_fit`, `cyclides`, `epipolar`, `fitting`, `kalman`, `multiview`, `odometry`, `origami`, `pose_diffusion`, `projection`, `qem`, `registration`, `rubiks_cube`, `scenegraph`, `skinning`, `sundial`, `surface_curvature`, `three_point_pose` |
| quadrics | `cayley_klein`, `cga_spherical_quadrics`, `conformal_elliptical`, `elliptic_physics`, `gaussian`, `quadric_collision`, `quadrics`, `s3_raytracer`, `spherical_quadrics` |
| surfaces | `arap`, `dirac_waves`, `koebe`, `motor_loop`, `projective_loop`, `reciprocal`, `spin_transformations`, `weighted_dual` |
| math | `extensor_representations`, `fractals`, `hopf`, `invariant_decomposition`, `klein_quadric`, `octonions`, `pascal`, `poncelet`, `reflections`, `spin_groups` |
| relativity | `curvature`, `dirac`, `gravitational_lensing`, `impulse`, `kerr`, `ligo`, `spinors`, `twistors` |
| quantum | `graphene`, `hubbard_dimer`, `hubbard_ring`, `kane_mele`, `kitaev_wire`, `magnetic_resonance`, `neutrino_isospin`, `process_tomography`, `stabilizer`, `superconductivity`, `two_spins` |
| electromagnetism | `constitutive`, `eddy_brake`, `electrostriction`, `fresnel`, `induced_dipoles`, `maxwell`, `moving_charge`, `second_harmonic` |

Binaries are named `<area>_<example>`. A few keep the area numga had when they were ported: numga
has since moved `camera_fit`, `epipolar`, `kalman`, `multiview`, `odometry` and `pose_diffusion`
to `estimation`, `cyclides` to `quadrics`, and `qem`, `skinning` and `surface_curvature` to
`surfaces`.

## Differences from numga

* **Conventions.** gax's dual and regressive product are metric-free (ADR-009): numga's PGA3D
  points and joins come out with the opposite sign, which flips a few signs (joint torques, the
  orientation of a joined axis); the ports note each. Where numga uses the metric (Hodge) dual,
  in STA and R(4,1), the ports write it out.
* **Random streams.** numpy's streams cannot be reproduced; the ports draw from `rand`'s seeded
  generator (`src/rng.rs`), and where a check depends on the draw (epipolar, three S³ scenes of
  elliptic physics, the tides' stars) the seed is chosen and the reason given in a comment.
* **Meshes.** numga's `examples/mesh.py` is `src/mesh.rs`: meshes whose vertices and quantities
  are gax fields and whose incidence operators are sparse field maps. Where numga calls scipy,
  the ports use gax: the disks' Delaunay triangulation and the convex hull of koebe and weighted
  dual are written with gax's orientation predicates; koebe's `least_squares` and BFGS are
  Gauss-Newton steps of sparse least-squares solves and Newton's method on gax's derivatives;
  weighted dual's `linprog` and SLSQP are a log barrier on gax's sparse solves.
* **Missing tools in gax** are written locally: polynomial roots (cyclides), quadrature (lensing,
  curvature, spin transformations). Generalized eigenproblems against a singular metric are
  solved exactly with the forms' roles swapped in `eigh`. Complex coefficients and the
  eigenvalues of non-symmetric maps, first written locally, are now gax's (`gax::Complex`,
  `eigvals`; ADR-042).
* **Found on the way.** numga's Kalman prediction adds the motion noise inside the conjugation
  (operator precedence), which breaks the covariance's symmetry; the port adds it after. numga's
  multiview comment of "~95% progress at step 9-10" for the alternating solver does not
  reproduce (about 53%); the Schur variant converges as expected.
* **Scale.** Large batches are scaled to stay interactive (crystal waves' headings, the
  dispersion scans, Kane-Mele's mass sweep), with the tests at numga's sizes where it matters;
  independent scans and runs at start-up are spread over the cores (`par_map`).

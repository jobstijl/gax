# numga's examples in gax

The examples of [numga](https://github.com/EelcoHoogendoorn/numga) (Eelco Hoogendoorn's Python
geometric algebra library), ported to gax: all of them as of numga `35797de`, 166 binaries, each a small
animation, each with numga's tests and the numeric checks of its scenarios. The math of each example is a module that does not
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
| mechanics | `area_transport`, `bending`, `composite_strip`, `crystal_waves`, `earthquake`, `elastica`, `foucault`, `freeform`, `hanging_chain`, `inertia` (with `simplex`), `jeffery`, `kepler`, `manipulability`, `modal_xpbd`, `modes`, `odd_elasticity`, `rattleback`, `riccati`, `robot_arm`, `rolling_ball`, `seismic_waves`, `shells`, `slip_systems`, `spinning_top`, `stress`, `symmetry`, `tennis_racket`, `tides`, `tippe_top`, `tricycle`, `vortices`, `wing`, `xpbd` (the Lie-group steppers are `src/shared/mechanics_lie.rs`) |
| fluids | `dec_stokes`, `eigen_fluids`, `floating`, `kirchhoff`, `projective_sphere_flow`, `smoke_rings`, `sphere_vortices`, `stokes_swimmer` |
| optics | `black_sandwich`, `coating`, `crystal_diffraction`, `lens_camera`, `photoelasticity`, `rainbow`, `specular_caustics`, `thin_lens` |
| geometry | `bezier`, `camera_fit`, `cyclides`, `epipolar`, `fitting`, `kalman`, `multiview`, `odometry`, `origami`, `pose_diffusion`, `projection`, `qem`, `registration`, `rubiks_cube`, `scenegraph`, `skinning`, `sundial`, `surface_curvature`, `three_point_pose` |
| quadrics | `cayley_klein`, `cga_spherical_quadrics`, `conformal_elliptical`, `csg_raytracer`, `elliptic_physics`, `gaussian`, `glass_cyclide`, `quadric_collision`, `quadrics`, `s3_raytracer`, `spherical_quadrics` |
| surfaces | `arap`, `connections`, `developability`, `dirac_kahler`, `dirac_waves`, `frame_fields`, `k_surfaces`, `koebe`, `minimal_surfaces`, `motor_loop`, `projective_loop`, `reciprocal`, `regge_calculus`, `spin_transformations`, `weighted_dual`, `whitney_forms` (numga's simplicial mesh is `src/shared/simplicial.rs`) |
| math | `doubled_algebra`, `e8`, `e_series`, `eight_conics`, `extensor_representations`, `fractals`, `g2_rolling`, `hopf`, `invariant_decomposition`, `isospectral_drums`, `klein_quadric`, `octonions`, `pascal`, `penrose`, `poncelet`, `reflections`, `spherical_synchronization`, `spin_groups`, `triality` |
| relativity | `curvature`, `dirac`, `gps`, `gravitational_lensing`, `impulse`, `kerr`, `ligo`, `moving_mirror`, `radar`, `spinors`, `terrell`, `thomas_precession`, `twistors` |
| quantum | `affine_toda`, `graphene`, `hubbard_dimer`, `hubbard_ring`, `kane_mele`, `kitaev_wire`, `magnetic_resonance`, `neutrino_isospin`, `particle_charges`, `process_tomography`, `stabilizer`, `superconductivity`, `two_spins` |
| electromagnetism | `beltrami_fields`, `constitutive`, `eddy_brake`, `electrostriction`, `fresnel`, `induced_dipoles`, `lie_advection`, `magnetic_bottle`, `maxwell`, `moving_charge`, `piezoelectric`, `second_harmonic` |
| orientation | `conformal_circle`, `mechanical_work`, `mirror_energy`, `normals`, `optical_rotation`, `wu_experiment` (intrinsic and extrinsic elements under a mirror, ADR-051) |
| thermodynamics | `contact_geometry`, `entropy_production`, `generic`, `heat_cycles`, `phase_change`, `property_charts`, `ruppeiner`, `state_space` (Grassmann algebras, every generator squaring to zero; the charts are `src/shared/thermodynamics_charts.rs`) |

Binaries are named `<area>_<example>`. A few keep the area numga had when they were ported: numga
has since moved `camera_fit`, `epipolar`, `kalman`, `multiview`, `odometry` and `pose_diffusion`
to `estimation`, `cyclides` to `quadrics`, `qem`, `skinning` and `surface_curvature` to
`surfaces`, and `jeffery`, `vortices` and `wing` to `fluids`. `maxwell` is numga's of `7dff347`:
numga has since folded its stress-energy into `constitutive` (6e38d341).

## Differences from numga

* **Conventions.** gax's dual and regressive product are metric-free (ADR-009), and numga's
  regressive product is too since `f9ced1bf`; numga's `dual()` is the metric dual, gax's
  `hodge()`, and numga's `adjugate()` is gax's (ADR-053). Ports written before numga `93c2ae3`
  note the PGA3D signs of that time (joint torques, the orientation of a joined axis).
* **STA's signature.** numga's STA was `t+x-y-z-` when most spacetime ports were written, and
  is `x+y+z+t-` since f9ced1bf; gax's `sta` followed (ADR-055), with numga's blades in numga's
  order. The older ports compute numga's physics of then in the new signature: a timelike
  vector squares to `−1`, and the comments say where a sign or a reading changed.
* **Random streams.** numpy's streams cannot be reproduced; the ports draw from `rand`'s seeded
  generator (`src/rng.rs`), and where a check depends on the draw (epipolar, three S³ scenes of
  elliptic physics, the tides' stars) the seed is chosen and the reason given in a comment.
* **Algebras beyond gax's.** numga's STA2D `x+y+t-` is `src/shared/sta2d.rs`, its diagonal CGA3D
  `src/shared/quadrics_diagonal_cga.rs`, E8's R(8,0) `src/shared/math_e8.rs`; particle charges'
  Cl(10,0), above gax's nine dimensions, runs in its even subalgebra Cl(0,9).
* **Meshes.** numga's `examples/mesh.py` is `src/mesh.rs`: meshes whose vertices and quantities
  are gax fields and whose incidence operators are sparse field maps. Where numga calls scipy,
  the ports use gax: the disks' Delaunay triangulation and the convex hull of koebe and weighted
  dual are written with gax's orientation predicates; koebe's `least_squares` and BFGS are
  Gauss-Newton steps of sparse least-squares solves and Newton's method on gax's derivatives;
  weighted dual's `linprog` and SLSQP are a log barrier on gax's sparse solves.
* **Missing tools in gax** are written locally: polynomial roots (cyclides), quadrature (lensing,
  curvature, spin transformations, Kirchhoff's Gauss–Legendre rule), an L-BFGS and a Halton
  sequence (freeform), a dense Jacobi eigensolver of run-time size (Beltrami fields), block
  saddle-point systems flattened into one sparse map (DEC Stokes, piezoelectric); TODO.md's
  Phase 13 lists what gax should take over. Generalized eigenproblems against a singular metric are
  solved exactly with the forms' roles swapped in `eigh`. Complex coefficients and the
  eigenvalues of non-symmetric maps, first written locally, are now gax's (`gax::Complex`,
  `eigvals`; ADR-042).
* **Found on the way.** numga `7dff347`'s table of turn senses in Kane-Mele gave every turn the
  sign opposite to its geometry (numga's newer bond commutator has it right; the port follows).
  numga's constitutive renderer animates the waves `s t − z` along `+z`; in numga's own algebra
  they travel along `−z`, and the port draws them so.
* **Scale.** Large batches are scaled to stay interactive (crystal waves' headings, the
  dispersion scans, Kane-Mele's mass sweep), with the tests at numga's sizes where it matters;
  independent scans and runs at start-up are spread over the cores (`par_map`).

//! The math: barycentric hats and Whitney forms on a triangle of the plane (VGA2D) and on a
//! tetrahedron of space (VGA3D), the Whitney maps of a triangle mesh, exact midpoint quadrature
//! of a material's conductance, and steady conduction. No drawing.
//!
//! Points are antivectors, intrinsic (numga's `Point-`): in the plane the Hodge duals of vectors,
//! in space intrinsic bivectors, whose coefficients are the coordinates. A triangle's patch is
//! the join of two of its sides, and a corner's hat gradient is the side facing it over twice
//! the patch: a vector, extrinsic (numga's `Plane+`), which pairs with a step by the join.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::{vga2d, vga3d};

use crate::simplicial::parity;
use gax::mesh::corners as local_corners;

/// A scalar (numga's `Scalar+`).
pub type Scalar = vga2d::Scalar;
/// A vector, a hat gradient or an edge form (numga's `Plane+`).
pub type Plane = vga2d::Vector;
/// A bivector, a face form (numga's `Bivector+`).
pub type Bivector = vga2d::Bivector;
/// A point or a step of the plane, the dual of its vector (numga's `Point-`).
pub type Point = vga2d::intrinsic::Vector;
/// A triangle's patch, the join of two sides (numga's `Patch-`).
pub type Patch = vga2d::intrinsic::Scalar;
/// What a conductance gives a potential: an amount, a quantity per area (numga's `Amount-`).
/// A material: a gradient to the current it drives (numga's `Point- <- Plane+`).
pub type Conductivity = vga2d::intrinsic::Vector<(Plane,)>;
/// A map from a scalar to a plane, the cell of a gradient or a Whitney map.
pub type PlaneMap = vga2d::Vector<(Scalar,)>;
/// The cell of a curl or a Whitney 2-map.
pub type BivectorMap = vga2d::Bivector<(Scalar,)>;
/// The cell of a conductance, from potentials to amounts (numga's `Amount- <- Scalar+`).
pub type AmountMap = vga2d::intrinsic::Pseudoscalar<(Scalar,)>;

/// The point at `(x, y)`: the Hodge dual of its vector (numga's `x * mv.x.dual() + ...`).
pub fn point(x: f64, y: f64) -> Point {
    Plane::new(x, y).hodge()
}

/// A point's coordinates, read by joining it with the slices across `x` and `y`.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (Plane::new(1.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0) & p).s(),
    ]
}

/// `complements(n, m)`: the corners each `m`-simplex of an `n`-simplex leaves out, in the order
/// of `corners(n, m)`, run so that the `m`-simplex followed by them runs the `n`-simplex's own
/// sense: facing each corner, its facet, outward; across each edge of a tetrahedron, the
/// opposite edge.
pub fn complements(n: usize, m: usize) -> Vec<Vec<usize>> {
    local_corners(n, m)
        .into_iter()
        .map(|simplex| {
            let left: Vec<usize> = (0..=n).filter(|c| !simplex.contains(c)).collect();
            let sense = parity(&[simplex, left.clone()].concat());
            if sense > 0 {
                left
            } else {
                left.into_iter().rev().collect()
            }
        })
        .collect()
}

/// The sides of a triangle facing each corner, counter-clockwise: from the corner after it to
/// the one after that (numga's `triangle.boundary[2, 1] * (triangle.d[0] * corners)`).
pub fn sides(corners: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3])
}

/// A triangle's patch: half the join of two of its sides, positive counter-clockwise.
pub fn patch(sides: [Point; 3]) -> Patch {
    (sides[0] & sides[1]) * 0.5
}

/// The barycentric hats at each point and the constant corner gradients: each opposite side
/// over twice the patch is the gradient that falls by one from its corner to that side.
pub fn barycentric_hats(corners: [Point; 3], points: &[Point]) -> (Vec<[Scalar; 3]>, [Plane; 3]) {
    let sides = sides(corners);
    let twice = (patch(sides) * 2.0).inverse();
    let gradients = sides.map(|s| s * twice);
    let hats = points
        .iter()
        .map(|&p| [0, 1, 2].map(|k| Scalar::new(1.0) + (gradients[k] & (p - corners[k]))))
        .collect();
    (hats, gradients)
}

/// The three edge Whitney 1-forms at each point, each circulating around the corner its edge
/// faces: the step from that corner over twice the patch. Along the two sides at the corner the
/// step runs along the side; along the opposite side its join with the side is twice the patch.
pub fn edge_whitney_forms(corners: [Point; 3], points: &[Point]) -> Vec<[Plane; 3]> {
    let twice = (patch(sides(corners)) * 2.0).inverse();
    points
        .iter()
        .map(|&p| corners.map(|c| (p - c) * twice))
        .collect()
}

/// A point of space: the intrinsic bivector whose coefficients are its coordinates.
pub type SpacePoint = vga3d::intrinsic::Bivector;
/// A vector of space (numga's `SpacePlane+`).
pub type SpacePlane = vga3d::Vector;
/// A bivector of space (numga's `SpaceBivector+`).
pub type SpaceBivector = vga3d::Bivector;
/// A face's patch, the join of two sides: an intrinsic vector (numga's `SpacePatch-`).
pub type SpacePatch = vga3d::intrinsic::Vector;
/// A tetrahedron's volume, the join of three edges (numga's `Volume-`).
pub type Volume = vga3d::intrinsic::Scalar;

/// The point of space at `(x, y, z)`.
pub fn space_point(x: f64, y: f64, z: f64) -> SpacePoint {
    SpacePlane::new(x, y, z).hodge()
}

/// The tetrahedron's mesh quantities numga's tetrahedral functions share: its corners, its edges
/// in its own vertex order, its signed volume.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Tetrahedron {
    pub corners: [SpacePoint; 4],
    pub volume: Volume,
    /// The outward patch of the face facing each corner.
    pub face_patches: [SpacePatch; 4],
    /// The corner gradients: each face's patch turned inward, over three times the volume.
    pub gradients: [SpacePlane; 4],
}

/// The tetrahedron `top` of a solid mesh.
pub fn tetrahedron(mesh: &Topology<4>, vertices: &Field<SpacePoint>, top: usize) -> Tetrahedron {
    let corners: [SpacePoint; 4] =
        core::array::from_fn(|k| mesh.incidence(3, 0).pick(k, vertices).site(top));
    let edges = &mesh.d::<vga3d::Scalar>(0) * vertices;
    // The first three edges of its own order run from its first corner.
    let own = mesh.oriented(3, 1).read_all(&edges);
    let volume = (own[0].site(top) & own[1].site(top) & own[2].site(top)) * (1.0 / 6.0);
    // Each face's patch, half the join of two of its sides, read as the face facing each corner,
    // outward.
    let sides = mesh.facets(2).read_all(&edges);
    let patches: Field<SpacePatch> = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
    let outward = mesh.facets(3).read_all(&patches);
    let face_patches: [SpacePatch; 4] = core::array::from_fn(|k| outward[k].site(top));
    // A step from a corner across to its face reads three times the volume on the outward
    // patch, and the hat falls by one along it.
    let thrice = (volume * 3.0).inverse();
    let gradients = face_patches.map(|p| -(p * thrice));
    Tetrahedron {
        corners,
        volume,
        face_patches,
        gradients,
    }
}

impl Tetrahedron {
    /// The barycentric hats at a point.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn hats(&self, p: SpacePoint) -> [vga3d::Scalar; 4] {
        core::array::from_fn(|k| {
            vga3d::Scalar::new(1.0) + (self.gradients[k] & (p - self.corners[k]))
        })
    }

    /// The six edge Whitney 1-forms at a point, each circulating around the edge across its edge
    /// (in the order of `corners(3, 1)`): the patch the opposite edge sweeps to the point,
    /// which vanishes along that edge and reads one along the edge itself over six times the
    /// volume.
    pub fn edge_forms(&self, p: SpacePoint) -> [SpacePlane; 6] {
        let across = complements(3, 1);
        let six = (self.volume * 6.0).inverse();
        core::array::from_fn(|e| {
            let (tail, head) = (across[e][0], across[e][1]);
            let opposite = self.corners[head] - self.corners[tail];
            (opposite & (p - self.corners[tail])) * six
        })
    }

    /// The edges' constant curls: the opposite edge over three times the volume.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn curls(&self) -> [SpaceBivector; 6] {
        let across = complements(3, 1);
        let thrice = (self.volume * 3.0).inverse();
        core::array::from_fn(|e| (self.corners[across[e][1]] - self.corners[across[e][0]]) * thrice)
    }

    /// The four face Whitney 2-forms at a point, each radiating from the corner its face faces:
    /// the step from the corner over three times the volume.
    pub fn face_forms(&self, p: SpacePoint) -> [SpaceBivector; 4] {
        let thrice = (self.volume * 3.0).inverse();
        self.corners.map(|c| (p - c) * thrice)
    }

    /// The volume pairing of each edge's Whitney 1-form with each edge's curl, made symmetric:
    /// it links only opposite edges, the same for every tetrahedron of the same sense. At the
    /// centroid every hat is a quarter, so each edge form is a quarter of the difference of its
    /// ends' gradients, and a linear form against a constant curl integrates exactly there.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn helicity(&self) -> [[vga3d::Scalar; 6]; 6] {
        let ends = local_corners(3, 1);
        let forms: [SpacePlane; 6] = core::array::from_fn(|e| {
            (self.gradients[ends[e][1]] - self.gradients[ends[e][0]]) * 0.25
        });
        let curls = self.curls();
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (((forms[i] ^ curls[j]) + (forms[j] ^ curls[i])) * 0.5) & self.volume
            })
        })
    }
}

/// The topological derivatives and the Whitney maps of a triangle mesh of the plane.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Operators {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub d0: SparseMap<Scalar>,
    pub d1: SparseMap<Scalar>,
    /// Each face's corners.
    pub corners: [Field<Point>; 3],
    pub patches: Field<Patch>,
    pub areas: Field<Scalar>,
    /// The constant corner gradients on each face.
    pub corner_gradients: [Field<Plane>; 3],
    /// Vertex potentials to piecewise-constant face gradients, assembled by corner, no metric.
    pub gradient: SparseMap<PlaneMap>,
    /// Edge cochains to their Whitney 1-forms at the face centroids.
    pub whitney1: SparseMap<PlaneMap>,
    /// Edge cochains to the derivatives of their Whitney 1-forms.
    pub curl: SparseMap<BivectorMap>,
    /// Face cochains to their constant Whitney 2-forms.
    pub whitney2: SparseMap<BivectorMap>,
    /// Edge cochains to their Whitney 1-forms at the midpoint of each side, the side facing each
    /// corner.
    pub midpoint_whitney1: [SparseMap<PlaneMap>; 3],
}

impl Operators {
    pub fn new(mesh: &Topology<3>, vertices: &Field<Point>) -> Self {
        let (d0, d1) = (mesh.d::<Scalar>(0), mesh.d::<Scalar>(1));
        let edges = &d0 * vertices;
        let at_corners = mesh.incidence(2, 0);
        let corners: [Field<Point>; 3] = core::array::from_fn(|k| at_corners.pick(k, vertices));
        let facing = mesh.facets(2);
        let sides: [Field<Point>; 3] = core::array::from_fn(|k| facing.read(k, &edges));
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = patches.map(|p| Scalar::new(p.s().abs()));

        // Each corner hat's gradient is constant across its face; assembling by corner maps
        // vertex potentials to piecewise-constant face gradients without any metric.
        let twice = patches.map(|p| (p * 2.0).inverse());
        let corner_gradients: [Field<Plane>; 3] =
            core::array::from_fn(|k| sides[k].zip_with(&twice, |s, t| s * t));
        let slot = Scalar::slot();
        let gradient = at_corners.assemble(|f, k| corner_gradients[k].site(f) * slot);

        // At each centroid the three hats are a third, so an edge's 1-form is a third of the
        // difference of its ends' gradients.
        let ends = local_corners(2, 1);
        let own = mesh.oriented(2, 1);
        let centroid_form = |f: usize, k: usize| {
            let (tail, head) = (ends[k][0], ends[k][1]);
            (corner_gradients[head].site(f) - corner_gradients[tail].site(f)) * (1.0 / 3.0)
        };
        let whitney1 = own.assemble(|f, k| centroid_form(f, k) * slot);

        // The derivative of each edge's 1-form is twice the wedge of its ends' gradients: the
        // constant 2-form of unit integral over the face, times the edge's share of it.
        let curl = own.assemble(|f, k| {
            let (tail, head) = (ends[k][0], ends[k][1]);
            (corner_gradients[tail].site(f) ^ corner_gradients[head].site(f)) * 2.0 * slot
        });
        let unit = Bivector::new(1.0);
        let whitney2 =
            SparseMap::from_diagonal(&patches.map(|p| unit * (unit & p).inverse() * slot));

        // The linear 1-forms at the three side midpoints integrate quadratic products over each
        // triangle exactly.
        let midpoint_whitney1 = core::array::from_fn(|q| {
            let middle = |f: usize| {
                let (a, b) = (corners[(q + 1) % 3].site(f), corners[(q + 2) % 3].site(f));
                (a + b) * 0.5
            };
            facing.assemble(|f, k| ((middle(f) - corners[k].site(f)) * twice.site(f)) * slot)
        });
        Operators {
            mesh: mesh.clone(),
            vertices: vertices.clone(),
            d0,
            d1,
            corners,
            patches,
            areas,
            corner_gradients,
            gradient,
            whitney1,
            curl,
            whitney2,
            midpoint_whitney1,
        }
    }

    /// The edges' Whitney 1-forms at a point of every face (`at`, one per face): the forms of
    /// the side facing each corner, each the step from that corner over twice the patch, signed
    /// as the face runs along the stored edge.
    pub fn whitney1_at(&self, at: &Field<Point>) -> SparseMap<PlaneMap> {
        let twice = self.patches.map(|p| (p * 2.0).inverse());
        self.mesh.facets(2).assemble(|f, k| {
            ((at.site(f) - self.corners[k].site(f)) * twice.site(f)) * Scalar::slot()
        })
    }

    /// `M <- M` maps on faces from a material map on each face.
    fn on_faces<
        C: gax::Extensor<Slots = (Plane,), Coef = f64> + core::ops::Mul<f64, Output = C>,
    >(
        &self,
        material: C,
        weights: &Field<Scalar>,
    ) -> SparseMap<C> {
        SparseMap::from_diagonal(&weights.map(|w| material * w.s()))
    }

    /// The 1-form conductance on edges, integrated exactly over each triangle from its three
    /// side midpoints: each carries a third of the area, and the material pulled back through
    /// the midpoint Whitney maps, summed over the three, is exact for quadratics.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
    pub fn edge_conductance(&self, conductivity: Conductivity) -> SparseMap<AmountMap> {
        let thirds = self.areas.map(|a| a * (1.0 / 3.0));
        let weights = self.on_faces(conductivity, &thirds);
        let shares = self
            .midpoint_whitney1
            .iter()
            .map(|w| w.adjugate().compose(&weights.compose(w)));
        shares.reduce(|a, b| &a + &b).expect("three midpoints")
    }

    /// The vertex conductance: the area-weighted material pulled back through the face
    /// gradient by its adjugate.
    pub fn conductance(&self, conductivity: Conductivity) -> SparseMap<AmountMap> {
        let weights = self.on_faces(conductivity, &self.areas);
        self.gradient
            .adjugate()
            .compose(&weights.compose(&self.gradient))
    }

    /// Steady conduction with the `electrodes` held at the `prescribed` potential (zero
    /// elsewhere): the potential, the current on each face, and the streamfunction whose level
    /// curves follow the current.
    pub fn conduction(
        &self,
        conductivity: Conductivity,
        electrodes: &[usize],
        prescribed: &Field<Scalar>,
    ) -> (Field<Scalar>, Field<Point>, Field<Scalar>) {
        let n = self.vertices.len();
        let conductance = self.conductance(conductivity);
        let free_vertices: Vec<usize> = (0..n).filter(|v| !electrodes.contains(v)).collect();
        let free =
            (&SparseMap::<Scalar>::selection(&free_vertices, n) * Scalar::slot()).transpose();
        let rhs = -&free.adjugate().of(&conductance.of(prescribed));
        let system = free.adjugate().compose(&conductance.compose(&free));
        let potential = prescribed + &free.of(&system.solve(&rhs));
        let current = self.gradient.of(&potential).map(|g| -conductivity.of(g));

        // The current turned a quarter turn is, through the unit isotropic conductivity, the
        // gradient of the streamfunction whose level curves follow the current paths.
        let quarter_turn = (Bivector::new(-core::f64::consts::FRAC_PI_4)).exp();
        let laplacian = self.conductance(isotropic());
        let unpinned: Vec<usize> = (1..n).collect();
        let unpinned = (&SparseMap::<Scalar>::selection(&unpinned, n) * Scalar::slot()).transpose();
        let turned = self
            .areas
            .zip_with(&current, |a, c| (quarter_turn >> c) * a.s());
        let stream_rhs = unpinned
            .adjugate()
            .of(&self.gradient.adjugate().of(&turned));
        let stream_system = unpinned.adjugate().compose(&laplacian.compose(&unpinned));
        let streamfunction = unpinned.of(&stream_system.solve(&stream_rhs));
        (potential, current, streamfunction)
    }
}

/// The unit isotropic conductivity: a gradient to the current along it (numga's
/// `Plane.dual()`).
pub fn isotropic() -> Conductivity {
    Plane::slot().hodge()
}

/// A conductivity with the principal values along a unit direction and across it: the sum of
/// the dyads of the two directions, each times its value.
pub fn anisotropic(fibre: Point, along: f64, across: f64) -> Conductivity {
    let quarter_turn = (Bivector::new(-core::f64::consts::FRAC_PI_4)).exp();
    let directions = [fibre, quarter_turn >> fibre];
    let dyad = |d: Point, k: f64| d * (Plane::slot() & d) * k;
    dyad(directions[0], along) + dyad(directions[1], across)
}

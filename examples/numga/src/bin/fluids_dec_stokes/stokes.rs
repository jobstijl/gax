//! The math: the exterior derivatives of a triangle mesh and its circumcentric Hodge maps, the
//! vorticity of edge fluxes, and steady Stokes flow by a saddle-point solve. No drawing.
//!
//! Integrated fluxes are numbers on the edges (VGA2D scalars). The derivatives hold only the
//! signs of the incidences; lengths and areas enter through the Hodge maps, each the oriented
//! patch of a dual cell over its primal cell's. Points are antivectors, intrinsic (numga's
//! `Point-`), and the join of two is a patch (`Patch-`); a vertex's vorticity is a bivector.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga2d;

use crate::simplicial::{boundary, incidence};

/// A number (numga's `Scalar+`).
pub type Scalar = vga2d::Scalar;
/// A vorticity (numga's `Bivector+`).
pub type Bivector = vga2d::Bivector;
/// A point or a displacement (numga's `Point-`).
pub type Point = vga2d::intrinsic::Vector;
/// An oriented patch, the join of two displacements (numga's `Patch-`).
pub type Patch = vga2d::intrinsic::Scalar;
/// A map of numbers.
pub type ScalarMap = vga2d::Scalar<(Scalar,)>;
/// A Hodge map's cell (numga's `Patch- <- Scalar+`).
pub type HodgeMap = vga2d::intrinsic::Scalar<(Scalar,)>;
/// The cell of a vorticity's derivative (numga's `Bivector+ <- Bivector+`).
pub type VorticityMap = vga2d::Bivector<(Bivector,)>;

/// The point at `(x, y)`: the dual of its vector.
pub fn point(x: f64, y: f64) -> Point {
    vga2d::Vector::new(x, y).hodge()
}

/// A point's coordinates, read with the lines across the axes.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (vga2d::Vector::new(1.0, 0.0) & p).s(),
        (vga2d::Vector::new(0.0, 1.0) & p).s(),
    ]
}

/// The edges with a single face beside them.
pub fn boundary_edges(mesh: &Topology<3>) -> Vec<usize> {
    let mut count = vec![0; mesh.len(1)];
    for f in 0..mesh.len(2) {
        for &(e, _) in mesh.faces(2, f) {
            count[e] += 1;
        }
    }
    (0..mesh.len(1)).filter(|&e| count[e] == 1).collect()
}

/// A map with its rows (`rows`) or columns multiplied by signs.
fn signed<C>(map: &SparseMap<C>, rows: Option<&[f64]>, columns: Option<&[f64]>) -> SparseMap<C>
where
    C: gax::Extensor + Copy + core::ops::Mul<f64, Output = C>,
{
    let triples: Vec<_> = map
        .couplings()
        .map(|(r, c, cell)| {
            let sign = rows.map_or(1.0, |s| s[r]) * columns.map_or(1.0, |s| s[c]);
            (r, c, cell * sign)
        })
        .collect();
    SparseMap::from_triples(triples, map.shape())
}

/// The topological derivatives and the circumcentric Hodge maps of a mesh of the plane.
pub struct Operators {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    /// The orientation of each edge against its stored one (its lower vertex first): `−1`
    /// where it is read the other way.
    pub signs: Vec<f64>,
    pub d0: SparseMap<ScalarMap>,
    pub d1: SparseMap<ScalarMap>,
    pub d0_vorticity: SparseMap<VorticityMap>,
    pub edges: Field<Point>,
    pub corners: [Field<Point>; 3],
    pub areas: Field<Scalar>,
    pub face_centres: Field<Point>,
    pub circumcentres: Field<Point>,
    /// Each edge's dual edge: between the circumcentres beside it, or from the circumcentre
    /// to the midpoint on the boundary, crossing the edge to its right.
    pub dual_edges: Field<Point>,
    #[cfg_attr(not(test), allow(dead_code))] // numga's operator, which the tests check
    pub edge_ratios: Field<Patch>,
    pub dual_patches: Field<Patch>,
    #[cfg_attr(not(test), allow(dead_code))] // numga's operator, which the tests check
    pub dual_areas: Field<Scalar>,
    #[cfg_attr(not(test), allow(dead_code))] // numga's operator, which the tests check
    pub star0: SparseMap<HodgeMap>,
    pub star1: SparseMap<HodgeMap>,
    #[cfg_attr(not(test), allow(dead_code))] // numga's operator, which the tests check
    pub star2: SparseMap<HodgeMap>,
    /// Patches to vorticities: each vertex's dual over its dual area.
    pub inverse_star0: SparseMap<vga2d::Bivector<(Patch,)>>,
    /// Edge fluxes to vertex vorticities: the circulation along the dual edges, summed around
    /// each dual cell by the adjugate of the vorticity's derivative, over the cell's area.
    pub curl: SparseMap<vga2d::Bivector<(Scalar,)>>,
}

impl Operators {
    /// The operators of a mesh, each edge oriented by `signs` (`+1` as stored).
    pub fn new(mesh: &Topology<3>, vertices: &Field<Point>, signs: &[f64]) -> Self {
        let differences = signed(&mesh.d::<Scalar>(0), Some(signs), None);
        let circulations = signed(&mesh.d::<Scalar>(1), None, Some(signs));
        let d0 = &differences * Scalar::slot();
        let d1 = &circulations * Scalar::slot();
        let d0_vorticity = &differences * Bivector::slot();
        let edges = &differences * vertices;
        let at_corners = incidence(mesh, 2, 0);
        let corners: [Field<Point>; 3] = [0, 1, 2].map(|k| at_corners.pick(k, vertices));
        let facing = boundary(mesh, 2);
        let sides: [Field<Point>; 3] = [0, 1, 2].map(|k| {
            Field::from_fn(mesh.len(2), |f| {
                let e = facing.sites[f][k];
                edges.site(e) * (f64::from(facing.signs[f][k]) * signs[e])
            })
        });
        let face_patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = face_patches.map(|p| Scalar::new(p.norm()));
        let face_centres = Field::from_fn(mesh.len(2), |f| {
            (corners[0].site(f) + corners[1].site(f) + corners[2].site(f)) * (1.0 / 3.0)
        });
        // On the paraboloid of half-squared distances, the plane through the corners has the
        // circumcentre as its slope.
        let circumcentres = Field::from_fn(mesh.len(2), |f| {
            let twice = (face_patches.site(f) * 2.0).inverse();
            (0..3)
                .map(|k| {
                    let c = corners[k].site(f);
                    (sides[k].site(f) * twice) * (c.norm_squared() / 2.0)
                })
                .fold(vga2d::Vector::zero(), |a, b| a + b)
                .hodge()
        });
        // The dual edge crosses each edge to its right: from the face on its left to the one on
        // its right, and on the boundary from the face to the edge's midpoint (numga's
        // `boundary * midpoints − ~d1 * circumcentres`, `boundary` the column sums of `d1`).
        let mut dual_edges = Field::<Point>::zeros(mesh.len(1));
        for (r, e, sign) in circulations.couplings() {
            let ends = mesh.simplex(1, e);
            let middle = (vertices.site(ends[0]) + vertices.site(ends[1])) * 0.5;
            dual_edges.sites_mut()[e] += (middle - circumcentres.site(r)) * sign.s();
        }
        // An interior edge's two faces add their midpoints with opposite signs.
        let crossings = dual_edges.zip_with(&edges, |d, e| d & e);
        let edge_ratios = crossings.zip_with(&edges, |c, e| c * (1.0 / e.norm_squared()));
        let mut dual_patches = Field::<Patch>::zeros(mesh.len(0));
        for e in 0..mesh.len(1) {
            for &v in mesh.simplex(1, e) {
                dual_patches.sites_mut()[v] += crossings.site(e) * 0.25;
            }
        }
        let dual_areas = dual_patches.map(|p| Scalar::new(p.norm()));
        let slot = Scalar::slot();
        let star0 = SparseMap::from_diagonal(&dual_patches.map(|p| p * slot));
        let star1 = SparseMap::from_diagonal(&edge_ratios.map(|p| p * slot));
        let star2 = SparseMap::from_diagonal(&face_patches.map(|p| p.inverse() * slot));
        let inverse_star0 =
            SparseMap::from_diagonal(&dual_areas.map(|a| Patch::slot().hodge() * (1.0 / a.s())));
        let curl = inverse_star0.compose(&d0_vorticity.adjugate().compose(&star1));
        Operators {
            mesh: mesh.clone(),
            vertices: vertices.clone(),
            signs: signs.to_vec(),
            d0,
            d1,
            d0_vorticity,
            edges,
            corners,
            areas,
            face_centres,
            circumcentres,
            dual_edges,
            edge_ratios,
            dual_patches,
            dual_areas,
            star0,
            star1,
            star2,
            inverse_star0,
            curl,
        }
    }
}

impl Operators {
    /// Each dual edge as a segment: from the circumcentre of the face on the edge's left (the one
    /// that runs along it), or from the edge's midpoint where it has none, along the dual edge.
    pub fn dual_segments(&self) -> Vec<[Point; 2]> {
        let mut starts: Vec<Option<Point>> = vec![None; self.mesh.len(1)];
        for (r, e, cell) in self.d1.couplings() {
            if cell.of(Scalar::new(1.0)).s() > 0.0 {
                starts[e] = Some(self.circumcentres.site(r));
            }
        }
        (0..self.mesh.len(1))
            .map(|e| {
                let ends = self.mesh.simplex(1, e);
                let middle = (self.vertices.site(ends[0]) + self.vertices.site(ends[1])) * 0.5;
                let start = starts[e].unwrap_or(middle);
                [start, start + self.dual_edges.site(e)]
            })
            .collect()
    }
}

/// A steady flow: the fluxes through the edges, the pressure on the faces (of zero
/// area-weighted mean), the vorticity at the vertices and the mean velocity of each face.
pub struct Flow {
    pub flux: Field<Scalar>,
    pub pressure: Field<Scalar>,
    pub vorticity: Field<Bivector>,
    pub velocity: Field<Point>,
}

/// Steady Stokes flow with prescribed boundary fluxes and tangential boundary circulation: flux
/// is positive through the right side of a directed edge, and the pressure enforces zero net
/// flux through every face against the viscous resistance, the vorticity's differences across
/// the edges paired back through the adjugate of the dual-edge Hodge map.
///
/// The constrained system `[K, −Dᵀ; −D, 0]` is assembled as one scalar field map over the free
/// fluxes and the pressures (the first face's pressure the gauge, its balance implied by the
/// others) and solved by sparse LU: gax has no block field maps.
pub fn stokes(
    operators: &Operators,
    viscosity: f64,
    prescribed: &Field<Scalar>,
    fixed: &[usize],
    circulation: &Field<Patch>,
) -> Flow {
    let mesh = &operators.mesh;
    let viscous = operators
        .star1
        .adjugate()
        .compose(&operators.d0_vorticity.compose(&operators.curl))
        .map_cells(|c| c * viscosity);
    let boundary_vorticity = operators.inverse_star0.of(circulation);
    let boundary_resistance = operators
        .star1
        .adjugate()
        .of(&operators.d0_vorticity.of(&boundary_vorticity))
        .map(|a| a * viscosity);
    let edges = mesh.len(1);
    let free: Vec<usize> = (0..edges).filter(|e| !fixed.contains(e)).collect();
    let embedding =
        (&crate::simplicial::reading::<Scalar>(&free, edges) * Scalar::slot()).transpose();
    let stiffness = embedding
        .adjugate()
        .compose(&viscous.compose(&embedding))
        .map_cells(|c| c.undual());
    let divergence = operators.d1.compose(&embedding);
    let force = -&embedding
        .adjugate()
        .of(&(&viscous.of(prescribed) + &boundary_resistance))
        .map(|a| a.undual());
    let balance = -&operators.d1.of(prescribed);

    // One scalar map over the free fluxes and every pressure but the first face's.
    let n = free.len();
    let faces = mesh.len(2);
    let number = |c: ScalarMap| c.of(Scalar::new(1.0)).s();
    let mut triples: Vec<(usize, usize, ScalarMap)> = stiffness.couplings().collect();
    for (r, c, cell) in divergence.couplings().filter(|&(r, _, _)| r > 0) {
        let minus = Scalar::slot() * -number(cell);
        triples.push((n + r - 1, c, minus));
        triples.push((c, n + r - 1, minus));
    }
    let system = SparseMap::from_triples(triples, (n + faces - 1, n + faces - 1));
    let rhs: Field<Scalar> = force
        .iter()
        .chain(balance.iter().skip(1).map(|b| -b))
        .collect();
    let solved = system.solve_direct(&rhs);
    let free_flux: Field<Scalar> = solved.iter().take(n).collect();
    let flux = prescribed + &embedding.of(&free_flux);
    let pressure: Field<Scalar> = core::iter::once(Scalar::new(0.0))
        .chain(solved.iter().skip(n))
        .collect();
    let total: f64 = operators.areas.iter().map(|a| a.s()).sum();
    let mean = pressure.dot(&operators.areas) / total;
    let pressure = pressure.map(|p| p - Scalar::new(mean));
    let vorticity = &operators.curl.of(&flux) + &boundary_vorticity;
    // The first moment of the outward flux gives the face's mean velocity.
    let facing = boundary(mesh, 2);
    let velocity = Field::from_fn(faces, |f| {
        let twice = 2.0 * operators.areas.site(f).s();
        (0..3)
            .map(|k| {
                let e = facing.sites[f][k];
                let outward = flux.site(e).s() * f64::from(facing.signs[f][k]) * operators.signs[e];
                (operators.corners[k].site(f) - operators.face_centres.site(f)) * outward
            })
            .fold(Point::zero(), |a, b| a + b)
            * (-1.0 / twice)
    });
    Flow {
        flux,
        pressure,
        vorticity,
        velocity,
    }
}

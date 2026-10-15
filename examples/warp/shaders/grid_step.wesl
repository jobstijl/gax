// One step of the warped-space lattice: every node runs the traced `grid_node` kernel (springs
// to its neighbours and its rest position, the traced `lattice_rest`), with external forces
// from the traced `source_force` (wells pull, blasts push). The border is pinned.
import package::common::{Node, Source};
import package::fused::{grid_node, lattice_rest, add_source_force};
import gax::pga2d::point_new;

struct Grid {
    // Columns, rows, sources, unused.
    dims: vec4<u32>,
    // Spring stiffness, anchor stiffness, damping, dt: the `k` of `grid_node`.
    k: vec4<f32>,
    // The lower-left node's rest position.
    origin: gax::pga2d::Point,
    // The distance between nodes, unused, unused, unused.
    spacing: vec4<f32>,
}

struct Sources {
    s: array<Source, 64>,
}

@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<uniform> sources: Sources;
@group(0) @binding(2) var<storage, read> src: array<Node>;
@group(0) @binding(3) var<storage, read_write> dst: array<Node>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = grid.dims.x;
    let h = grid.dims.y;
    if (id.x >= w * h) {
        return;
    }
    let i = id.x % w;
    let j = id.x / w;
    let rest = lattice_rest(grid.origin, array<f32, 3>(f32(i), f32(j), grid.spacing.x));
    if (i == 0u || j == 0u || i == w - 1u || j == h - 1u) {
        dst[id.x] = Node(rest, point_new(0.0, 0.0, 0.0));
        return;
    }
    let node = src[id.x];
    var f = point_new(0.0, 0.0, 0.0);
    for (var s = 0u; s < grid.dims.z; s++) {
        let q = sources.s[s];
        f = add_source_force(f, node.p, q.at, array<f32, 2>(q.k.x, q.k.y));
    }
    let n = array<gax::pga2d::Point, 4>(src[id.x - 1u].p, src[id.x + 1u].p, src[id.x - w].p, src[id.x + w].p);
    let r = grid_node(node.p, node.v, rest, n, f, array<f32, 4>(grid.k.x, grid.k.y, grid.k.z, grid.k.w));
    dst[id.x] = Node(r.f0, r.f1);
}

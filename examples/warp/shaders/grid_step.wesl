// One step of the warped-space lattice: every node runs the traced `grid_node` kernel (springs
// to its neighbours and its rest position), with external forces from the traced
// `source_force` (blasts push, wells pull). The border is pinned.
import package::fused::{grid_node, source_force};
import gax::pga2d::{Point, point_new};

struct Node {
    p: Point,
    v: Point,
}

struct Grid {
    // Columns, rows, sources, unused.
    dims: vec4<u32>,
    // Spring stiffness, rest length, anchor stiffness, damping.
    k: vec4<f32>,
    // Origin x, origin y, spacing, dt.
    frame: vec4<f32>,
}

struct Sources {
    // x, y, strength, radius².
    s: array<vec4<f32>, 64>,
}

@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<uniform> sources: Sources;
@group(0) @binding(2) var<storage, read> src: array<Node>;
@group(0) @binding(3) var<storage, read_write> dst: array<Node>;

fn rest(i: u32, j: u32) -> Point {
    return point_new(grid.frame.x + f32(i) * grid.frame.z, grid.frame.y + f32(j) * grid.frame.z, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = grid.dims.x;
    let h = grid.dims.y;
    if (id.x >= w * h) {
        return;
    }
    let i = id.x % w;
    let j = id.x / w;
    let node = src[id.x];
    if (i == 0u || j == 0u || i == w - 1u || j == h - 1u) {
        dst[id.x] = Node(rest(i, j), point_new(0.0, 0.0, 0.0));
        return;
    }
    var f = point_new(0.0, 0.0, 0.0);
    for (var s = 0u; s < grid.dims.z; s++) {
        let q = sources.s[s];
        let a = source_force(node.p, point_new(q.x, q.y, 1.0), array<f32, 2>(q.z, q.w));
        f = Point(f.c0 + a.c0);
    }
    let n = array<Point, 4>(src[id.x - 1u].p, src[id.x + 1u].p, src[id.x - w].p, src[id.x + w].p);
    let r = grid_node(node.p, node.v, rest(i, j), n, f, array<f32, 5>(grid.k.x, grid.k.y, grid.k.z, grid.k.w, grid.frame.w));
    dst[id.x] = Node(r.f0, r.f1);
}

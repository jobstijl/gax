//! The closed-form logarithm and the plain sandwiches in 8D (four invariant planes, a quartic;
//! docs/log6d.md), on an algebra declared with `algebra!`: plane-based PGA of 7D Euclidean space,
//! R(7,0,1).

mod common;

gax::algebra! {
    algebra pga7d "Plane-based PGA of 7D Euclidean space, R(7,0,1).";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, e4, e5, e6, e7];
    kind Bivector = [e01, e02, e12, e03, e13, e23, e04, e14, e24, e34, e05, e15, e25, e35, e45, e06, e16, e26, e36, e46, e56, e07, e17, e27, e37, e47, e57, e67];
    versor Even = [1, e01, e02, e12, e03, e13, e23, e04, e14, e24, e34, e05, e15, e25, e35, e45, e06, e16, e26, e36, e46, e56, e07, e17, e27, e37, e47, e57, e67, e0123, e0124, e0134, e0234, e1234, e0125, e0135, e0235, e1235, e0145, e0245, e1245, e0345, e1345, e2345, e0126, e0136, e0236, e1236, e0146, e0246, e1246, e0346, e1346, e2346, e0156, e0256, e1256, e0356, e1356, e2356, e0456, e1456, e2456, e3456, e0127, e0137, e0237, e1237, e0147, e0247, e1247, e0347, e1347, e2347, e0157, e0257, e1257, e0357, e1357, e2357, e0457, e1457, e2457, e3457, e0167, e0267, e1267, e0367, e1367, e2367, e0467, e1467, e2467, e3467, e0567, e1567, e2567, e3567, e4567, e012345, e012346, e012356, e012456, e013456, e023456, e123456, e012347, e012357, e012457, e013457, e023457, e123457, e012367, e012467, e013467, e023467, e123467, e012567, e013567, e023567, e123567, e014567, e024567, e124567, e034567, e134567, e234567, e01234567];
}

// One basis plane per invariant plane (position in `Bivector`, rotation or not).
checks!(
    pga7d,
    checks,
    [(2, true), (9, true), (20, true), (21, false)]
);

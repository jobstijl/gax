//! The closed-form logarithm and the plain sandwiches in 8D (four invariant planes, a quartic;
//! docs/log6d.md), on an algebra declared with `algebra!`: Euclidean 8D, R(8,0): four rotation
//! planes.

mod common;

gax::algebra! {
    algebra r80 "Euclidean 8D, R(8,0): four rotation planes.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1, e8 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6, e7, e8];
    kind Bivector = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e18, e28, e38, e48, e58, e68, e78];
    versor Even = [1, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e18, e28, e38, e48, e58, e68, e78, e1234, e1235, e1245, e1345, e2345, e1236, e1246, e1346, e2346, e1256, e1356, e2356, e1456, e2456, e3456, e1237, e1247, e1347, e2347, e1257, e1357, e2357, e1457, e2457, e3457, e1267, e1367, e2367, e1467, e2467, e3467, e1567, e2567, e3567, e4567, e1238, e1248, e1348, e2348, e1258, e1358, e2358, e1458, e2458, e3458, e1268, e1368, e2368, e1468, e2468, e3468, e1568, e2568, e3568, e4568, e1278, e1378, e2378, e1478, e2478, e3478, e1578, e2578, e3578, e4578, e1678, e2678, e3678, e4678, e5678, e123456, e123457, e123467, e123567, e124567, e134567, e234567, e123458, e123468, e123568, e124568, e134568, e234568, e123478, e123578, e124578, e134578, e234578, e123678, e124678, e134678, e234678, e125678, e135678, e235678, e145678, e245678, e345678, e12345678];
}

// One basis plane per invariant plane (position in `Bivector`, rotation or not).
checks!(r80, checks, [(0, true), (5, true), (14, true), (27, true)]);

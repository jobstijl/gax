//! The closed-form logarithm and the plain sandwiches in 7D (three invariant planes, the cubic of
//! 6D; docs/log6d.md), on an algebra declared with `algebra!`: Euclidean 7D, R(7,0): three rotation
//! planes.

mod common;

gax::algebra! {
    algebra r70 "Euclidean 7D, R(7,0): three rotation planes.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6, e7];
    kind Bivector = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67];
    versor Even = [1, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e1234, e1235, e1245, e1345, e2345, e1236, e1246, e1346, e2346, e1256, e1356, e2356, e1456, e2456, e3456, e1237, e1247, e1347, e2347, e1257, e1357, e2357, e1457, e2457, e3457, e1267, e1367, e2367, e1467, e2467, e3467, e1567, e2567, e3567, e4567, e123456, e123457, e123467, e123567, e124567, e134567, e234567];
}

// One basis plane per invariant plane (position in `Bivector`, rotation or not).
checks!(r70, checks, [(0, true), (5, true), (14, true)]);
